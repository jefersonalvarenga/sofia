"""
Iris LangGraph subgraph — multi-agent pipeline (v2).

    START → load_context → detect_intents
          → dispatch_specialists → aggregate_response
          → save_session → send_evolution → END

Replaces the legacy IrisRouterAgent/SofiaRouterAgent + KnowledgeSpecialist
Anthropic stack with the new agents built in the May 2026 sprint:

  - GreetingAgent          (deepseek-v4-flash, JSON output, v26 schema)
  - RouterAgent            (deepseek-v4-flash, 8 intents + scope_text)
  - ScheduleRouter         (deepseek-v4-flash, 11 sub-intents)
  - KnowledgeAgent         (deepseek-v4-pro, RAG via pgvector)
  - HumanEscalationAgent   (unchanged)

Routing model:
  - RouterAgent emits {intent, scope_text} list (informational → CTA → terminal)
  - dispatch_specialists fans out: one specialist call per intent
  - SCHEDULE is a guard-umbrella — when seen, we run the ScheduleRouter sub-step
    with a mocked sequence and fall back to UNKNOWN for whichever sub-intent
    it returns (sub-agents land in follow-up PRs)
  - aggregate_response joins specialist replies in order
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.greeting.agent import GreetingAgent
from app.agents.human_escalation.agent import HumanEscalationAgent
from app.agents.knowledge.agent import KnowledgeAgent
from app.agents.router.agent import RouterAgent
from app.agents.router.schedule_router import ScheduleRouter
from app.agents.schedule_intake.agent import ScheduleIntakeAgent
from app.agents.scheduler.agent import SchedulerAgent
from app.core.config import get_settings
from app.core.supabase_client import get_supabase
from app.core.telemetry import build_agent_run, log
from app.repositories.intake_questions import load_intake_questions
from app.iris.evolution_client import (
    EvolutionAPIError,
    persist_outbound_message,
    send_text_message,
)
from app.iris.schemas import ParsedMessage
from app.session.manager import load_session, load_services_context, save_session
from services.iris.webhook import notify_receptionist


UNKNOWN_FALLBACK_TEXT = "Ainda estou aprendendo. Em breve te ajudo melhor 😊"

# Mocked schedule sequences. Pulled from the schedule-router spec — the upstream
# Manager agent decides which sequence to activate per session in production.
# For now we hardcode the evaluation flow as the default.
SCHEDULE_SEQUENCE_EVALUATION = [
    "SCHEDULE_INTAKE",
    "SCHEDULE_CASHIER",
    "SCHEDULE_EVALUATION",
    "SCHEDULE_COMPLETION",
]
SCHEDULE_SEQUENCE_SERVICE = [
    "SCHEDULE_CASHIER",
    "SCHEDULE_SERVICE",
    "SCHEDULE_SERVICE_PROTOCOL",
    "SCHEDULE_COMPLETION",
]
SCHEDULE_SEQUENCE_CONFIRMATION = ["SCHEDULE_CONFIRMATION", "SCHEDULE_COMPLETION"]
SCHEDULE_SEQUENCE_REMINDER = ["SCHEDULE_REMINDER", "SCHEDULE_COMPLETION"]

# Fallback greeting few-shot when the clinic hasn't configured one yet.
DEFAULT_GREETING_FEW_SHOT = "Olá! Como posso te ajudar?"


class IrisState(TypedDict, total=False):
    """LangGraph state for the Iris pipeline.

    Structurally compatible with the keys `app.session.manager.save_session`
    reads, so we can hand the same dict to it without a bridge type.
    """

    # ---- Inputs from webhook ----
    message_id: str
    instance_name: str
    instance_id: str
    clinic_id: str
    remote_jid: str
    push_name: Optional[str]
    message: str
    message_type: str
    wamid: str
    trace_id: str

    # ---- Loaded context (load_session) ----
    session_id: str
    clinic_name: str
    assistant_name: str
    history: List[Dict[str, str]]
    conversation_stage: str
    conversation_type: str
    patient_name: Optional[str]
    customer_id: Optional[str]
    clinic_style: Optional[Dict[str, Any]]
    attribution_id: Optional[str]
    paused: bool

    # ---- Scheduling ----
    available_slots: List[str]
    services_context: str

    # ---- Routing ----
    intents: List[Dict[str, str]]
    detected_intents: List[str]
    language: str
    primary_intent: str
    router_reasoning: str
    router_confidence: float

    # ---- Schedule sub-routing (populated when SCHEDULE in detected_intents) ----
    schedule_sub_intent: Optional[str]
    schedule_is_deviation: bool
    schedule_session_data: List[Dict[str, Any]]

    # ---- Fan-out outputs (one entry per intent, same order) ----
    specialist_responses: List[Dict[str, str]]

    # ---- Outputs ----
    agent_runs: List[Dict[str, Any]]
    response_text: Optional[str]
    outbound_wamid: Optional[str]
    routing_hint: Optional[str]


# Singletons — keep one instance per agent to avoid per-message client churn.
_router_agent = RouterAgent()
_schedule_router = ScheduleRouter()
_schedule_intake_agent = ScheduleIntakeAgent()
_greeting_agent = GreetingAgent()
_knowledge_agent = KnowledgeAgent()
_scheduler_agent = SchedulerAgent()
_escalation_agent = HumanEscalationAgent()


# ============================================================================
# Helpers
# ============================================================================

def resolve_schedule_sequence(state: IrisState) -> List[str]:
    """Decide which schedule sub-flow this session is in.

    MVP placeholder: always returns the evaluation sequence. A real Manager
    agent will replace this in a follow-up PR — it will read trajectory state
    (last interaction outcome, service of interest, whether a reminder is due,
    etc.) and pick the appropriate sequence.

    Hooks already exposed in state so the swap is mechanical:
      - state["conversation_stage"]  → likely main signal
      - state["history"]             → trajectory
      - state["clinic_style"]        → clinic-specific defaults
    """
    return SCHEDULE_SEQUENCE_EVALUATION


def _extract_service_names(services_context: str) -> List[str]:
    try:
        ctx = json.loads(services_context)
        return [s.get("name", "") for s in ctx.get("services", []) if s.get("name")]
    except Exception:
        return []


def _resolve_few_shot(state: IrisState) -> str:
    """Pull the greeting few-shot from clinic_style with a sensible fallback."""
    clinic_style = state.get("clinic_style") or {}
    example = (clinic_style.get("greeting_example") or "").strip()
    return example or DEFAULT_GREETING_FEW_SHOT


# ============================================================================
# Nodes
# ============================================================================

def node_load_context(state: IrisState) -> Dict[str, Any]:
    """Resolve session / customer / clinic context via load_session()."""
    log.info(
        "iris.node.load_context",
        trace_id=state.get("trace_id"),
        clinic_id=state.get("clinic_id"),
        remote_jid=state.get("remote_jid"),
    )
    try:
        ctx = load_session(
            remote_jid=state["remote_jid"],
            clinic_id=state["clinic_id"],
            push_name=state.get("push_name"),
            instance_id=state.get("instance_name", ""),
        )
        is_paused = bool(ctx.get("paused", False))
        log.info(
            "iris.node.load_context.ok",
            trace_id=state.get("trace_id"),
            session_id=ctx.get("session_id"),
            history_length=len(ctx.get("history", [])),
            conversation_stage=ctx.get("conversation_stage"),
            paused=is_paused,
        )
        ctx["paused"] = is_paused
        return ctx
    except Exception as exc:
        log.error(
            "iris.node.load_context.error",
            trace_id=state.get("trace_id"),
            error=str(exc),
        )
        return {
            "session_id": f"{state['remote_jid']}:{state['clinic_id']}",
            "customer_id": None,
            "history": [],
            "conversation_stage": "new",
            "conversation_type": "first_contact",
            "patient_name": state.get("push_name"),
            "clinic_name": "Clínica",
            "assistant_name": "Iris",
            "clinic_style": None,
            "paused": False,
        }


def node_detect_intents(state: IrisState) -> Dict[str, Any]:
    """Classify the latest message via RouterAgent (deepseek-v4-flash).

    Propagates the exception path (no silent fallback). If the router fails,
    detected_intents will be empty and dispatch_specialists routes everything
    to the deterministic UNKNOWN fallback.
    """
    sofia_version = get_settings().sofia_version

    def _call_router() -> Dict[str, Any]:
        result = _router_agent.forward(
            latest_message=state["message"],
            history=state.get("history", []),
            conversation_stage=state.get("conversation_stage", "new"),
        )
        return {
            "messages": [],
            "conversation_stage": state.get("conversation_stage", "new"),
            "reasoning": result.get("reasoning", ""),
            "data": {
                "type": "router",
                "intents": result.get("intents", []),
                "detected_intents": result.get("detected_intents", []),
                "confidence": result.get("confidence", 0.0),
            },
        }

    run = build_agent_run(
        agent_name="RouterAgent",
        reason="iris.detect_intents",
        trace_id=state.get("trace_id", ""),
        clinic_id=state.get("clinic_id", ""),
        session_id=state.get("session_id", ""),
        language="pt-BR",
        sofia_version=sofia_version,
        call=_call_router,
    )

    data = run.get("data") or {}
    intents = data.get("intents") or [
        {"intent": "UNCLASSIFIED", "scope_text": state.get("message", "")}
    ]
    detected_intents = data.get("detected_intents") or [i["intent"] for i in intents]
    primary_intent = detected_intents[-1] if detected_intents else "UNCLASSIFIED"

    log.info(
        "iris.node.detect_intents.ok",
        trace_id=state.get("trace_id"),
        intents=intents,
        primary_intent=primary_intent,
        confidence=data.get("confidence", 0.0),
    )

    return {
        "agent_runs": [run],
        "intents": intents,
        "detected_intents": detected_intents,
        "language": "pt-BR",
        "primary_intent": primary_intent,
        "router_reasoning": run.get("reasoning", "") or "",
        "router_confidence": data.get("confidence", 0.0),
    }


# ============================================================================
# Specialist registry
# ============================================================================
#
# Each entry maps an intent value to a (agent_name, callable) tuple where the
# callable is `(state, scope_text) -> agent_run-like dict`. Unknown intents
# fall back to `_call_unknown_fallback`.

def _call_greeting(state: IrisState, scope_text: str) -> Dict[str, Any]:
    patient_name = (
        state.get("patient_name")
        or state.get("push_name")
        or None
    )
    if patient_name == "Paciente":
        patient_name = None

    return _greeting_agent.forward(
        patient_message=scope_text or state.get("message", ""),
        patient_intents=[],
        patient_name=patient_name,
        clinic_name=state.get("clinic_name", "Clínica"),
        assistant_name=state.get("assistant_name", "Iris"),
        few_shot=_resolve_few_shot(state),
        session_summary="",
        recent_relevant_messages=state.get("history", []),
        time_gap_hours=None,
    )


def _call_knowledge(state: IrisState, scope_text: str) -> Dict[str, Any]:
    return _knowledge_agent.forward(
        question=scope_text or state.get("message", ""),
        clinic_name=state.get("clinic_name", "Clínica"),
        tenant_id=state.get("clinic_id", ""),
        history=state.get("history"),
    )


def _call_escalation(state: IrisState, scope_text: str) -> Dict[str, Any]:
    patient_name = (
        state.get("patient_name")
        or state.get("push_name")
        or "Paciente"
    )
    result = _escalation_agent.forward(
        patient_name=patient_name,
        assistant_name=state.get("assistant_name", "Iris"),
        clinic_name=state.get("clinic_name", "Clínica"),
    )
    notify_receptionist(
        tenant_id=state.get("clinic_id", ""),
        conversation_id=state.get("remote_jid", ""),
        patient_name=patient_name,
        trigger="explicit_request",
    )
    return result


def _ensure_evaluation_entry(
    session_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Guarantee an ``evaluation`` entry exists in ``session_data``.

    Cold-start case: ScheduleRouter just decided SCHEDULE_INTAKE but the
    upstream Manager (mocked) didn't seed an evaluation entry yet. We
    synthesize one with ``service=None`` so the intake agent gets a
    well-formed envelope — it short-circuits to completion when both
    ``service`` and ``questions`` are absent, which is the correct cold-start
    behaviour until Manager wires service detection.

    Returns a NEW list; never mutates the input.
    """
    result = list(session_data or [])
    if not any(e.get("name") == "evaluation" for e in result):
        result.append(
            {
                "name": "evaluation",
                "data": {"service": None, "intake_answers": []},
            }
        )
    return result


def _extract_evaluation_service(
    session_data: List[Dict[str, Any]],
) -> Optional[str]:
    """Pull the service-of-interest from the evaluation entry (may be None)."""
    for entry in session_data or []:
        if entry.get("name") == "evaluation":
            return (entry.get("data") or {}).get("service")
    return None


def _resolve_service_metadata(
    clinic_id: str,
    service_name: Optional[str],
    *,
    trace_id: str = "",
) -> tuple[Optional[str], List[str]]:
    """Resolve ``(service_id, contraindications)`` for ``service_name``.

    Strategy:
      - If ``service_name`` is falsy, return ``(None, [])``.
      - Query ``sf_clinic_services`` filtered by ``(clinic_id, name)``,
        ordered by ``created_at ASC``, ``limit 1``. Pull ``id`` and
        ``contraindications``.
      - On any error (network, missing column, etc.) return ``(None, [])``
        so the pipeline degrades gracefully — the agent receives an empty
        questions list when service can't be resolved, which keeps the
        conversation moving instead of crashing.

    No mutation of state. Pure I/O wrapper.
    """
    if not service_name:
        return None, []
    try:
        sb = get_supabase()
        rows = (
            sb.table("sf_clinic_services")
            .select("id, contraindications")
            .eq("clinic_id", clinic_id)
            .ilike("name", service_name)
            .order("created_at")
            .limit(1)
            .execute()
            .data
            or []
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "iris.schedule_intake.service_lookup_failed",
            trace_id=trace_id,
            clinic_id=clinic_id,
            service=service_name,
            error=str(exc),
        )
        return None, []

    if not rows:
        log.info(
            "iris.schedule_intake.service_not_found",
            trace_id=trace_id,
            clinic_id=clinic_id,
            service=service_name,
        )
        return None, []

    row = rows[0]
    contraindications = list(row.get("contraindications") or [])
    return row.get("id"), contraindications


def _call_schedule_intake(
    state: IrisState,
    *,
    scope_text: str,
    session_data: List[Dict[str, Any]],
    sub_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Dispatch SCHEDULE_INTAKE to ``ScheduleIntakeAgent``.

    Pre-loads ``questions`` (clinic baseline + service override) and
    ``contraindications`` per spec §11 / §15. Returns the standard envelope
    augmented with ``schedule_sub_intent`` / ``schedule_is_deviation`` /
    ``schedule_session_data`` so the dispatcher's state propagation keeps
    working.
    """
    clinic_id = state.get("clinic_id", "") or ""
    trace_id = state.get("trace_id", "") or ""

    # Ensure evaluation entry exists; pull service.
    session_data = _ensure_evaluation_entry(session_data)
    service_name = _extract_evaluation_service(session_data)

    # Resolve service metadata (id + contraindications). Either may be missing.
    service_id, contraindications = _resolve_service_metadata(
        clinic_id, service_name, trace_id=trace_id
    )

    # Load intake questions (baseline + optional service-specific union).
    try:
        questions = load_intake_questions(clinic_id, service_id)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "iris.schedule_intake.questions_load_failed",
            trace_id=trace_id,
            clinic_id=clinic_id,
            service_id=service_id,
            error=str(exc),
        )
        questions = []

    log.info(
        "iris.schedule_intake.invoke",
        trace_id=trace_id,
        clinic_id=clinic_id,
        service=service_name,
        service_id=service_id,
        questions_count=len(questions),
        contraindications_count=len(contraindications),
    )

    envelope = _schedule_intake_agent.forward(
        latest_message=scope_text or state.get("message", ""),
        history=state.get("history", []),
        session_data=session_data,
        clinic_id=clinic_id,
        service=service_name or "",
        questions=questions,
        contraindications=contraindications,
        clinic_name=state.get("clinic_name", "Clínica"),
        assistant_name=state.get("assistant_name", "Iris"),
    )

    # Surface the agent's session_data update into schedule_session_data so
    # the dispatcher can roll it back into state for the next turn.
    updated_session_data = (envelope.get("data") or {}).get("session_data") or session_data

    data = dict(envelope.get("data") or {})
    data["schedule_sub_intent"] = "SCHEDULE_INTAKE"
    data["schedule_is_deviation"] = bool(sub_result.get("is_deviation", False))
    data["schedule_session_data"] = updated_session_data
    data["schedule_confidence"] = sub_result.get("confidence", 0.0)
    data["schedule_reasoning"] = sub_result.get("reasoning", "")

    return {
        "messages": envelope.get("messages", []),
        "conversation_stage": envelope.get(
            "conversation_stage", state.get("conversation_stage", "schedule_intake")
        ),
        "reasoning": envelope.get("reasoning", ""),
        "data": data,
    }


def _call_scheduler(state: IrisState, scope_text: str) -> Dict[str, Any]:
    """Run the schedule sub-router, then dispatch to the sub-agent it picks.

    Wiring status:
      - ``SCHEDULE_INTAKE`` → :class:`ScheduleIntakeAgent` (wired here).
      - Other sub-intents (CASHIER, EVALUATION, SERVICE, COMPLETION,
        CONFIRMATION, REMINDER, CHANGE, CANCEL, FALLBACK) → deterministic
        UNKNOWN fallback. Sub-router run still produces a real agent_run for
        telemetry so we can audit sub-routing decisions in prod before each
        sub-agent ships.
    """
    sequence = resolve_schedule_sequence(state)
    current_stage = state.get("conversation_stage", "new") or "new"
    session_data = state.get("schedule_session_data") or []

    sub_result = _schedule_router.forward(
        latest_message=scope_text or state.get("message", ""),
        history=state.get("history", []),
        sequence=sequence,
        current_stage=current_stage,
        session_data=session_data,
    )

    next_intent = sub_result.get("next_intent", "SCHEDULE_FALLBACK")
    is_deviation = sub_result.get("is_deviation", False)

    log.info(
        "iris.schedule_router.decision",
        trace_id=state.get("trace_id"),
        next_intent=next_intent,
        is_deviation=is_deviation,
        sequence=sequence,
        current_stage=current_stage,
    )

    if next_intent == "SCHEDULE_INTAKE":
        try:
            return _call_schedule_intake(
                state,
                scope_text=scope_text,
                session_data=sub_result.get("session_data") or session_data,
                sub_result=sub_result,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "iris.schedule_intake.dispatch_failed",
                trace_id=state.get("trace_id"),
                error=str(exc),
            )
            # Fall through to UNKNOWN fallback below so the patient still
            # gets a reply rather than a 500.

    # No sub-agents wired yet for the remaining sub-intents → deterministic
    # placeholder so the patient still sees a reply. When each sub-agent
    # ships, add a branch above this point.
    return {
        "messages": [{"type": "text", "content": UNKNOWN_FALLBACK_TEXT}],
        "conversation_stage": state.get("conversation_stage", "new"),
        "reasoning": (
            f"ScheduleRouter chose {next_intent} (deviation={is_deviation}); "
            "no sub-agent registered yet — UNKNOWN fallback returned."
        ),
        "data": {
            "schedule_sub_intent": next_intent,
            "schedule_is_deviation": is_deviation,
            "schedule_session_data": sub_result.get("session_data", []),
            "schedule_confidence": sub_result.get("confidence", 0.0),
            "schedule_reasoning": sub_result.get("reasoning", ""),
        },
    }


def _call_reengage(state: IrisState, scope_text: str) -> Dict[str, Any]:
    patient_name = (
        state.get("patient_name")
        or state.get("push_name")
        or "Paciente"
    )
    name = patient_name if patient_name != "Paciente" else ""
    greeting = f"Olá, {name}!" if name else "Olá!"
    return {
        "messages": [
            {
                "type": "text",
                "content": (
                    f"{greeting} Que bom que você voltou! "
                    f"Como posso ajudar você hoje?"
                ),
            }
        ],
        "conversation_stage": state.get("conversation_stage", "active"),
        "reasoning": "Patient re-engaging — welcome back message.",
        "data": None,
    }


def _call_unknown_fallback(state: IrisState, scope_text: str) -> Dict[str, Any]:
    return {
        "messages": [{"type": "text", "content": UNKNOWN_FALLBACK_TEXT}],
        "conversation_stage": state.get("conversation_stage", "new"),
        "reasoning": (
            f"No specialist registered for scope={scope_text!r}; "
            "deterministic fallback returned."
        ),
        "data": None,
    }


# Vocabulary from app.agents.router.intents.IntentType (8 values).
# BUSINESS_INFO, TOPIC_KNOWLEDGE, INTAKE all currently route to the Knowledge
# agent as the closest existing specialist — INTAKE gets a dedicated agent in
# a follow-up PR; BUSINESS_INFO needs a small specialist that reads
# sf_clinic_services / business_rules (also follow-up).
SPECIALIST_REGISTRY: Dict[str, tuple[str, Callable[[IrisState, str], Dict[str, Any]]]] = {
    "GREETING": ("GreetingAgent", _call_greeting),
    "BUSINESS_INFO": ("KnowledgeAgent", _call_knowledge),
    "TOPIC_KNOWLEDGE": ("KnowledgeAgent", _call_knowledge),
    "INTAKE": ("KnowledgeAgent", _call_knowledge),
    "SCHEDULE": ("ScheduleRouter", _call_scheduler),
    "REENGAGE": ("ReEngage", _call_reengage),
    "HUMAN_ESCALATION": ("HumanEscalation", _call_escalation),
    "UNCLASSIFIED": ("UnknownFallback", _call_unknown_fallback),
}


def node_dispatch_specialists(state: IrisState) -> Dict[str, Any]:
    """Fan out: call one specialist per detected intent with its scope_text."""
    sofia_version = get_settings().sofia_version
    intents = state.get("intents") or []
    if not intents:
        intents = [{"intent": "UNCLASSIFIED", "scope_text": state.get("message", "")}]

    runs: List[Dict[str, Any]] = []
    responses: List[Dict[str, str]] = []
    schedule_sub_intent: Optional[str] = None
    schedule_is_deviation: bool = False
    schedule_session_data: List[Dict[str, Any]] = []

    for intent in intents:
        macro = intent.get("intent") or "UNCLASSIFIED"
        scope = intent.get("scope_text") or state.get("message", "")
        agent_name, caller = SPECIALIST_REGISTRY.get(
            macro, ("UnknownFallback", _call_unknown_fallback)
        )

        if agent_name == "UnknownFallback" and macro != "UNCLASSIFIED":
            log.info(
                "iris.node.unknown_fallback",
                trace_id=state.get("trace_id"),
                clinic_id=state.get("clinic_id"),
                node_name="unknown_fallback",
                intent=macro,
                scope_text=scope,
            )

        run = build_agent_run(
            agent_name=agent_name,
            reason=f"iris.dispatch.{macro.lower()}",
            trace_id=state.get("trace_id", ""),
            clinic_id=state.get("clinic_id", ""),
            session_id=state.get("session_id", ""),
            language=state.get("language", "pt-BR"),
            sofia_version=sofia_version,
            call=lambda c=caller, s=scope: c(state, s),
        )
        runs.append(run)

        # Capture schedule sub-router decision for state propagation.
        if agent_name == "ScheduleRouter":
            sub_data = run.get("data") or {}
            schedule_sub_intent = sub_data.get("schedule_sub_intent")
            schedule_is_deviation = bool(sub_data.get("schedule_is_deviation"))
            schedule_session_data = sub_data.get("schedule_session_data") or []

        text: Optional[str] = None
        for msg in run.get("messages", []):
            if msg.get("type") == "text" and msg.get("content"):
                text = msg["content"]
                break
        responses.append(
            {
                "intent": macro,
                "scope_text": scope,
                "response_text": text or "",
                "agent_name": agent_name,
            }
        )

    routing_hint: Optional[str] = None
    for run in runs:
        data = run.get("data") or {}
        if data.get("routing_hint"):
            routing_hint = data["routing_hint"]

    update: Dict[str, Any] = {
        "agent_runs": [*state.get("agent_runs", []), *runs],
        "specialist_responses": responses,
        "routing_hint": routing_hint,
    }
    if schedule_sub_intent is not None:
        update["schedule_sub_intent"] = schedule_sub_intent
        update["schedule_is_deviation"] = schedule_is_deviation
        update["schedule_session_data"] = schedule_session_data
    return update


def _consolidate(responses: List[Dict[str, str]]) -> Optional[str]:
    """Consolidate specialist replies into one outbound message.

    Keeps the router's informational → CTA ordering, drops empties, and joins
    with paragraph breaks. With a single response we pass it through unchanged.
    Multi-intent currently uses plain paragraph concatenation; an LLM-driven
    rewrite can come later once specialists ship.
    """
    texts = [r["response_text"].strip() for r in responses if r.get("response_text", "").strip()]
    if not texts:
        return None
    if len(texts) == 1:
        return texts[0]
    return "\n\n".join(texts)


def node_aggregate_response(state: IrisState) -> Dict[str, Any]:
    """Pick the outbound `response_text` from the fan-out responses."""
    responses = state.get("specialist_responses") or []
    response_text = _consolidate(responses)
    log.info(
        "iris.node.aggregate_response",
        trace_id=state.get("trace_id"),
        intent_count=len(responses),
        intents=[r.get("intent") for r in responses],
        empty=response_text is None,
    )
    return {"response_text": response_text}


def node_save_session(state: IrisState) -> Dict[str, Any]:
    """Persist sf_sessions + sf_agent_activations + agent data via save_session()."""
    log.info(
        "iris.node.save_session",
        trace_id=state.get("trace_id"),
        clinic_id=state.get("clinic_id"),
        session_id=state.get("session_id"),
        agents=[r.get("agent") for r in state.get("agent_runs", [])],
    )
    try:
        save_session(state)  # type: ignore[arg-type]
    except Exception as exc:
        log.error(
            "iris.node.save_session.error",
            trace_id=state.get("trace_id"),
            error=str(exc),
        )
    return {}


async def node_send_evolution(state: IrisState) -> Dict[str, Any]:
    """Send the response via Evolution API and persist the outbound row."""
    response_text = state.get("response_text")
    if not response_text:
        log.info(
            "iris.node.send_evolution.skipped",
            trace_id=state.get("trace_id"),
            reason="empty_response_text",
        )
        return {}

    settings = get_settings()
    instance_name = state.get("instance_name", "")
    api_key = settings.evolution_api_key
    if not api_key:
        log.error(
            "iris.node.send_evolution.skipped",
            trace_id=state.get("trace_id"),
            reason="missing_evolution_api_key",
        )
        return {}

    try:
        response = await send_text_message(
            instance=instance_name,
            remote_jid=state["remote_jid"],
            content=response_text,
            api_key=api_key,
        )
    except EvolutionAPIError as exc:
        log.error(
            "iris.node.send_evolution.error",
            trace_id=state.get("trace_id"),
            clinic_id=state.get("clinic_id"),
            error=str(exc),
            status=exc.status_code,
            attempts=exc.attempts,
        )
        return {}

    outbound_wamid = ((response or {}).get("key") or {}).get("id") or ""
    if not outbound_wamid:
        outbound_wamid = f"iris-out-{uuid.uuid4()}"

    try:
        persist_outbound_message(
            clinic_id=state["clinic_id"],
            session_id=state.get("session_id"),
            wamid=outbound_wamid,
            content=response_text,
        )
    except Exception as exc:
        log.error(
            "iris.node.send_evolution.persist_error",
            trace_id=state.get("trace_id"),
            wamid=outbound_wamid,
            error=str(exc),
        )

    log.info(
        "iris.node.send_evolution.ok",
        trace_id=state.get("trace_id"),
        clinic_id=state.get("clinic_id"),
        wamid=outbound_wamid,
    )
    return {"outbound_wamid": outbound_wamid}


# ============================================================================
# Graph construction — linear, no conditional edges
# ============================================================================

workflow = StateGraph(IrisState)
workflow.add_node("load_context", node_load_context)
workflow.add_node("detect_intents", node_detect_intents)
workflow.add_node("dispatch_specialists", node_dispatch_specialists)
workflow.add_node("aggregate_response", node_aggregate_response)
workflow.add_node("save_session", node_save_session)
workflow.add_node("send_evolution", node_send_evolution)

workflow.set_entry_point("load_context")
workflow.add_conditional_edges(
    "load_context",
    lambda state: END if state.get("paused") else "detect_intents",
    {"detect_intents": "detect_intents", END: END},
)
workflow.add_edge("detect_intents", "dispatch_specialists")
workflow.add_edge("dispatch_specialists", "aggregate_response")
workflow.add_edge("aggregate_response", "save_session")
workflow.add_edge("save_session", "send_evolution")
workflow.add_edge("send_evolution", END)

iris_graph = workflow.compile()


# ============================================================================
# Public dispatcher — kept stable for the webhook handler
# ============================================================================

async def invoke(
    *,
    clinic_id: str,
    message_id: str,
    parsed: ParsedMessage,
    trace_id: str,
) -> Dict[str, Any]:
    """Run the Iris pipeline for an inserted inbound message."""
    initial: IrisState = {
        "message_id": message_id,
        "instance_name": parsed.instance_name,
        "instance_id": parsed.instance_name,
        "clinic_id": clinic_id,
        "remote_jid": parsed.remote_jid,
        "push_name": parsed.push_name or None,
        "message": parsed.message_content,
        "message_type": parsed.message_type,
        "wamid": parsed.wamid,
        "trace_id": trace_id,
        "agent_runs": [],
    }

    log.info(
        "iris.pipeline.start",
        trace_id=trace_id,
        clinic_id=clinic_id,
        message_id=message_id,
        wamid=parsed.wamid,
        remote_jid=parsed.remote_jid,
    )

    result = await iris_graph.ainvoke(initial)

    agent_runs = result.get("agent_runs", [])
    log.info(
        "iris.pipeline.done",
        trace_id=trace_id,
        clinic_id=clinic_id,
        message_id=message_id,
        agents=[r.get("agent") for r in agent_runs],
        primary_intent=result.get("primary_intent"),
        outbound_wamid=result.get("outbound_wamid"),
    )

    return {
        "status": "ok",
        "message_id": message_id,
        "agent_runs": agent_runs,
        "intents": result.get("intents", []),
        "detected_intents": result.get("detected_intents", []),
        "primary_intent": result.get("primary_intent"),
        "language": result.get("language", "pt-BR"),
        "specialist_responses": result.get("specialist_responses", []),
        "schedule_sub_intent": result.get("schedule_sub_intent"),
        "schedule_is_deviation": result.get("schedule_is_deviation"),
        "outbound_wamid": result.get("outbound_wamid"),
    }
