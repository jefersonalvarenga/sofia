"""
Iris LangGraph subgraph (C8 / [EASAA-29](../../../EASAA/issues/EASAA-29)).

Multi-intent pipeline ([EASAA-140](../../../EASAA/issues/EASAA-140)):

    START → load_context → detect_intents
          → dispatch_specialists → aggregate_response
          → save_session → send_evolution → END

Reuses Sofia's `app.session.manager` so DNA/style and audit rows
(`sf_sessions`, `sf_agent_activations`) stay in one place. Routing goes
through `IrisRouterAgent` (C6) instead of `SofiaRouterAgent`. The router
emits one or more `{macro_state, scope_text}` intents per inbound message;
`dispatch_specialists` calls one specialist per intent (with the matching
`scope_text` as input) and `aggregate_response` consolidates the N specialist
replies into a single outbound message.

Scope as of EASAA-140: only `GREETING` has a real specialist
(`GreetingAgent`). Every other intent — including `UNCLASSIFIED` — falls
through to a deterministic `unknown_fallback` so the patient still sees a
reply. FAQ / Knowledge / Scheduler / Escalation specialists ship as
follow-ups ([EASAA-142](../../../EASAA/issues/EASAA-142),
[EASAA-143](../../../EASAA/issues/EASAA-143),
[EASAA-144](../../../EASAA/issues/EASAA-144),
[EASAA-145](../../../EASAA/issues/EASAA-145)).
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.greeting.agent import GreetingAgent
from app.agents.human_escalation.agent import HumanEscalationAgent
from app.agents.knowledge.agent import KnowledgeSpecialist
from app.agents.router.agent_iris import IRIS_ROUTER_MODEL, IrisRouterAgent
from app.agents.scheduler.agent import SchedulerAgent
from app.core.config import get_settings
from app.core.pricing import compute_cost
from app.core.telemetry import build_agent_run, extract_tokens_anthropic, log
from app.iris.evolution_client import (
    EvolutionAPIError,
    persist_outbound_message,
    send_text_message,
)
from app.iris.schemas import ParsedMessage
from app.session.manager import load_session, load_services_context, save_session
from services.iris.webhook import notify_receptionist


UNKNOWN_FALLBACK_TEXT = "Ainda estou aprendendo. Em breve te ajudo melhor 😊"


class IrisState(TypedDict, total=False):
    """LangGraph state for the Iris pipeline.

    Structurally compatible with the keys `app.session.manager.save_session`
    reads, so we can hand the same dict to it without a bridge type.
    """

    # ---- Inputs from webhook (C7) ----
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

    # ---- Fan-out outputs (one entry per intent, same order) ----
    specialist_responses: List[Dict[str, str]]

    # ---- Outputs ----
    agent_runs: List[Dict[str, Any]]
    response_text: Optional[str]
    outbound_wamid: Optional[str]
    routing_hint: Optional[str]


# Singletons — keep one instance per agent to avoid per-message client churn.
_router_agent = IrisRouterAgent()
_greeting_agent = GreetingAgent()
_knowledge_agent = KnowledgeSpecialist()
_scheduler_agent = SchedulerAgent()
_escalation_agent = HumanEscalationAgent()


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
        # Graceful fallback so the pipeline still produces a fallback reply
        # instead of crashing in the background task.
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
    """Classify the latest message via IrisRouterAgent (Anthropic + tool use)."""
    sofia_version = get_settings().sofia_version

    def _call() -> Dict[str, Any]:
        result = _router_agent.forward(
            latest_message=state["message"],
            history=state.get("history", []),
            conversation_stage=state.get("conversation_stage", "new"),
        )
        # The router doesn't speak to the patient; we keep its decision in
        # `data` so build_agent_run() captures it for the audit row.
        return {
            "messages": [],
            "conversation_stage": state.get("conversation_stage", "new"),
            "reasoning": result.get("reasoning", ""),
            "data": {
                "type": "router",
                "intents": result.get("intents", []),
                "detected_intents": result.get("detected_intents", []),
                "language": result.get("language", "pt-BR"),
                "confidence": result.get("confidence", 0.0),
            },
        }

    run = build_agent_run(
        agent_name="IrisRouterAgent",
        reason="iris.detect_intents",
        trace_id=state.get("trace_id", ""),
        clinic_id=state.get("clinic_id", ""),
        session_id=state.get("session_id", ""),
        language="pt-BR",
        sofia_version=sofia_version,
        call=_call,
    )

    # IrisRouterAgent uses the Anthropic SDK directly. build_agent_run()
    # only knows how to read DSPy LM history, so we patch in real token
    # usage from the router's stashed last_response and recompute the cost
    # against the Anthropic pricing table.
    tokens = extract_tokens_anthropic(_router_agent.last_response)
    run["prompt_tokens"] = tokens["prompt_tokens"]
    run["completion_tokens"] = tokens["completion_tokens"]
    run["total_tokens"] = tokens["total_tokens"]
    run["cost_usd"] = str(
        compute_cost(
            IRIS_ROUTER_MODEL,
            tokens["prompt_tokens"],
            tokens["completion_tokens"],
        )
    )

    data = run.get("data") or {}
    intents = data.get("intents") or [
        {"macro_state": "UNCLASSIFIED", "scope_text": state.get("message", "")}
    ]
    detected_intents = data.get("detected_intents") or [i["macro_state"] for i in intents]
    language = data.get("language", "pt-BR")
    primary_intent = detected_intents[-1] if detected_intents else "UNCLASSIFIED"

    log.info(
        "iris.node.detect_intents.ok",
        trace_id=state.get("trace_id"),
        intents=intents,
        primary_intent=primary_intent,
        language=language,
        confidence=data.get("confidence", 0.0),
    )

    return {
        "agent_runs": [run],
        "intents": intents,
        "detected_intents": detected_intents,
        "language": language,
        "primary_intent": primary_intent,
        "router_reasoning": run.get("reasoning", "") or "",
        "router_confidence": data.get("confidence", 0.0),
    }


# ----------------------------------------------------------------------------
# Specialist registry
# ----------------------------------------------------------------------------
#
# Each entry maps a macro_state to a callable `(state, scope_text) -> run dict`
# that mirrors `build_agent_run`'s `_call` signature. Unknown macro_states fall
# back to `_call_unknown_fallback`. Specialists added in follow-up stories
# (FAQ / Knowledge / Scheduler / Escalation) register here.


def _call_greeting(state: IrisState, scope_text: str) -> Dict[str, Any]:
    patient_name = (
        state.get("patient_name")
        or state.get("push_name")
        or "Paciente"
    )
    clinic_style = state.get("clinic_style") or {}
    return _greeting_agent.forward(
        patient_name=patient_name,
        clinic_name=state.get("clinic_name", "Clínica"),
        assistant_name=state.get("assistant_name", "Iris"),
        history_length=len(state.get("history", [])),
        greeting_example=clinic_style.get("greeting_example", ""),
    )


def _call_knowledge(state: IrisState, scope_text: str) -> Dict[str, Any]:
    return _knowledge_agent.forward(
        question=scope_text,
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


def _extract_service_names(services_context: str) -> List[str]:
    try:
        ctx = json.loads(services_context)
        return [s.get("name", "") for s in ctx.get("services", []) if s.get("name")]
    except Exception:
        return []


def _call_scheduler(state: IrisState, scope_text: str) -> Dict[str, Any]:
    current_stage = state.get("conversation_stage", "new")
    if current_stage not in {"collecting_service", "presenting_slots", "booked"}:
        current_stage = "collecting_service"

    services_ctx = state.get("services_context") or load_services_context(state.get("clinic_id", ""))
    service_names = _extract_service_names(services_ctx)

    return _scheduler_agent.forward(
        patient_message=scope_text or state.get("message", ""),
        history=state.get("history", []),
        available_slots=state.get("available_slots", []),
        clinic_name=state.get("clinic_name", "Clínica"),
        patient_name=state.get("patient_name") or state.get("push_name") or "Paciente",
        stage=current_stage,
        services_list=service_names,
    )


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


SPECIALIST_REGISTRY: Dict[str, tuple[str, Callable[[IrisState, str], Dict[str, Any]]]] = {
    "GREETING": ("GreetingAgent", _call_greeting),
    "FAQ": ("KnowledgeSpecialist", _call_knowledge),
    "SCHEDULE": ("Scheduler", _call_scheduler),
    "HUMAN_ESCALATION": ("HumanEscalation", _call_escalation),
}


def node_dispatch_specialists(state: IrisState) -> Dict[str, Any]:
    """Fan out: call one specialist per detected intent with its scope_text.

    Specialist order matches `intents` order (informational → CTA last).
    Each specialist call produces one `agent_run` row; the resulting text is
    captured into `specialist_responses` and consolidated in
    `node_aggregate_response`.
    """
    sofia_version = get_settings().sofia_version
    intents = state.get("intents") or []
    if not intents:
        intents = [
            {"macro_state": "UNCLASSIFIED", "scope_text": state.get("message", "")}
        ]

    runs: List[Dict[str, Any]] = []
    responses: List[Dict[str, str]] = []

    for intent in intents:
        macro = intent.get("macro_state") or "UNCLASSIFIED"
        scope = intent.get("scope_text") or state.get("message", "")
        agent_name, caller = SPECIALIST_REGISTRY.get(
            macro, ("UnknownFallback", _call_unknown_fallback)
        )

        if agent_name == "UnknownFallback":
            log.info(
                "iris.node.unknown_fallback",
                trace_id=state.get("trace_id"),
                clinic_id=state.get("clinic_id"),
                node_name="unknown_fallback",
                macro_state=macro,
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

        text: Optional[str] = None
        for msg in run.get("messages", []):
            if msg.get("type") == "text" and msg.get("content"):
                text = msg["content"]
                break
        responses.append(
            {
                "macro_state": macro,
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

    return {
        "agent_runs": [*state.get("agent_runs", []), *runs],
        "specialist_responses": responses,
        "routing_hint": routing_hint,
    }


def _consolidate(responses: List[Dict[str, str]]) -> Optional[str]:
    """Consolidate specialist replies into one outbound message.

    Keeps the router's informational → CTA ordering, drops empties, and joins
    with paragraph breaks. With a single response we pass it through unchanged
    so the existing single-intent UX is preserved. Multi-intent currently uses
    plain concatenation by paragraph; a higher-quality LLM-driven rewrite
    lands once the real specialists ship.
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
        macro_states=[r.get("macro_state") for r in responses],
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
        # Fall back to a synthetic id so the audit row still persists.
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
# Public dispatcher — kept stable for the C7 webhook handler
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
        "outbound_wamid": result.get("outbound_wamid"),
    }
