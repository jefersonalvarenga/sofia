"""
Iris LangGraph subgraph (C8 / [EASAA-29](../../../EASAA/issues/EASAA-29)).

Linear pipeline for the greeting smoke:

    START → load_context → detect_intents → execute_greeting → save_session → send_evolution → END

Reuses Sofia's `app.session.manager` so DNA/style and audit rows
(`sf_sessions`, `sf_agent_activations`) stay in one place. Routing goes
through `IrisRouterAgent` (C6) instead of `SofiaRouterAgent`. Greeting
delivery uses the deterministic `GreetingAgent`. Outbound delivery goes
through `app.iris.evolution_client` (C9).

Scope: only `GREETING` produces a real response. Every other intent —
including `UNCLASSIFIED` — falls through to a deterministic
`unknown_fallback` so the patient still sees a reply during the smoke.
FAQ / Scheduler / etc. ship in follow-up stories.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.greeting.agent import GreetingAgent
from app.agents.router.agent_iris import IrisRouterAgent
from app.core.config import get_settings
from app.core.telemetry import build_agent_run, extract_tokens_anthropic, log
from app.iris.evolution_client import (
    EvolutionAPIError,
    persist_outbound_message,
    send_text_message,
)
from app.iris.schemas import ParsedMessage
from app.session.manager import load_session, save_session


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

    # ---- Routing ----
    detected_intents: List[str]
    language: str
    primary_intent: str
    router_reasoning: str
    router_confidence: float

    # ---- Outputs ----
    agent_runs: List[Dict[str, Any]]
    response_text: Optional[str]
    outbound_wamid: Optional[str]


# Singletons — IrisRouterAgent owns an Anthropic client; GreetingAgent is
# stateless. Keep one of each to avoid per-message client churn.
_router_agent = IrisRouterAgent()
_greeting_agent = GreetingAgent()


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
        log.info(
            "iris.node.load_context.ok",
            trace_id=state.get("trace_id"),
            session_id=ctx.get("session_id"),
            history_length=len(ctx.get("history", [])),
            conversation_stage=ctx.get("conversation_stage"),
        )
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
    # usage from the router's stashed last_response.
    tokens = extract_tokens_anthropic(_router_agent.last_response)
    run["prompt_tokens"] = tokens["prompt_tokens"]
    run["completion_tokens"] = tokens["completion_tokens"]
    run["total_tokens"] = tokens["total_tokens"]

    data = run.get("data") or {}
    detected_intents = data.get("detected_intents") or ["UNCLASSIFIED"]
    language = data.get("language", "pt-BR")
    primary_intent = detected_intents[-1]

    log.info(
        "iris.node.detect_intents.ok",
        trace_id=state.get("trace_id"),
        detected_intents=detected_intents,
        primary_intent=primary_intent,
        language=language,
        confidence=data.get("confidence", 0.0),
    )

    return {
        "agent_runs": [run],
        "detected_intents": detected_intents,
        "language": language,
        "primary_intent": primary_intent,
        "router_reasoning": run.get("reasoning", "") or "",
        "router_confidence": data.get("confidence", 0.0),
    }


def node_execute_greeting(state: IrisState) -> Dict[str, Any]:
    """Dispatch GREETING → GreetingAgent, otherwise emit unknown_fallback."""
    primary_intent = state.get("primary_intent", "UNCLASSIFIED")
    sofia_version = get_settings().sofia_version

    if primary_intent == "GREETING":
        def _call_greeting() -> Dict[str, Any]:
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

        run = build_agent_run(
            agent_name="GreetingAgent",
            reason="greeting_detected",
            trace_id=state.get("trace_id", ""),
            clinic_id=state.get("clinic_id", ""),
            session_id=state.get("session_id", ""),
            language=state.get("language", "pt-BR"),
            sofia_version=sofia_version,
            call=_call_greeting,
        )
    else:
        # Out-of-scope intent for the greeting smoke. Log a node_logs-style
        # structured entry with `node_name=unknown_fallback` so observability
        # surfaces this branch without a dedicated table.
        log.info(
            "iris.node.unknown_fallback",
            trace_id=state.get("trace_id"),
            clinic_id=state.get("clinic_id"),
            node_name="unknown_fallback",
            primary_intent=primary_intent,
            detected_intents=state.get("detected_intents", []),
        )

        def _call_fallback() -> Dict[str, Any]:
            return {
                "messages": [{"type": "text", "content": UNKNOWN_FALLBACK_TEXT}],
                "conversation_stage": state.get("conversation_stage", "new"),
                "reasoning": (
                    f"Out-of-scope intent={primary_intent}. Iris greeting smoke "
                    "only handles GREETING; deterministic fallback returned."
                ),
                "data": None,
            }

        run = build_agent_run(
            agent_name="UnknownFallback",
            reason="unknown_fallback",
            trace_id=state.get("trace_id", ""),
            clinic_id=state.get("clinic_id", ""),
            session_id=state.get("session_id", ""),
            language=state.get("language", "pt-BR"),
            sofia_version=sofia_version,
            call=_call_fallback,
        )

    response_text: Optional[str] = None
    for msg in run.get("messages", []):
        if msg.get("type") == "text" and msg.get("content"):
            response_text = msg["content"]
            break

    return {
        "agent_runs": [*state.get("agent_runs", []), run],
        "response_text": response_text,
    }


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
workflow.add_node("execute_greeting", node_execute_greeting)
workflow.add_node("save_session", node_save_session)
workflow.add_node("send_evolution", node_send_evolution)

workflow.set_entry_point("load_context")
workflow.add_edge("load_context", "detect_intents")
workflow.add_edge("detect_intents", "execute_greeting")
workflow.add_edge("execute_greeting", "save_session")
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
        "detected_intents": result.get("detected_intents", []),
        "primary_intent": result.get("primary_intent"),
        "language": result.get("language", "pt-BR"),
        "outbound_wamid": result.get("outbound_wamid"),
    }
