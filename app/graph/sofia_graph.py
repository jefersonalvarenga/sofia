"""
Sofia LangGraph v2 — multi-intent conversational flow.

START → load_context → detect_intents → execute_agents → save_session → END

Agents run in priority order: informational first, CTA (most important) last.
Each agent produces an AgentRun with messages + data + observability fields.
"""

import json
from typing import Dict, Any, List

from langgraph.graph import StateGraph, END

from app.session.models import SofiaState
from app.session.manager import (
    load_session, save_session,
    load_services_context, load_business_rules,
)
from app.core.telemetry import build_agent_run, log
from app.agents.router.agent import SofiaRouterAgent
from app.agents.greeting.agent import GreetingAgent
from app.agents.faq_responder.agent import FAQResponderAgent
from app.agents.scheduler.agent import SchedulerAgent
from app.agents.human_escalation.agent import HumanEscalationAgent

# ============================================================================
# Singleton agents
# ============================================================================

_router_agent = SofiaRouterAgent()
_greeting_agent = GreetingAgent()
_faq_agent = FAQResponderAgent()
_scheduler_agent = SchedulerAgent()
_escalation_agent = HumanEscalationAgent()

# ============================================================================
# Intent priority — higher number = informational (runs first)
#                  lower number  = CTA (runs last)
# ============================================================================

INTENT_PRIORITY = {
    "HUMAN_ESCALATION": 1,
    "SCHEDULE":         2,
    "REENGAGE":         3,
    "FAQ":              4,
    "GREETING":         5,
    "UNCLASSIFIED":     6,
}

INTENT_REASONS = {
    "GREETING":          "greeting_detected",
    "FAQ":               "faq_intent",
    "REENGAGE":          "reengage_intent",
    "SCHEDULE":          "schedule_intent",
    "HUMAN_ESCALATION":  "human_escalation_requested",
    "UNCLASSIFIED":      "unclassified_intent",
}


# ============================================================================
# Agent dispatcher — maps intent → agent.forward() with correct args
# ============================================================================

def _extract_service_names(services_context: str) -> List[str]:
    try:
        ctx = json.loads(services_context)
        return [s.get("name", "") for s in ctx.get("services", []) if s.get("name")]
    except Exception:
        return []


def _call_agent(intent: str, state: SofiaState) -> Dict[str, Any]:
    """Dispatch to the correct agent.forward() with the right arguments."""
    patient_name = state.get("patient_name") or state.get("push_name") or "Paciente"
    clinic_name = state.get("clinic_name", "Clínica")
    assistant_name = state.get("assistant_name", "Sofia")
    history = state.get("history", [])

    if intent == "GREETING":
        return _greeting_agent.forward(
            patient_name=patient_name,
            clinic_name=clinic_name,
            assistant_name=assistant_name,
            history_length=len(history),
        )

    if intent == "SCHEDULE":
        services_ctx = load_services_context(state["clinic_id"])
        service_names = _extract_service_names(services_ctx)
        current_stage = state.get("conversation_stage", "new")
        if current_stage not in {"collecting_service", "presenting_slots", "confirming", "booked"}:
            current_stage = "collecting_service"
        return _scheduler_agent.forward(
            patient_message=state["message"],
            history=history,
            available_slots=state.get("available_slots", []),
            clinic_name=clinic_name,
            patient_name=patient_name,
            stage=current_stage,
            services_list=service_names,
        )

    if intent == "HUMAN_ESCALATION":
        return _escalation_agent.forward(
            patient_name=patient_name,
            assistant_name=assistant_name,
            clinic_name=clinic_name,
        )

    # FAQ | REENGAGE | UNCLASSIFIED → FAQResponder
    services_ctx = load_services_context(state["clinic_id"])
    business_rules = load_business_rules(state["clinic_id"])
    return _faq_agent.forward(
        patient_message=state["message"],
        history=history,
        clinic_name=clinic_name,
        patient_name=patient_name,
        services_context=services_ctx,
        business_rules=business_rules,
    )


# ============================================================================
# Graph nodes
# ============================================================================

def node_load_context(state: SofiaState) -> dict:
    """Load session + clinic context from Supabase."""
    log.info("node.load_context", remote_jid=state["remote_jid"],
             trace_id=state.get("trace_id"), clinic_id=state.get("clinic_id"))
    try:
        ctx = load_session(
            remote_jid=state["remote_jid"],
            clinic_id=state["clinic_id"],
            push_name=state.get("push_name"),
            instance_id=state.get("instance_id", ""),
            attribution_id=state.get("attribution_id"),
        )
        return ctx
    except Exception as e:
        log.error("node.load_context.error", error=str(e), trace_id=state.get("trace_id"))
        return {
            "session_id": f"{state['remote_jid']}:{state['clinic_id']}",
            "customer_id": None,
            "history": [],
            "conversation_stage": "new",
            "patient_name": state.get("push_name"),
            "clinic_name": "Clínica",
            "assistant_name": "Sofia",
        }


def node_detect_intents(state: SofiaState) -> dict:
    """Classify patient message into one or more intents and detect language."""
    log.info("node.detect_intents", message_preview=state["message"][:60],
             trace_id=state.get("trace_id"))
    try:
        result = _router_agent.forward(
            latest_message=state["message"],
            history=state.get("history", []),
            conversation_stage=state.get("conversation_stage", "new"),
        )
        detected_intents = result.get("detected_intents", ["UNCLASSIFIED"])
        language = result.get("language", "pt-BR")
        log.info("node.detect_intents.result", intents=detected_intents,
                 language=language, trace_id=state.get("trace_id"))
        return {
            "detected_intents": detected_intents,
            "language": language,
        }
    except Exception as e:
        log.error("node.detect_intents.error", error=str(e), trace_id=state.get("trace_id"))
        return {
            "detected_intents": ["UNCLASSIFIED"],
            "language": "pt-BR",
        }


def node_execute_agents(state: SofiaState) -> dict:
    """
    Run each detected agent in priority order (informational first, CTA last).
    Wraps each call with build_agent_run() for timing + token tracking.
    """
    detected_intents = list(state.get("detected_intents", ["UNCLASSIFIED"]))
    trace_id = state.get("trace_id", "")
    clinic_id = state.get("clinic_id", "")
    session_id = state.get("session_id", "")
    language = state.get("language", "pt-BR")

    # Suppress GREETING when an action intent is present — avoids double messages
    # and WhatsApp out-of-order delivery (e.g. greeting arriving after schedule prompt).
    _ACTION_INTENTS = {"SCHEDULE", "HUMAN_ESCALATION"}
    if "GREETING" in detected_intents and any(i in _ACTION_INTENTS for i in detected_intents):
        detected_intents = [i for i in detected_intents if i != "GREETING"]

    from app.core.config import get_settings
    sofia_version = get_settings().sofia_version

    # Sort: highest priority number first (informational), lowest last (CTA)
    sorted_intents = sorted(
        detected_intents,
        key=lambda x: INTENT_PRIORITY.get(x, 99),
        reverse=True,
    )

    agent_runs = []
    requires_human = False

    for intent in sorted_intents:
        log.info("node.execute_agents.run", intent=intent, trace_id=trace_id)

        agent_run = build_agent_run(
            agent_name=_agent_name_for(intent),
            reason=INTENT_REASONS.get(intent, intent.lower()),
            trace_id=trace_id,
            clinic_id=clinic_id,
            session_id=session_id,
            language=language,
            sofia_version=sofia_version,
            call=lambda i=intent: _call_agent(i, state),
        )

        agent_runs.append(agent_run)

        if agent_run.get("status") == "success" and intent == "HUMAN_ESCALATION":
            requires_human = True

    return {
        "agent_runs": agent_runs,
        "requires_human": requires_human,
    }


def _agent_name_for(intent: str) -> str:
    return {
        "GREETING":         "GreetingAgent",
        "FAQ":              "FAQResponder",
        "REENGAGE":         "FAQResponder",
        "SCHEDULE":         "Scheduler",
        "HUMAN_ESCALATION": "HumanEscalation",
        "UNCLASSIFIED":     "FAQResponder",
    }.get(intent, "FAQResponder")


def node_save_session(state: SofiaState) -> dict:
    """Persist session, agent activations, and data payloads to Supabase."""
    log.info("node.save_session", agents=[r.get("agent") for r in state.get("agent_runs", [])],
             trace_id=state.get("trace_id"))
    try:
        save_session(state)
    except Exception as e:
        log.error("node.save_session.error", error=str(e), trace_id=state.get("trace_id"))
    return {}


# ============================================================================
# Graph construction — linear, no conditional edges
# ============================================================================

workflow = StateGraph(SofiaState)

workflow.add_node("load_context",    node_load_context)
workflow.add_node("detect_intents",  node_detect_intents)
workflow.add_node("execute_agents",  node_execute_agents)
workflow.add_node("save_session",    node_save_session)

workflow.set_entry_point("load_context")
workflow.add_edge("load_context",   "detect_intents")
workflow.add_edge("detect_intents", "execute_agents")
workflow.add_edge("execute_agents", "save_session")
workflow.add_edge("save_session",   END)

sofia_graph = workflow.compile()
