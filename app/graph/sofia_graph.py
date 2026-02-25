"""
Sofia LangGraph — main conversational flow.

START → load_context → route_intent → [faq_responder | scheduler | human_escalation] → save_session → END
"""

from langgraph.graph import StateGraph, END

from app.session.models import SofiaState
from app.session.manager import load_session, save_session
from app.agents.router.agent import SofiaRouterAgent
from app.agents.faq_responder.agent import FAQResponderAgent
from app.agents.scheduler.agent import SchedulerAgent
from app.agents.human_escalation.agent import HumanEscalationAgent

# ============================================================================
# Singleton agents
# ============================================================================

_router_agent = SofiaRouterAgent()
_faq_agent = FAQResponderAgent()
_scheduler_agent = SchedulerAgent()
_escalation_agent = HumanEscalationAgent()

# ============================================================================
# Node functions
# ============================================================================

def node_load_context(state: SofiaState) -> dict:
    """Load session + clinic context from Supabase."""
    print(f"[load_context] Loading session for {state['remote_jid']}")
    try:
        ctx = load_session(
            remote_jid=state["remote_jid"],
            clinic_id=state["clinic_id"],
            push_name=state.get("push_name"),
            instance_id=state.get("instance_id", ""),
        )
        return ctx
    except Exception as e:
        print(f"[load_context] Error: {e}")
        return {
            "session_id": f"{state['remote_jid']}:{state['clinic_id']}",
            "customer_id": None,
            "history": [],
            "conversation_stage": "new",
            "patient_name": state.get("push_name"),
            "clinic_name": "Clínica",
            "assistant_name": "Sofia",
            "services_context": "{}",
            "business_rules": "[]",
        }


def node_route_intent(state: SofiaState) -> dict:
    """Classify patient message into a single intent."""
    print(f"[route_intent] Classifying: {state['message'][:60]}")
    result = _router_agent.forward(
        latest_message=state["message"],
        history=state.get("history", []),
        conversation_stage=state.get("conversation_stage", "new"),
    )
    print(f"[route_intent] Intent: {result['intent']} ({result['confidence']:.2f})")
    return result


def node_faq_responder(state: SofiaState) -> dict:
    """Handle FAQ, greetings, and general inquiries."""
    print(f"[faq_responder] Handling intent: {state.get('intent')}")
    result = _faq_agent.forward(
        patient_message=state["message"],
        history=state.get("history", []),
        clinic_name=state.get("clinic_name", "Clínica"),
        patient_name=state.get("patient_name") or state.get("push_name") or "Paciente",
        services_context=state.get("services_context", "{}"),
        business_rules=state.get("business_rules", "[]"),
    )
    return result


def node_scheduler(state: SofiaState) -> dict:
    """Handle appointment scheduling (multi-turn)."""
    print(f"[scheduler] Stage: {state.get('conversation_stage')}")
    current_stage = state.get("conversation_stage", "new")
    # Map non-scheduling stages to collecting_service as starting point
    if current_stage not in {"collecting_service", "presenting_slots", "confirming", "booked"}:
        current_stage = "collecting_service"

    result = _scheduler_agent.forward(
        patient_message=state["message"],
        history=state.get("history", []),
        available_slots=state.get("available_slots", []),
        clinic_name=state.get("clinic_name", "Clínica"),
        patient_name=state.get("patient_name") or state.get("push_name") or "Paciente",
        stage=current_stage,
    )

    # If booked, create appointment record
    appointment_created = None
    if result.get("conversation_stage") == "booked" and result.get("chosen_slot"):
        appointment_created = _create_appointment(state, result)

    return {**result, "appointment_created": appointment_created}


def _create_appointment(state: SofiaState, scheduler_result: dict):
    """Insert appointment record into Supabase."""
    try:
        from app.core.supabase_client import get_supabase
        supabase = get_supabase()
        record = {
            "clinic_id": state["clinic_id"],
            "customer_id": state.get("customer_id"),
            "session_id": state.get("session_id"),
            "remote_jid": state["remote_jid"],
            "patient_name": state.get("patient_name") or state.get("push_name"),
            "service_name": scheduler_result.get("service_requested"),
            "scheduled_at": scheduler_result.get("chosen_slot"),
            "status": "scheduled",
            "source": "sofia",
        }
        result = supabase.table("appointments").insert(record).execute()
        return result.data[0] if result.data else record
    except Exception as e:
        print(f"[_create_appointment] Error: {e}")
        return None


def node_human_escalation(state: SofiaState) -> dict:
    """Deterministic human escalation — no LLM."""
    print("[human_escalation] Escalating to human")
    return _escalation_agent.forward(
        patient_name=state.get("patient_name") or state.get("push_name") or "Paciente",
        assistant_name=state.get("assistant_name", "Sofia"),
        clinic_name=state.get("clinic_name", "Clínica"),
    )


def node_save_session(state: SofiaState) -> dict:
    """Persist session, conversations, and audit trail to Supabase."""
    print(f"[save_session] Agent: {state.get('agent_name')}")
    try:
        save_session(state)
    except Exception as e:
        print(f"[save_session] Error: {e}")
    return {}


# ============================================================================
# Routing logic
# ============================================================================

def _route_after_intent(state: SofiaState) -> str:
    intent = state.get("intent", "UNCLASSIFIED")
    if intent == "SCHEDULE":
        return "scheduler"
    if intent == "HUMAN_ESCALATION":
        return "human_escalation"
    # FAQ, GREETING, REENGAGE, UNCLASSIFIED → faq_responder
    return "faq_responder"


# ============================================================================
# Graph construction
# ============================================================================

workflow = StateGraph(SofiaState)

workflow.add_node("load_context", node_load_context)
workflow.add_node("route_intent", node_route_intent)
workflow.add_node("faq_responder", node_faq_responder)
workflow.add_node("scheduler", node_scheduler)
workflow.add_node("human_escalation", node_human_escalation)
workflow.add_node("save_session", node_save_session)

workflow.set_entry_point("load_context")
workflow.add_edge("load_context", "route_intent")

workflow.add_conditional_edges(
    "route_intent",
    _route_after_intent,
    {
        "faq_responder": "faq_responder",
        "scheduler": "scheduler",
        "human_escalation": "human_escalation",
    }
)

workflow.add_edge("faq_responder", "save_session")
workflow.add_edge("scheduler", "save_session")
workflow.add_edge("human_escalation", "save_session")
workflow.add_edge("save_session", END)

sofia_graph = workflow.compile()
