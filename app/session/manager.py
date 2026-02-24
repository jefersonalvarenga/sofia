"""
Session Manager — loads and saves Sofia conversation state in Supabase.
"""

import json
from typing import Optional, Dict, Any
from app.core.supabase_client import get_supabase
from app.session.models import SofiaState


def _phone_from_jid(remote_jid: str) -> str:
    """Extract numeric phone from JID (strips @s.whatsapp.net)."""
    return remote_jid.split("@")[0]


def load_session(remote_jid: str, clinic_id: str, push_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load or create a Sofia session for the given patient + clinic.

    Returns a partial SofiaState dict with:
      session_id, customer_id, history, conversation_stage, patient_name,
      clinic_name, assistant_name, services_context, business_rules
    """
    supabase = get_supabase()
    phone = _phone_from_jid(remote_jid)
    session_id = f"{remote_jid}:{clinic_id}"

    # 1. Upsert customer
    customer_result = (
        supabase.table("customers")
        .upsert(
            {
                "phone": phone,
                "clinic_id": clinic_id,
                "push_name": push_name or "",
            },
            on_conflict="phone,clinic_id",
        )
        .execute()
    )
    customer_id = None
    if customer_result.data:
        customer_id = customer_result.data[0].get("id")

    # 2. Load or create session
    session_result = (
        supabase.table("sessions")
        .select("id, session_id, history, conversation_stage, intentions, intake, appointment")
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )

    history: list = []
    conversation_stage: str = "new"
    patient_name: Optional[str] = push_name

    if session_result.data:
        raw_history = session_result.data.get("history") or []
        history = raw_history if isinstance(raw_history, list) else []
        conversation_stage = session_result.data.get("conversation_stage") or "new"
    else:
        # Create new session
        supabase.table("sessions").insert(
            {
                "session_id": session_id,
                "lead_id": customer_id,
                "history": [],
                "conversation_stage": "new",
            }
        ).execute()

    # 3. Load clinic profile
    clinic_result = (
        supabase.table("clinic_profiles")
        .select("clinic_name, assistant_name, avg_ticket, address")  # clinic_name added via migration 004
        .eq("clinic_id", clinic_id)
        .maybe_single()
        .execute()
    )
    clinic_name = "Clínica"
    assistant_name = "Sofia"
    if clinic_result.data:
        clinic_name = clinic_result.data.get("clinic_name") or "Clínica"
        assistant_name = clinic_result.data.get("assistant_name") or "Sofia"

    # 4. Load services + offers
    services_result = (
        supabase.table("clinic_services")
        .select("name, description, price")
        .eq("clinic_id", clinic_id)
        .execute()
    )
    offers_result = (
        supabase.table("clinic_offers")
        .select("offer_name, final_price, valid_to, is_active")
        .eq("clinic_id", clinic_id)
        .eq("is_active", True)
        .execute()
    )
    services_context = json.dumps(
        {
            "services": services_result.data or [],
            "offers": offers_result.data or [],
        },
        ensure_ascii=False,
    )

    # 5. Load business rules
    rules_result = (
        supabase.table("clinic_business_rules")
        .select("rule_type, content")
        .eq("clinic_id", clinic_id)
        .execute()
    )
    business_rules = json.dumps(rules_result.data or [], ensure_ascii=False)

    return {
        "session_id": session_id,
        "customer_id": customer_id,
        "history": history,
        "conversation_stage": conversation_stage,
        "patient_name": patient_name,
        "clinic_name": clinic_name,
        "assistant_name": assistant_name,
        "services_context": services_context,
        "business_rules": business_rules,
    }


def save_session(state: SofiaState) -> None:
    """
    Persist session history, conversation rows, and agent activation audit.
    """
    supabase = get_supabase()

    # Build updated history with new human turn + agent response
    new_history = list(state.get("history", []))
    new_history.append({"role": "human", "content": state["message"]})
    if state.get("response_message") and state.get("agent_name"):
        new_history.append({"role": state["agent_name"], "content": state["response_message"]})

    # 1. Update sessions table
    supabase.table("sessions").update(
        {
            "history": new_history,
            "conversation_stage": state.get("conversation_stage", "active"),
        }
    ).eq("session_id", state["session_id"]).execute()

    # 2. Insert conversation rows
    customer_id = state.get("customer_id")
    if customer_id:
        # Human message row
        supabase.table("conversations").insert(
            {
                "lead_id": customer_id,
                "flow": "sofia",
                "role": "human",
                "content": state["message"],
                "stage": state.get("conversation_stage", "active"),
            }
        ).execute()

        # Agent response row
        if state.get("response_message") and state.get("agent_name"):
            supabase.table("conversations").insert(
                {
                    "lead_id": customer_id,
                    "flow": "sofia",
                    "role": state["agent_name"],
                    "content": state["response_message"],
                    "stage": state.get("conversation_stage", "active"),
                }
            ).execute()

    # 3. Insert agent activation audit
    if state.get("agent_name"):
        from app.core.config import get_settings
        sofia_version = get_settings().sofia_version
        supabase.table("sf_agent_activations").insert(
            {
                "session_id": state["session_id"],
                "agent_name": state["agent_name"],
                "triggered_by": state.get("intent"),
                "reasoning": state.get("reasoning"),
                "processing_ms": state.get("processing_time_ms"),
                "sofia_version": sofia_version,
            }
        ).execute()
