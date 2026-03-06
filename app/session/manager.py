"""
Session Manager — loads and saves Sofia conversation state in Supabase.
"""

import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from app.core.supabase_client import get_supabase
from app.session.models import SofiaState


def _phone_from_jid(remote_jid: str) -> str:
    """Extract numeric phone from JID (strips @s.whatsapp.net)."""
    return remote_jid.split("@")[0]


def load_session(remote_jid: str, clinic_id: str,
                 push_name: Optional[str] = None,
                 instance_id: str = "",
                 attribution_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Load or create a Sofia session for the given patient + clinic.
    Returns session context only — services/rules loaded lazily by agents.
    """
    supabase = get_supabase()

    # Auto-resolve clinic_id via instance_clinic_map if invalid
    if (not clinic_id or clinic_id == "unknown") and instance_id:
        map_result = (
            supabase.table("sf_instance_clinic_map")
            .select("clinic_id")
            .eq("instance_name", instance_id)
            .maybe_single()
            .execute()
        )
        if map_result.data:
            clinic_id = map_result.data["clinic_id"]

    if not clinic_id or clinic_id == "unknown":
        raise ValueError(
            f"Unresolvable clinic_id for remote_jid='{remote_jid}' instance_id='{instance_id}'. "
            "Verify instance_clinic_map entry and n8n clinic_id configuration."
        )

    phone = _phone_from_jid(remote_jid)
    session_id = f"{remote_jid}:{clinic_id}"

    # 1. Upsert customer
    customer_result = (
        supabase.table("sf_customers")
        .upsert(
            {
                "phone": phone,
                "clinic_id": clinic_id,
                "full_name": push_name or "",
            },
            on_conflict="clinic_id,phone",
        )
        .execute()
    )
    customer_id = None
    if customer_result.data:
        customer_id = customer_result.data[0].get("id")

    # First-touch attribution: write only once per customer (never overwrite)
    if attribution_id and customer_id:
        try:
            supabase.table("sf_customers").update(
                {"first_attribution_id": attribution_id}
            ).eq("id", customer_id).is_("first_attribution_id", "null").execute()
        except Exception:
            pass  # best-effort, never block the conversation

    # 2. Load or create session
    # NOTE: .maybe_single() sends Accept: application/vnd.pgrst.object+json which
    # causes PostgREST to return HTTP 406 when 0 rows match, breaking the flow.
    # Use .limit(1).execute() instead — returns an empty list safely.
    session_result = (
        supabase.table("sf_sessions")
        .select("session_id, history, conversation_stage")
        .eq("session_id", session_id)
        .limit(1)
        .execute()
    )

    history: list = []
    conversation_stage: str = "new"
    patient_name: Optional[str] = push_name

    if session_result.data and len(session_result.data) > 0:
        row = session_result.data[0]
        raw_history = row.get("history") or []
        history = raw_history if isinstance(raw_history, list) else []
        conversation_stage = row.get("conversation_stage") or "new"
    else:
        supabase.table("sf_sessions").insert(
            {
                "session_id": session_id,
                "clinic_id": clinic_id,
                "remote_jid": remote_jid,
                "customer_id": customer_id,
                "history": [],
                "conversation_stage": "new",
            }
        ).execute()

    # 3. Load clinic profile
    clinic_result = (
        supabase.table("sf_clinic_profiles")
        .select("clinic_name, assistant_name, avg_ticket, address")
        .eq("clinic_id", clinic_id)
        .maybe_single()
        .execute()
    )
    clinic_name = "Clínica"
    assistant_name = "Sofia"
    if clinic_result.data:
        clinic_name = clinic_result.data.get("clinic_name") or "Clínica"
        assistant_name = clinic_result.data.get("assistant_name") or "Sofia"

    # Determine the real conversation type based on history loaded from DB.
    # This overrides whatever n8n sent (n8n defaults to "first_contact" always).
    real_conversation_type = "first_contact" if not history else "returning"

    clinic_style = load_style(clinic_id)

    return {
        "session_id": session_id,
        "customer_id": customer_id,
        "history": history,
        "conversation_stage": conversation_stage,
        "conversation_type": real_conversation_type,
        "patient_name": patient_name,
        "clinic_name": clinic_name,
        "assistant_name": assistant_name,
        "attribution_id": attribution_id,  # passed through state to _persist_appointment
        "clinic_style": clinic_style,
    }


def load_services_context(clinic_id: str) -> str:
    """Load clinic services + active offers. Returns JSON string."""
    supabase = get_supabase()
    services_result = (
        supabase.table("sf_clinic_services")
        .select("name, description, price")
        .eq("clinic_id", clinic_id)
        .execute()
    )
    offers_result = (
        supabase.table("sf_clinic_offers")
        .select("offer_name, final_price, valid_to, is_active")
        .eq("clinic_id", clinic_id)
        .eq("is_active", True)
        .execute()
    )
    return json.dumps(
        {
            "services": services_result.data or [],
            "offers": offers_result.data or [],
        },
        ensure_ascii=False,
    )


def load_business_rules(clinic_id: str) -> str:
    """Load clinic business rules. Returns JSON string."""
    supabase = get_supabase()
    result = (
        supabase.table("sf_clinic_business_rules")
        .select("rule_type, content")
        .eq("clinic_id", clinic_id)
        .execute()
    )
    return json.dumps(result.data or [], ensure_ascii=False)


_STYLE_RULE_TYPES = {"tom_voz", "personalidade", "saudacao_exemplo", "fechamento", "estilo_resposta"}

_DEFAULT_STYLE: Dict[str, Any] = {
    "tone": "Informal",
    "personality_traits": [],
    "greeting_example": "",
    "closing_example": "",
    "attendance_flow": [],
    "avg_response_tokens": 100,
    "forbidden_terms": [],
    "common_objections": [],
    "source": "default",
}


def load_style(clinic_id: str) -> Dict[str, Any]:
    """
    Load clinic style config for agents.

    Priority:
      1. la_blueprints (generated by Legacy Analyzer) — full behavioral profile
      2. sf_clinic_business_rules style keys — manual seed fallback
      3. Generic defaults
    """
    supabase = get_supabase()

    # 1. Try la_blueprints (most recent Blueprint for this clinic)
    try:
        bp_result = (
            supabase.table("la_blueprints")
            .select("blueprint_json")
            .eq("clinic_id", clinic_id)
            .order("created_at", desc=True)
            .limit(1)
            .maybe_single()
            .execute()
        )
        if bp_result.data and bp_result.data.get("blueprint_json"):
            bp = bp_result.data["blueprint_json"]
            dna = bp.get("shadow_dna_profile", {})
            identity = bp.get("agent_identity", {})
            flow = bp.get("conversational_flow", {})
            return {
                "tone": dna.get("tone_classification", "Informal"),
                "personality_traits": identity.get("personality_traits", []),
                "greeting_example": flow.get("greeting_style", {}).get("example", ""),
                "closing_example": flow.get("closing_style", {}).get("example", ""),
                "attendance_flow": flow.get("attendance_flow", []),
                "avg_response_tokens": dna.get("average_response_length_tokens", 100),
                "forbidden_terms": identity.get("forbidden_terms", []),
                "common_objections": dna.get("common_objections", []),
                "source": "blueprint",
            }
    except Exception:
        pass

    # 2. Fallback: style keys in sf_clinic_business_rules
    try:
        rules_result = (
            supabase.table("sf_clinic_business_rules")
            .select("rule_type, content")
            .eq("clinic_id", clinic_id)
            .in_("rule_type", list(_STYLE_RULE_TYPES))
            .execute()
        )
        if rules_result.data:
            rules = {r["rule_type"]: r["content"] for r in rules_result.data}
            traits_raw = rules.get("personalidade", "")
            traits = [t.strip() for t in traits_raw.split(",") if t.strip()]
            return {
                "tone": rules.get("tom_voz", "Informal"),
                "personality_traits": traits,
                "greeting_example": rules.get("saudacao_exemplo", ""),
                "closing_example": rules.get("fechamento", ""),
                "attendance_flow": [],
                "avg_response_tokens": 100,
                "forbidden_terms": [],
                "common_objections": [],
                "source": "business_rules",
            }
    except Exception:
        pass

    # 3. Generic defaults
    return dict(_DEFAULT_STYLE)


# ============================================================================
# Data persisters registry
# ============================================================================

def _persist_appointment(data: Dict[str, Any], state: SofiaState) -> None:
    """Insert appointment into sf_appointments. Guards against double booking."""
    try:
        supabase = get_supabase()
        chosen_slot = data.get("chosen_slot")
        clinic_id = state["clinic_id"]

        if not chosen_slot:
            return

        existing = (
            supabase.table("sf_appointments")
            .select("id")
            .eq("clinic_id", clinic_id)
            .eq("scheduled_at", chosen_slot)
            .neq("status", "cancelled")
            .neq("status", "no_show")
            .limit(1)
            .execute()
        )
        if existing.data:
            print(f"[persist_appointment] Slot {chosen_slot} already booked, skipping.")
            return

        supabase.table("sf_appointments").insert({
            "clinic_id": clinic_id,
            "customer_id": state.get("customer_id"),
            "session_id": state.get("session_id"),
            "remote_jid": state["remote_jid"],
            "patient_name": state.get("patient_name") or state.get("push_name"),
            "service_name": data.get("service"),
            "scheduled_at": chosen_slot,
            "status": "scheduled",
            "source": "sofia",
            "attribution_id": state.get("attribution_id"),  # null for organic contacts
        }).execute()
    except Exception as e:
        print(f"[persist_appointment] Error: {e}")


_DATA_PERSISTERS = {
    "appointment": _persist_appointment,
}


def persist_agent_data(agent_runs: List[Dict[str, Any]], state: SofiaState) -> None:
    """
    Loop through agent_runs and dispatch each run's data payload
    to the appropriate persister. Extensible via _DATA_PERSISTERS registry.
    """
    for run in agent_runs:
        data = run.get("data")
        if not data:
            continue
        persister = _DATA_PERSISTERS.get(data.get("type", ""))
        if persister:
            persister(data, state)


# ============================================================================
# Save session
# ============================================================================

def save_session(state: SofiaState) -> None:
    """
    Persist session history, agent activations audit, and agent data payloads.
    """
    supabase = get_supabase()
    agent_runs = state.get("agent_runs", [])

    # Build updated history: human turn + each agent's text messages
    new_history = list(state.get("history", []))
    new_history.append({"role": "human", "content": state["message"]})
    for run in agent_runs:
        for msg in run.get("messages", []):
            if msg.get("type") == "text":
                new_history.append({"role": run["agent"], "content": msg["content"]})

    # Truncate to last 20 entries (10 human+agent pairs)
    new_history = new_history[-20:]

    # Derive conversation_stage from last agent_run (CTA agent)
    conversation_stage = state.get("conversation_stage", "active")
    if agent_runs:
        last_stage = agent_runs[-1].get("conversation_stage")
        if last_stage:
            conversation_stage = last_stage

    # 1. Update sf_sessions
    supabase.table("sf_sessions").update(
        {
            "history": new_history,
            "conversation_stage": conversation_stage,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ).eq("session_id", state["session_id"]).execute()

    # 2. Insert one sf_agent_activations row per agent_run
    try:
        from app.core.config import get_settings
        sofia_version = get_settings().sofia_version
        for run in agent_runs:
            supabase.table("sf_agent_activations").insert({
                "session_id": state["session_id"],
                "agent_name": run.get("agent"),
                "triggered_by": ", ".join(state.get("detected_intents", [])),
                "reasoning": run.get("reasoning"),
                "processing_ms": run.get("duration_ms"),
                "sofia_version": sofia_version,
                "prompt_tokens": run.get("prompt_tokens", 0),
                "completion_tokens": run.get("completion_tokens", 0),
                "total_tokens": run.get("total_tokens", 0),
                "trace_id": run.get("trace_id"),
                "language": run.get("language", "pt-BR"),
                "clinic_id": state.get("clinic_id"),
                "messages": run.get("messages"),
                "data": run.get("data"),
                "started_at": run.get("started_at"),
                "duration_ms": run.get("duration_ms"),
            }).execute()
    except Exception as e:
        print(f"[save_session] sf_agent_activations insert failed: {e}")

    # 3. Persist agent data (appointments, escalations, etc.)
    persist_agent_data(agent_runs, state)
