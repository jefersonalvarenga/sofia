"""
Iris Evolution webhook handler.

POST /v1/iris/webhook/evolution
  - Parses Evolution payload, applies deterministic filters, resolves clinic_id,
    atomically inserts inbound row into `sf_messages` (UNIQUE clinic_id, wamid),
    and dispatches the Iris pipeline on first-seen messages.
  - Always responds 200 so Evolution does not retry on app-side errors. Internal
    errors are logged structured and returned with `{ok: false, reason}`.
  - During smoke (Iris greeting), `IRIS_ALLOWED_JIDS` (comma-separated) gates
    delivery to Jeferson's number only. Empty = allow all.
  - Auth: this route bypasses `X-API-Key` (Evolution does not send custom
    headers). Defense-in-depth comes from instance_name → clinic_id resolution
    + Supabase service_role being the only path that writes.
"""

import os
import uuid
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, BackgroundTasks, Request

from app.core.supabase_client import get_supabase
from app.core.telemetry import log
from app.iris import debounce, pipeline
from app.iris.parser import parse_evolution_payload
from app.iris.schemas import ParsedMessage

router = APIRouter(prefix="/v1/iris", tags=["iris"])


def _allowed_jids() -> Set[str]:
    raw = os.getenv("IRIS_ALLOWED_JIDS", "").strip()
    if not raw:
        return set()
    return {jid.strip() for jid in raw.split(",") if jid.strip()}


def _resolve_clinic_id(instance_name: str) -> Optional[str]:
    supabase = get_supabase()
    result = (
        supabase.table("sf_instance_clinic_map")  # tenant-lint: exempt — bootstrap; this query resolves clinic_id
        .select("clinic_id")
        .eq("instance_name", instance_name)
        .maybe_single()
        .execute()
    )
    if result and getattr(result, "data", None):
        return result.data.get("clinic_id")
    return None


def _insert_inbound_message(clinic_id: str, parsed: ParsedMessage) -> Optional[str]:
    """
    Atomic idempotent insert. Returns the new row id, or None if duplicate.

    Uses Supabase upsert with `ignoreDuplicates=True`, which compiles to
    `INSERT ... ON CONFLICT DO NOTHING RETURNING *`. supabase-py returns the
    inserted row(s) on success, empty list on conflict.
    """
    supabase = get_supabase()
    result = (
        supabase.table("sf_messages")
        .upsert(
            {
                "clinic_id": clinic_id,
                "wamid": parsed.wamid,
                "direction": "inbound",
                "content": parsed.message_content,
                "message_type": parsed.message_type,
            },
            on_conflict="clinic_id,wamid",
            ignore_duplicates=True,
        )
        .execute()
    )
    rows: List[Dict[str, Any]] = getattr(result, "data", None) or []
    if not rows:
        return None
    return rows[0].get("id")


@router.post("/webhook/evolution")
async def evolution_webhook(request: Request, background_tasks: BackgroundTasks):
    trace_id = str(uuid.uuid4())
    try:
        raw = await request.json()
    except Exception as exc:
        log.warning("iris.webhook.invalid_json", trace_id=trace_id, error=str(exc))
        return {"ok": False, "reason": "invalid_json", "trace_id": trace_id}

    log.info(
        "iris.webhook.received",
        trace_id=trace_id,
        payload_event=raw.get("event") if isinstance(raw, dict) else None,
        instance=raw.get("instance") if isinstance(raw, dict) else None,
    )

    parsed, skip_reason = parse_evolution_payload(raw if isinstance(raw, dict) else {})
    if skip_reason is not None:
        log.info("iris.webhook.skipped", trace_id=trace_id, reason=skip_reason)
        return {"ok": True, "skipped": skip_reason, "trace_id": trace_id}

    assert parsed is not None  # narrow type for readers

    allowed = _allowed_jids()
    if allowed and parsed.remote_jid not in allowed:
        log.info(
            "iris.webhook.skipped",
            trace_id=trace_id,
            reason="jid_not_allowed",
            remote_jid=parsed.remote_jid,
        )
        return {"ok": True, "skipped": "jid_not_allowed", "trace_id": trace_id}

    try:
        clinic_id = _resolve_clinic_id(parsed.instance_name)
    except Exception as exc:
        log.error(
            "iris.webhook.clinic_resolve_error",
            trace_id=trace_id,
            instance=parsed.instance_name,
            error=str(exc),
        )
        return {"ok": False, "reason": "clinic_resolve_error", "trace_id": trace_id}

    if not clinic_id:
        log.warning(
            "iris.webhook.unknown_instance",
            trace_id=trace_id,
            instance=parsed.instance_name,
        )
        return {"ok": True, "skipped": "unknown_instance", "trace_id": trace_id}

    try:
        message_id = _insert_inbound_message(clinic_id, parsed)
    except Exception as exc:
        log.error(
            "iris.webhook.insert_error",
            trace_id=trace_id,
            clinic_id=clinic_id,
            wamid=parsed.wamid,
            error=str(exc),
        )
        return {"ok": False, "reason": "insert_error", "trace_id": trace_id}

    if message_id is None:
        log.info(
            "iris.webhook.duplicate",
            trace_id=trace_id,
            clinic_id=clinic_id,
            wamid=parsed.wamid,
        )
        return {"ok": True, "duplicate": True, "trace_id": trace_id}

    log.info(
        "iris.webhook.accepted",
        trace_id=trace_id,
        clinic_id=clinic_id,
        message_id=message_id,
        wamid=parsed.wamid,
        remote_jid=parsed.remote_jid,
    )

    background_tasks.add_task(
        _safe_debounce_receive,
        clinic_id=clinic_id,
        conversation_id=parsed.remote_jid,
        message_id=message_id,
        parsed=parsed,
        trace_id=trace_id,
    )

    return {
        "ok": True,
        "trace_id": trace_id,
        "message_id": message_id,
    }


async def _safe_debounce_receive(
    *,
    clinic_id: str,
    conversation_id: str,
    message_id: str,
    parsed: ParsedMessage,
    trace_id: str,
) -> None:
    try:
        await debounce.receive(
            clinic_id=clinic_id,
            conversation_id=conversation_id,
            message_id=message_id,
            parsed=parsed,
            trace_id=trace_id,
            pipeline_invoke=pipeline.invoke,
        )
    except Exception as exc:
        log.error(
            "iris.debounce.error",
            trace_id=trace_id,
            clinic_id=clinic_id,
            message_id=message_id,
            error=str(exc),
        )
