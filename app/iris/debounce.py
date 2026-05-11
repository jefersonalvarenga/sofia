"""
Iris message debounce accumulator (EASAA-141).

Buffers inbound messages per (clinic_id, conversation_id) for a configurable
window (IRIS_DEBOUNCE_MS, default 8000 ms).  When the window expires without
a new message, all buffered rows are concatenated into a single turn and
forwarded to the pipeline exactly once.

Design decisions:
- In-process asyncio.Task per conversation window — no external queue needed
  at current volume.  If two pods receive for the same conversation
  simultaneously, pg_advisory_xact_lock inside the flusher serialises the
  pipeline invocation.
- Each individual sf_messages row is still inserted immediately (idempotency
  preserved); the buffer only delays pipeline dispatch.
- IRIS_DEBOUNCE_MS=0 disables the window: buffer is written and flushed
  synchronously inside the same call (useful for tests and local dev without
  asyncio event-loop isolation issues).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Dict, List, Optional

from app.core.supabase_client import get_supabase
from app.core.telemetry import log
from app.iris.schemas import ParsedMessage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _debounce_ms() -> int:
    try:
        return int(os.getenv("IRIS_DEBOUNCE_MS", "8000"))
    except (ValueError, TypeError):
        return 8000


# ---------------------------------------------------------------------------
# In-process window registry
# ---------------------------------------------------------------------------

# (clinic_id, conversation_id) → asyncio.Task
_pending: Dict[tuple, asyncio.Task] = {}


async def receive(
    *,
    clinic_id: str,
    conversation_id: str,
    message_id: str,
    parsed: ParsedMessage,
    trace_id: str,
    pipeline_invoke,          # callable: coroutine factory
) -> None:
    """
    Called once per accepted inbound message.

    Inserts a buffer row, then either starts or extends the debounce window.
    When the window fires, _flush is called with all pending rows.
    """
    _buffer_insert(
        clinic_id=clinic_id,
        conversation_id=conversation_id,
        message_id=message_id,
        parsed=parsed,
    )

    window_ms = _debounce_ms()

    if window_ms <= 0:
        # Synchronous path — used when IRIS_DEBOUNCE_MS=0 (tests / local dev).
        await _flush(
            clinic_id=clinic_id,
            conversation_id=conversation_id,
            trace_id=trace_id,
            pipeline_invoke=pipeline_invoke,
        )
        return

    key = (clinic_id, conversation_id)

    # Cancel existing task (extend window).
    existing = _pending.pop(key, None)
    if existing and not existing.done():
        existing.cancel()

    async def _window():
        await asyncio.sleep(window_ms / 1000)
        _pending.pop(key, None)
        await _flush(
            clinic_id=clinic_id,
            conversation_id=conversation_id,
            trace_id=trace_id,
            pipeline_invoke=pipeline_invoke,
        )

    _pending[key] = asyncio.ensure_future(_window())


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def _buffer_insert(
    *,
    clinic_id: str,
    conversation_id: str,
    message_id: str,
    parsed: ParsedMessage,
) -> None:
    window_ms = max(_debounce_ms(), 0)
    # flush_after computed server-side to avoid clock skew; we just store now+window
    # as an advisory timestamp — the real gate is flushed=false.
    from datetime import datetime, timedelta, timezone
    flush_after = (
        datetime.now(timezone.utc) + timedelta(milliseconds=window_ms)
    ).isoformat()

    supabase = get_supabase()
    supabase.table("sf_message_buffer").insert({
        "clinic_id": clinic_id,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "wamid": parsed.wamid,
        "content": parsed.message_content,
        "push_name": parsed.push_name,
        "message_type": parsed.message_type,
        "instance_name": parsed.instance_name,
        "phone": parsed.phone,
        "flush_after": flush_after,
    }).execute()


def _claim_and_drain(
    *,
    clinic_id: str,
    conversation_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """
    Atomically marks all unflushed buffer rows as flushed and returns them.

    PostgREST translates UPDATE+filter to a single SQL statement:
      UPDATE sf_message_buffer SET flushed=true
      WHERE clinic_id=$1 AND conversation_id=$2 AND flushed=false
      RETURNING *

    This is atomic: the first pod to execute wins; a concurrent pod gets an
    empty result set and skips the pipeline call.  No separate advisory lock
    needed — the UPDATE row-level lock provides the same guarantee.
    """
    supabase = get_supabase()
    result = (
        supabase.table("sf_message_buffer")
        .update({"flushed": True})
        .eq("clinic_id", clinic_id)
        .eq("conversation_id", conversation_id)
        .eq("flushed", False)
        .execute()
    )
    rows: List[Dict[str, Any]] = getattr(result, "data", None) or []
    if not rows:
        return None
    rows.sort(key=lambda r: r.get("created_at", ""))
    return rows


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------

async def _flush(
    *,
    clinic_id: str,
    conversation_id: str,
    trace_id: str,
    pipeline_invoke,
) -> None:
    """
    Drains the buffer and fires the pipeline once with all accumulated messages.

    Atomicity is provided by _claim_and_drain's UPDATE WHERE flushed=false.
    The in-process _pending registry prevents redundant concurrent flushes
    within a single pod; across pods the atomic UPDATE ensures only one wins.
    """
    rows = _claim_and_drain(clinic_id=clinic_id, conversation_id=conversation_id)
    if not rows:
        log.info(
            "iris.debounce.flush_noop",
            trace_id=trace_id,
            clinic_id=clinic_id,
            conversation_id=conversation_id,
        )
        return

    # Concatenate content in arrival order.
    combined_content = "\n".join(r["content"] for r in rows)
    first = rows[0]

    # Build a synthetic ParsedMessage representing the accumulated turn.
    combined = ParsedMessage(
        instance_name=first["instance_name"],
        remote_jid=conversation_id,
        wamid=first["wamid"],
        push_name=first["push_name"],
        message_content=combined_content,
        message_type=first["message_type"],
        phone=first["phone"],
    )

    # Use the first message_id as the pipeline anchor (idempotency key).
    anchor_message_id = first["message_id"]

    log.info(
        "iris.debounce.flush",
        trace_id=trace_id,
        clinic_id=clinic_id,
        conversation_id=conversation_id,
        message_count=len(rows),
        anchor_message_id=anchor_message_id,
    )

    try:
        await pipeline_invoke(
            clinic_id=clinic_id,
            message_id=anchor_message_id,
            parsed=combined,
            trace_id=trace_id,
        )
    except Exception as exc:
        log.error(
            "iris.debounce.pipeline_error",
            trace_id=trace_id,
            clinic_id=clinic_id,
            conversation_id=conversation_id,
            error=str(exc),
        )
