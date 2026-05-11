"""
Iris debounce accumulator ([EASAA-141](../../../EASAA/issues/EASAA-141)).

Pacientes que digitam mensagens em rajada ("oi" + "quero agendar" +
"amanhã de tarde") devem receber **uma** resposta consolidada, não três
respostas fora de ordem. Este módulo é o motor:

1. `enqueue_inbound` — chamado pelo webhook (C7) logo após inserir a
   linha em `sf_messages`. Persiste em `sf_message_buffer` e devolve o
   id do buffer (o "watermark" do caller).
2. `schedule_flush` — agenda uma background task que dorme
   `IRIS_DEBOUNCE_MS` (default 8000ms) e depois chama
   `iris_try_flush_conversation` via Supabase RPC. A RPC serializa via
   `pg_advisory_xact_lock` por `(clinic_id, remote_jid)` e devolve
   `flushed=TRUE` só pro task cuja mensagem é a mais recente. Os outros
   tasks veem que existe mensagem unflushed mais nova e desistem.
3. Se o flush vier `flushed=TRUE`, montamos uma `ParsedMessage`
   sintética com o conteúdo concatenado (separador `\n`) e o anchor
   (push_name / wamid / instance_name) da mensagem mais recente da
   janela, e invocamos `pipeline.invoke` uma única vez.

Idempotência preservada: cada mensagem continua sendo persistida
individualmente em `sf_messages` (UNIQUE message_id em
sf_message_buffer impede re-enqueue).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from app.core.supabase_client import get_supabase
from app.core.telemetry import log
from app.iris.schemas import ParsedMessage


DEFAULT_DEBOUNCE_MS = 8000


def debounce_ms() -> int:
    raw = os.getenv("IRIS_DEBOUNCE_MS", "").strip()
    if not raw:
        return DEFAULT_DEBOUNCE_MS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_DEBOUNCE_MS
    return max(value, 0)


def enqueue_inbound(
    *,
    clinic_id: str,
    message_id: str,
    parsed: ParsedMessage,
    trace_id: str,
) -> Optional[str]:
    """Insert into sf_message_buffer. Returns the buffer row id."""
    supabase = get_supabase()
    result = (
        supabase.table("sf_message_buffer")
        .insert(
            {
                "clinic_id": clinic_id,
                "remote_jid": parsed.remote_jid,
                "message_id": message_id,
                "content": parsed.message_content,
                "instance_name": parsed.instance_name,
                "push_name": parsed.push_name or None,
                "message_type": parsed.message_type,
                "wamid": parsed.wamid,
                "trace_id": trace_id,
            }
        )
        .execute()
    )
    rows: List[Dict[str, Any]] = getattr(result, "data", None) or []
    if not rows:
        return None
    return rows[0].get("id")


async def _sleep_then_flush(
    *,
    clinic_id: str,
    parsed: ParsedMessage,
    watermark_buffer_id: str,
    trace_id: str,
    pipeline_invoke,
) -> None:
    """Background task body. Sleep the debounce window, then try to flush."""
    await asyncio.sleep(debounce_ms() / 1000.0)

    try:
        rpc_result = (
            get_supabase()
            .rpc(
                "iris_try_flush_conversation",
                {
                    "p_clinic_id": clinic_id,
                    "p_remote_jid": parsed.remote_jid,
                    "p_watermark_buffer_id": watermark_buffer_id,
                },
            )
            .execute()
        )
    except Exception as exc:
        log.error(
            "iris.accumulator.rpc_error",
            trace_id=trace_id,
            clinic_id=clinic_id,
            remote_jid=parsed.remote_jid,
            error=str(exc),
        )
        return

    rows: List[Dict[str, Any]] = getattr(rpc_result, "data", None) or []
    row = rows[0] if rows else {}

    if not row.get("flushed"):
        log.info(
            "iris.accumulator.deferred",
            trace_id=trace_id,
            clinic_id=clinic_id,
            remote_jid=parsed.remote_jid,
            reason="newer_message_pending",
        )
        return

    message_ids: List[str] = row.get("message_ids") or []
    concatenated: str = row.get("concatenated_content") or ""
    buffer_count: int = int(row.get("buffer_count") or 0)

    log.info(
        "iris.accumulator.flushed",
        trace_id=trace_id,
        clinic_id=clinic_id,
        remote_jid=parsed.remote_jid,
        buffer_count=buffer_count,
        message_ids=message_ids,
    )

    if not message_ids or not concatenated:
        return

    # Build the synthetic ParsedMessage that drives the pipeline. We keep
    # the caller's anchor (push_name / wamid / instance_name) — by design,
    # the canonical flusher is the task scheduled by the newest message
    # in the window, so its anchor is the right one to use.
    consolidated = ParsedMessage(
        instance_name=parsed.instance_name,
        remote_jid=parsed.remote_jid,
        wamid=parsed.wamid,
        push_name=parsed.push_name,
        message_content=concatenated,
        message_type=parsed.message_type,
        phone=getattr(parsed, "phone", "") or "",
    )

    try:
        await pipeline_invoke(
            clinic_id=clinic_id,
            message_id=message_ids[-1],
            parsed=consolidated,
            trace_id=trace_id,
        )
    except Exception as exc:
        log.error(
            "iris.accumulator.pipeline_error",
            trace_id=trace_id,
            clinic_id=clinic_id,
            remote_jid=parsed.remote_jid,
            error=str(exc),
        )


def schedule_flush(
    *,
    background_tasks,
    clinic_id: str,
    parsed: ParsedMessage,
    watermark_buffer_id: str,
    trace_id: str,
    pipeline_invoke,
) -> None:
    """Register the debounced flush as a FastAPI BackgroundTask."""
    background_tasks.add_task(
        _sleep_then_flush,
        clinic_id=clinic_id,
        parsed=parsed,
        watermark_buffer_id=watermark_buffer_id,
        trace_id=trace_id,
        pipeline_invoke=pipeline_invoke,
    )
