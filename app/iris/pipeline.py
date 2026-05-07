"""
Iris pipeline dispatcher.

C7 ([EASAA-28](../../../EASAA/issues/EASAA-28)) lands a no-op stub. C8
([EASAA-29](../../../EASAA/issues/EASAA-29)) replaces this with the real
LangGraph subgraph (load_context → router → greeting/unknown_fallback →
save_session → send).

Keeping the public surface (`invoke`) stable from day one so the webhook
handler doesn't need to change when C8 ships.
"""

from typing import Any, Dict

from app.core.telemetry import log
from app.iris.schemas import ParsedMessage


async def invoke(
    *,
    clinic_id: str,
    message_id: str,
    parsed: ParsedMessage,
    trace_id: str,
) -> Dict[str, Any]:
    """
    Run the Iris greeting pipeline against an inserted inbound message.

    Stub for C7 — logs and returns. Real subgraph wired in C8.
    """
    log.info(
        "iris.pipeline.dispatched",
        trace_id=trace_id,
        clinic_id=clinic_id,
        message_id=message_id,
        wamid=parsed.wamid,
        remote_jid=parsed.remote_jid,
        message_type=parsed.message_type,
        stub=True,
    )
    return {"status": "stub", "message_id": message_id}
