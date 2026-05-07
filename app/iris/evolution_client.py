"""Evolution API client — async send + outbound persistence.

Iris C9 ([EASAA-30](../../../EASAA/issues/EASAA-30)).

Sends outbound text messages to Evolution (WhatsApp Business API gateway)
and persists them to ``sf_messages`` for audit/observability. Designed to
be invoked from the LangGraph pipeline (C8) once a response has been
generated, so it stays free of Iris-pipeline state — it only needs the
ids it will write.

Retry policy: 3 attempts with exponential backoff (0.5s, 1s, 2s) on
network errors and 5xx responses. 4xx responses raise immediately —
those are auth/payload bugs, retrying is a waste.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import httpx
import structlog

log = structlog.get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10.0
RETRY_BACKOFFS_SECONDS: tuple[float, ...] = (0.5, 1.0, 2.0)


class EvolutionAPIError(RuntimeError):
    """Raised when Evolution send fails after exhausting retries."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text
        self.attempts = attempts


def _phone_from_jid(remote_jid: str) -> str:
    """Strip ``@s.whatsapp.net`` suffix; mirrors ``app.session.manager``."""
    return remote_jid.split("@", 1)[0]


def _resolve_base_url(base_url: Optional[str]) -> str:
    if base_url:
        return base_url.rstrip("/")
    from app.core.config import get_settings

    settings = get_settings()
    url = getattr(settings, "evolution_api_url", "") or ""
    if not url:
        raise EvolutionAPIError(
            "EVOLUTION_API_URL is not configured. Set it in .env or pass base_url=."
        )
    return url.rstrip("/")


async def send_text_message(
    instance: str,
    remote_jid: str,
    content: str,
    api_key: str,
    *,
    base_url: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    backoffs_seconds: tuple[float, ...] = RETRY_BACKOFFS_SECONDS,
) -> dict[str, Any]:
    """POST ``/message/sendText/{instance}`` with retries.

    Body: ``{"number": "<phone>", "text": "<content>"}``. Header: ``apikey``.

    Returns the parsed JSON response from Evolution. Typical shape::

        {"key": {"id": "<wamid>", "remoteJid": "..."}, "status": "PENDING", ...}

    Raises ``EvolutionAPIError`` after the last retry fails.
    """
    url = f"{_resolve_base_url(base_url)}/message/sendText/{instance}"
    payload = {"number": _phone_from_jid(remote_jid), "text": content}
    headers = {"apikey": api_key, "Content-Type": "application/json"}

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout_seconds)

    last_error_text: Optional[str] = None
    last_status: Optional[int] = None

    try:
        for attempt_idx in range(len(backoffs_seconds) + 1):
            attempt = attempt_idx + 1
            started_at = time.perf_counter()
            try:
                response = await http.post(url, json=payload, headers=headers)
                latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
                last_status = response.status_code

                if response.status_code < 400:
                    log.info(
                        "evolution.send.ok",
                        instance=instance,
                        remote_jid=remote_jid,
                        attempt=attempt,
                        status=response.status_code,
                        latency_ms=latency_ms,
                    )
                    return response.json()

                if 400 <= response.status_code < 500:
                    last_error_text = response.text
                    log.error(
                        "evolution.send.client_error",
                        instance=instance,
                        remote_jid=remote_jid,
                        attempt=attempt,
                        status=response.status_code,
                        latency_ms=latency_ms,
                        body=last_error_text[:500] if last_error_text else None,
                    )
                    raise EvolutionAPIError(
                        f"Evolution returned {response.status_code} (no retry on 4xx)",
                        status_code=response.status_code,
                        response_text=last_error_text,
                        attempts=attempt,
                    )

                last_error_text = response.text
                log.warning(
                    "evolution.send.server_error",
                    instance=instance,
                    remote_jid=remote_jid,
                    attempt=attempt,
                    status=response.status_code,
                    latency_ms=latency_ms,
                )
            except httpx.HTTPError as exc:
                last_error_text = str(exc)
                log.warning(
                    "evolution.send.network_error",
                    instance=instance,
                    remote_jid=remote_jid,
                    attempt=attempt,
                    error=last_error_text,
                )

            if attempt_idx < len(backoffs_seconds):
                await asyncio.sleep(backoffs_seconds[attempt_idx])

        raise EvolutionAPIError(
            "Evolution send failed after retries",
            status_code=last_status,
            response_text=last_error_text,
            attempts=len(backoffs_seconds) + 1,
        )
    finally:
        if owns_client:
            await http.aclose()


def persist_outbound_message(
    *,
    clinic_id: str,
    session_id: Optional[str],
    wamid: str,
    content: str,
    supabase: Optional[Any] = None,
    message_type: str = "text",
) -> dict[str, Any] | None:
    """Insert outbound message into ``sf_messages`` (idempotent via UNIQUE).

    ``ON CONFLICT DO NOTHING`` is enforced at the DB level by
    ``sf_messages_clinic_wamid_unique``; supabase-py's ``insert`` with
    ``upsert=False`` (default) will raise on conflict, so we use
    ``.upsert(..., on_conflict='clinic_id,wamid', ignore_duplicates=True)``
    to keep a duplicate write a no-op rather than an exception.

    Returns the inserted row or ``None`` when the row already existed.
    """
    if supabase is None:
        from app.core.supabase_client import get_supabase

        supabase = get_supabase()

    row = {
        "clinic_id": clinic_id,
        "session_id": session_id,
        "wamid": wamid,
        "direction": "outbound",
        "content": content,
        "message_type": message_type,
    }

    result = (
        supabase.table("sf_messages")
        .upsert(row, on_conflict="clinic_id,wamid", ignore_duplicates=True)
        .execute()
    )

    inserted = result.data[0] if getattr(result, "data", None) else None
    log.info(
        "evolution.outbound.persisted",
        clinic_id=clinic_id,
        session_id=session_id,
        wamid=wamid,
        inserted=bool(inserted),
    )
    return inserted


__all__ = [
    "EvolutionAPIError",
    "send_text_message",
    "persist_outbound_message",
    "DEFAULT_TIMEOUT_SECONDS",
    "RETRY_BACKOFFS_SECONDS",
]
