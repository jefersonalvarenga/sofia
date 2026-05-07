"""Unit tests for ``app.iris.evolution_client``.

Iris C9 ([EASAA-30](../../../EASAA/issues/EASAA-30)).

Uses ``httpx.MockTransport`` for HTTP mocking — no extra dependency on
respx/pytest-asyncio. Async tests are wrapped in ``asyncio.run`` because
the project does not configure pytest-asyncio.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from app.iris.evolution_client import (
    EvolutionAPIError,
    persist_outbound_message,
    send_text_message,
)


BASE_URL = "https://evolution.test"
INSTANCE = "iris-prod"
REMOTE_JID = "5511999998888@s.whatsapp.net"
PHONE = "5511999998888"
CONTENT = "Olá! Sou a Iris, da Clínica Vitória."
API_KEY = "test-evo-key"

NO_BACKOFF: tuple[float, ...] = (0.0, 0.0, 0.0)


def _ok_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "key": {"id": "BAE5F6A1B2C3D4", "remoteJid": REMOTE_JID, "fromMe": True},
            "status": "PENDING",
            "message": {"conversation": CONTENT},
        },
    )


async def _run_with_transport(handler, *, backoffs=NO_BACKOFF):
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        return await send_text_message(
            INSTANCE,
            REMOTE_JID,
            CONTENT,
            API_KEY,
            base_url=BASE_URL,
            client=client,
            backoffs_seconds=backoffs,
        )


def test_send_text_payload_and_headers():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode())
        return _ok_response(request)

    result = asyncio.run(_run_with_transport(handler))

    assert captured["method"] == "POST"
    assert captured["url"] == f"{BASE_URL}/message/sendText/{INSTANCE}"
    assert captured["headers"]["apikey"] == API_KEY
    assert captured["headers"]["content-type"].startswith("application/json")
    assert captured["body"] == {"number": PHONE, "text": CONTENT}
    assert result["key"]["id"] == "BAE5F6A1B2C3D4"


def test_retry_then_success_on_5xx():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503, text="upstream temporarily unavailable")
        return _ok_response(request)

    result = asyncio.run(_run_with_transport(handler))
    assert len(attempts) == 3
    assert result["status"] == "PENDING"


def test_retry_then_success_on_network_error():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 2:
            raise httpx.ConnectError("connection reset", request=request)
        return _ok_response(request)

    result = asyncio.run(_run_with_transport(handler))
    assert len(attempts) == 2
    assert result["key"]["id"] == "BAE5F6A1B2C3D4"


def test_retry_exhausted_raises_evolution_error():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(502, text="bad gateway")

    with pytest.raises(EvolutionAPIError) as exc_info:
        asyncio.run(_run_with_transport(handler))

    err = exc_info.value
    assert err.status_code == 502
    assert err.attempts == 4
    assert len(attempts) == 4


def test_4xx_short_circuits_retries():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(401, text="invalid apikey")

    with pytest.raises(EvolutionAPIError) as exc_info:
        asyncio.run(_run_with_transport(handler))

    err = exc_info.value
    assert err.status_code == 401
    assert err.attempts == 1
    assert len(attempts) == 1, "4xx must NOT trigger retry"


def test_default_backoffs_are_exponential():
    from app.iris.evolution_client import RETRY_BACKOFFS_SECONDS

    assert RETRY_BACKOFFS_SECONDS == (0.5, 1.0, 2.0)


def test_resolve_base_url_missing_raises(monkeypatch):
    """No env var set, no base_url override — should refuse to send."""
    from app.core import config as cfg

    settings = cfg.get_settings()
    monkeypatch.setattr(settings, "evolution_api_url", "", raising=False)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(_ok_response)) as c:
            await send_text_message(
                INSTANCE,
                REMOTE_JID,
                CONTENT,
                API_KEY,
                client=c,
                backoffs_seconds=NO_BACKOFF,
            )

    with pytest.raises(EvolutionAPIError):
        asyncio.run(go())


# ---------------------------------------------------------------------------
# persist_outbound_message
# ---------------------------------------------------------------------------


class _FakeTable:
    def __init__(self, captured: dict[str, Any], result_data: list[dict[str, Any]]):
        self._captured = captured
        self._result_data = result_data

    def upsert(self, row, *, on_conflict, ignore_duplicates):
        self._captured["table"] = "sf_messages"
        self._captured["row"] = row
        self._captured["on_conflict"] = on_conflict
        self._captured["ignore_duplicates"] = ignore_duplicates
        return self

    def execute(self):
        class _Result:
            def __init__(self, data):
                self.data = data

        return _Result(self._result_data)


class _FakeSupabase:
    def __init__(self, captured: dict[str, Any], result_data: list[dict[str, Any]]):
        self._captured = captured
        self._result_data = result_data

    def table(self, name: str):
        self._captured["table_called"] = name
        return _FakeTable(self._captured, self._result_data)


def test_persist_outbound_writes_correct_row():
    captured: dict[str, Any] = {}
    fake_row = {
        "id": "uuid-1",
        "clinic_id": "clinic-uuid",
        "wamid": "BAE5F6A1B2C3D4",
        "direction": "outbound",
    }
    supabase = _FakeSupabase(captured, [fake_row])

    inserted = persist_outbound_message(
        clinic_id="clinic-uuid",
        session_id="5511999998888@s.whatsapp.net:clinic-uuid",
        wamid="BAE5F6A1B2C3D4",
        content=CONTENT,
        supabase=supabase,
    )

    assert inserted == fake_row
    assert captured["table_called"] == "sf_messages"
    assert captured["on_conflict"] == "clinic_id,wamid"
    assert captured["ignore_duplicates"] is True
    assert captured["row"]["clinic_id"] == "clinic-uuid"
    assert captured["row"]["wamid"] == "BAE5F6A1B2C3D4"
    assert captured["row"]["direction"] == "outbound"
    assert captured["row"]["content"] == CONTENT
    assert captured["row"]["message_type"] == "text"


def test_persist_outbound_returns_none_on_duplicate():
    captured: dict[str, Any] = {}
    supabase = _FakeSupabase(captured, [])

    inserted = persist_outbound_message(
        clinic_id="clinic-uuid",
        session_id=None,
        wamid="duplicate-wamid",
        content=CONTENT,
        supabase=supabase,
    )

    assert inserted is None
    assert captured["row"]["session_id"] is None
