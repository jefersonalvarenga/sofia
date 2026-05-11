"""
Iris debounce accumulator — unit tests ([EASAA-141](../../../EASAA/issues/EASAA-141)).

These tests exercise the buffer + RPC flush dance against an in-memory
fake of supabase-py. They cover the acceptance smoke directly:
  - 3 messages in 5s → 1 pipeline call with concatenated content
  - each message is persisted in sf_messages individually (3 rows)
  - earlier flushes defer to the latest (no fragmented responses)

The Postgres `pg_advisory_xact_lock` is exercised in the real RPC; here
the fake serializes via Python ordering, which is enough to validate the
caller-side logic (watermark, defer, concatenate).
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


CLINIC_ID = "57952a29-e228-4cac-b5fa-3d20ba478f5d"
PATIENT_JID = "5511999990000@s.whatsapp.net"


def _payload(*, wamid: str, text: str) -> Dict[str, Any]:
    return {
        "event": "messages.upsert",
        "instance": "Sofia-EasyScale",
        "data": {
            "key": {"remoteJid": PATIENT_JID, "fromMe": False, "id": wamid},
            "pushName": "Maria",
            "message": {"conversation": text},
            "messageType": "conversation",
        },
    }


# ---------------------------------------------------------------------------
# In-memory fakes (same shape used by tests/test_iris_webhook.py — duplicated
# here so this file is self-contained for future refactors)
# ---------------------------------------------------------------------------


class _SupabaseFake:
    def __init__(self):
        self._inserted: List[Dict[str, Any]] = []
        self._buffer: List[Dict[str, Any]] = []
        self._next_message_id = 1
        self._next_buffer_id = 1

    def table(self, name: str):
        return _SupabaseTable(self, name)

    def rpc(self, name: str, params: Dict[str, Any]):
        return _SupabaseRpc(self, name, params)


class _SupabaseTable:
    def __init__(self, parent: _SupabaseFake, name: str):
        self._parent = parent
        self._name = name
        self._chain: Dict[str, Any] = {}

    def select(self, *_a, **_kw):
        self._chain["op"] = "select"
        return self

    def eq(self, key: str, value: Any):
        self._chain.setdefault("filters", {})[key] = value
        return self

    def maybe_single(self):
        return self

    def upsert(self, row: Dict[str, Any], *, on_conflict: str, ignore_duplicates: bool = False):
        self._chain.update(op="upsert", row=row, on_conflict=on_conflict, ignore_duplicates=ignore_duplicates)
        return self

    def insert(self, row: Dict[str, Any]):
        self._chain.update(op="insert", row=row)
        return self

    def execute(self):
        if self._name == "sf_instance_clinic_map":
            return MagicMock(data={"clinic_id": CLINIC_ID})

        if self._name == "sf_messages" and self._chain.get("op") == "upsert":
            row = self._chain["row"]
            if any(
                r["clinic_id"] == row["clinic_id"] and r["wamid"] == row["wamid"]
                for r in self._parent._inserted
            ):
                return MagicMock(data=[])
            new_id = f"msg-{self._parent._next_message_id:04d}"
            self._parent._next_message_id += 1
            stored = dict(row, id=new_id)
            self._parent._inserted.append(stored)
            return MagicMock(data=[stored])

        if self._name == "sf_message_buffer" and self._chain.get("op") == "insert":
            row = self._chain["row"]
            buf_id = f"buf-{self._parent._next_buffer_id:04d}"
            stored = dict(row, id=buf_id, flushed_at=None, seq=self._parent._next_buffer_id)
            self._parent._next_buffer_id += 1
            self._parent._buffer.append(stored)
            return MagicMock(data=[stored])

        return MagicMock(data=None)


class _SupabaseRpc:
    def __init__(self, parent: _SupabaseFake, name: str, params: Dict[str, Any]):
        self._parent = parent
        self._name = name
        self._params = params

    def execute(self):
        if self._name != "iris_try_flush_conversation":
            return MagicMock(data=None)
        clinic_id = self._params["p_clinic_id"]
        remote_jid = self._params["p_remote_jid"]
        watermark_buffer_id = self._params["p_watermark_buffer_id"]

        caller = next((r for r in self._parent._buffer if r["id"] == watermark_buffer_id), None)
        if caller is None:
            return MagicMock(data=[{"flushed": False, "message_ids": [], "concatenated_content": "", "buffer_count": 0, "latest_buffer_id": None}])

        pending = [
            r for r in self._parent._buffer
            if r["clinic_id"] == clinic_id and r["remote_jid"] == remote_jid and r["flushed_at"] is None
        ]
        if any(r["seq"] > caller["seq"] for r in pending):
            return MagicMock(data=[{"flushed": False, "message_ids": [], "concatenated_content": "", "buffer_count": 0, "latest_buffer_id": None}])

        pending.sort(key=lambda r: r["seq"])
        for r in pending:
            r["flushed_at"] = "now"
        return MagicMock(data=[{
            "flushed": True,
            "message_ids": [r["message_id"] for r in pending],
            "concatenated_content": "\n".join(r["content"] for r in pending),
            "buffer_count": len(pending),
            "latest_buffer_id": pending[-1]["id"],
        }])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def supabase_fake():
    fake = _SupabaseFake()
    with patch("app.iris.webhook.get_supabase", return_value=fake), \
         patch("app.iris.accumulator.get_supabase", return_value=fake):
        yield fake


@pytest.fixture
def pipeline_spy():
    with patch("app.iris.webhook.pipeline.invoke", new_callable=AsyncMock) as spy:
        yield spy


@pytest.fixture(autouse=True)
def _zero_debounce(monkeypatch):
    monkeypatch.setenv("IRIS_DEBOUNCE_MS", "0")
    monkeypatch.delenv("IRIS_ALLOWED_JIDS", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)


@pytest.fixture
def client():
    from main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Acceptance smoke ([EASAA-141](../../../EASAA/issues/EASAA-141))
# ---------------------------------------------------------------------------


class TestDebounceAccumulator:
    def test_three_messages_in_window_yield_single_pipeline_call(
        self, client, supabase_fake, pipeline_spy
    ):
        """AC: 3 messages in 5s → 1 pipeline call, content concatenated with \\n,
        3 rows in sf_messages."""
        for i, text in enumerate(("oi", "quero agendar", "amanhã de tarde"), start=1):
            resp = client.post(
                "/v1/iris/webhook/evolution",
                json=_payload(wamid=f"wamid-{i:03d}", text=text),
            )
            assert resp.status_code == 200
            assert resp.json()["ok"] is True

        # Each inbound is its own sf_messages row (idempotência preservada).
        assert len(supabase_fake._inserted) == 3

        # Only one pipeline invocation, with concatenated content.
        assert pipeline_spy.await_count == 1
        kwargs = pipeline_spy.await_args.kwargs
        assert kwargs["clinic_id"] == CLINIC_ID
        assert kwargs["parsed"].message_content == "oi\nquero agendar\namanhã de tarde"
        # Anchor (message_id) is the latest message of the window.
        latest_id = supabase_fake._inserted[-1]["id"]
        assert kwargs["message_id"] == latest_id

    def test_single_message_still_flushes(
        self, client, supabase_fake, pipeline_spy
    ):
        resp = client.post(
            "/v1/iris/webhook/evolution",
            json=_payload(wamid="wamid-solo", text="olá Iris"),
        )
        assert resp.status_code == 200
        assert pipeline_spy.await_count == 1
        assert pipeline_spy.await_args.kwargs["parsed"].message_content == "olá Iris"

    def test_duplicate_inbound_does_not_reach_buffer(
        self, client, supabase_fake, pipeline_spy
    ):
        """sf_messages dedupe (UNIQUE clinic_id, wamid) means duplicates never
        enqueue a buffer row → no duplicate pipeline trigger."""
        payload = _payload(wamid="wamid-dup", text="oi")
        client.post("/v1/iris/webhook/evolution", json=payload)
        second = client.post("/v1/iris/webhook/evolution", json=payload)
        assert second.json().get("duplicate") is True

        # First call flushed → 1 pipeline; duplicate short-circuits.
        assert pipeline_spy.await_count == 1
        assert len(supabase_fake._buffer) == 1


class TestDebounceEnvVar:
    def test_default_when_unset(self, monkeypatch):
        from app.iris import accumulator
        monkeypatch.delenv("IRIS_DEBOUNCE_MS", raising=False)
        assert accumulator.debounce_ms() == accumulator.DEFAULT_DEBOUNCE_MS

    def test_override(self, monkeypatch):
        from app.iris import accumulator
        monkeypatch.setenv("IRIS_DEBOUNCE_MS", "2500")
        assert accumulator.debounce_ms() == 2500

    def test_invalid_falls_back_to_default(self, monkeypatch):
        from app.iris import accumulator
        monkeypatch.setenv("IRIS_DEBOUNCE_MS", "not-a-number")
        assert accumulator.debounce_ms() == accumulator.DEFAULT_DEBOUNCE_MS

    def test_zero_allowed(self, monkeypatch):
        from app.iris import accumulator
        monkeypatch.setenv("IRIS_DEBOUNCE_MS", "0")
        assert accumulator.debounce_ms() == 0

    def test_negative_clamped_to_zero(self, monkeypatch):
        from app.iris import accumulator
        monkeypatch.setenv("IRIS_DEBOUNCE_MS", "-100")
        assert accumulator.debounce_ms() == 0
