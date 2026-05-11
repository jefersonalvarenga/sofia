"""
Iris C7 — webhook + parser unit tests.

Exercises the parse + filter chain and the idempotent insert path with the
Supabase client and pipeline mocked. Real e2e + idempotency-against-Postgres
smoke lives in C10 ([EASAA-31](../../../EASAA/issues/EASAA-31)).
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


CLINIC_ID = "57952a29-e228-4cac-b5fa-3d20ba478f5d"
PATIENT_JID = "5511999990000@s.whatsapp.net"
GROUP_JID = "120363012345678901@g.us"


def _make_payload(
    *,
    instance: str = "Sofia-EasyScale",
    remote_jid: str = PATIENT_JID,
    from_me: bool = False,
    wamid: str = "wamid-001",
    push_name: str = "Maria",
    text: str = "oi",
    message: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    msg = message if message is not None else {"conversation": text}
    return {
        "event": "messages.upsert",
        "instance": instance,
        "data": {
            "key": {"remoteJid": remote_jid, "fromMe": from_me, "id": wamid},
            "pushName": push_name,
            "message": msg,
            "messageType": "conversation",
        },
    }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_parses_conversation_text(self):
        from app.iris.parser import parse_evolution_payload

        parsed, skip = parse_evolution_payload(_make_payload())
        assert skip is None
        assert parsed is not None
        assert parsed.instance_name == "Sofia-EasyScale"
        assert parsed.remote_jid == PATIENT_JID
        assert parsed.wamid == "wamid-001"
        assert parsed.message_content == "oi"
        assert parsed.message_type == "text"
        assert parsed.phone == "5511999990000"
        assert parsed.push_name == "Maria"

    def test_parses_extended_text(self):
        from app.iris.parser import parse_evolution_payload

        payload = _make_payload(message={"extendedTextMessage": {"text": "olá!"}})
        parsed, skip = parse_evolution_payload(payload)
        assert skip is None
        assert parsed is not None
        assert parsed.message_content == "olá!"
        assert parsed.message_type == "text"

    def test_parses_messages_array_shape(self):
        from app.iris.parser import parse_evolution_payload

        payload = {
            "instance": "Sofia-EasyScale",
            "data": {
                "messages": [
                    {
                        "key": {"remoteJid": PATIENT_JID, "fromMe": False, "id": "w1"},
                        "pushName": "Ana",
                        "message": {"conversation": "oi"},
                    }
                ]
            },
        }
        parsed, skip = parse_evolution_payload(payload)
        assert skip is None
        assert parsed is not None and parsed.wamid == "w1"

    def test_filters_from_me(self):
        from app.iris.parser import parse_evolution_payload

        parsed, skip = parse_evolution_payload(_make_payload(from_me=True))
        assert parsed is None and skip == "from_me"

    def test_filters_group(self):
        from app.iris.parser import parse_evolution_payload

        parsed, skip = parse_evolution_payload(_make_payload(remote_jid=GROUP_JID))
        assert parsed is None and skip == "group_message"

    def test_filters_status_broadcast(self):
        from app.iris.parser import parse_evolution_payload

        parsed, skip = parse_evolution_payload(_make_payload(remote_jid="status@broadcast"))
        assert parsed is None and skip == "status_broadcast"

    def test_filters_no_text_content(self):
        from app.iris.parser import parse_evolution_payload

        parsed, skip = parse_evolution_payload(_make_payload(message={}))
        assert parsed is None and skip == "no_text_content"

    def test_audio_message_gets_audio_type(self):
        from app.iris.parser import parse_evolution_payload

        parsed, skip = parse_evolution_payload(
            _make_payload(message={"audioMessage": {"url": "x"}})
        )
        assert skip is None
        assert parsed is not None and parsed.message_type == "audio"

    def test_missing_instance(self):
        from app.iris.parser import parse_evolution_payload

        payload = _make_payload()
        del payload["instance"]
        parsed, skip = parse_evolution_payload(payload)
        assert parsed is None and skip == "missing_instance"


# ---------------------------------------------------------------------------
# Webhook handler
# ---------------------------------------------------------------------------


class _SupabaseFake:
    """In-memory fake of the supabase-py client surface the webhook touches."""

    def __init__(self, *, instance_to_clinic: Dict[str, str]):
        self._instance_to_clinic = instance_to_clinic
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

    def select(self, *_args, **_kwargs):
        self._chain["op"] = "select"
        return self

    def eq(self, key: str, value: Any):
        self._chain.setdefault("filters", {})[key] = value
        return self

    def maybe_single(self):
        self._chain["single"] = True
        return self

    def upsert(self, row: Dict[str, Any], *, on_conflict: str, ignore_duplicates: bool = False):
        self._chain["op"] = "upsert"
        self._chain["row"] = row
        self._chain["on_conflict"] = on_conflict
        self._chain["ignore_duplicates"] = ignore_duplicates
        return self

    def insert(self, row: Dict[str, Any]):
        self._chain["op"] = "insert"
        self._chain["row"] = row
        return self

    def execute(self):
        if self._name == "sf_instance_clinic_map" and self._chain.get("op") == "select":
            instance = self._chain.get("filters", {}).get("instance_name")
            clinic_id = self._parent._instance_to_clinic.get(instance)
            data = {"clinic_id": clinic_id} if clinic_id else None
            return MagicMock(data=data)

        if self._name == "sf_messages" and self._chain.get("op") == "upsert":
            row = self._chain["row"]
            existing = next(
                (
                    r for r in self._parent._inserted
                    if r["clinic_id"] == row["clinic_id"] and r["wamid"] == row["wamid"]
                ),
                None,
            )
            if existing is not None and self._chain.get("ignore_duplicates"):
                return MagicMock(data=[])
            new_id = f"msg-{self._parent._next_message_id:04d}"
            self._parent._next_message_id += 1
            stored = dict(row, id=new_id)
            self._parent._inserted.append(stored)
            return MagicMock(data=[stored])

        if self._name == "sf_message_buffer" and self._chain.get("op") == "insert":
            row = self._chain["row"]
            buf_id = f"buf-{self._parent._next_buffer_id:04d}"
            self._parent._next_buffer_id += 1
            stored = dict(row, id=buf_id, flushed_at=None, seq=self._parent._next_buffer_id)
            self._parent._buffer.append(stored)
            return MagicMock(data=[stored])

        return MagicMock(data=None)


class _SupabaseRpc:
    """Fakes the iris_try_flush_conversation RPC.

    Semantics: returns flushed=True only when no later unflushed buffer row
    exists for the same (clinic_id, remote_jid). When it flushes, marks the
    rows in the parent's buffer as flushed_at=<sentinel>.
    """

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

        caller = next(
            (r for r in self._parent._buffer if r["id"] == watermark_buffer_id),
            None,
        )
        if caller is None:
            return MagicMock(data=[{
                "flushed": False, "message_ids": [], "concatenated_content": "",
                "buffer_count": 0, "latest_buffer_id": None,
            }])

        pending = [
            r for r in self._parent._buffer
            if r["clinic_id"] == clinic_id
            and r["remote_jid"] == remote_jid
            and r["flushed_at"] is None
        ]
        has_newer = any(r["seq"] > caller["seq"] for r in pending)
        if has_newer:
            return MagicMock(data=[{
                "flushed": False, "message_ids": [], "concatenated_content": "",
                "buffer_count": 0, "latest_buffer_id": None,
            }])

        pending.sort(key=lambda r: r["seq"])
        for r in pending:
            r["flushed_at"] = "now"

        return MagicMock(data=[{
            "flushed": True,
            "message_ids": [r["message_id"] for r in pending],
            "concatenated_content": "\n".join(r["content"] for r in pending),
            "buffer_count": len(pending),
            "latest_buffer_id": pending[-1]["id"] if pending else None,
        }])


@pytest.fixture
def supabase_fake():
    fake = _SupabaseFake(instance_to_clinic={"Sofia-EasyScale": CLINIC_ID})
    with patch("app.iris.webhook.get_supabase", return_value=fake), \
         patch("app.iris.accumulator.get_supabase", return_value=fake):
        yield fake


@pytest.fixture
def pipeline_spy():
    with patch("app.iris.webhook.pipeline.invoke", new_callable=AsyncMock) as spy:
        yield spy


@pytest.fixture(autouse=True)
def _zero_debounce(monkeypatch):
    """Run the debounce timer at 0ms so tests don't sleep 8s."""
    monkeypatch.setenv("IRIS_DEBOUNCE_MS", "0")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("IRIS_ALLOWED_JIDS", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    from main import app

    return TestClient(app)


class TestEvolutionWebhook:
    def test_happy_path_inserts_and_dispatches(self, client, supabase_fake, pipeline_spy):
        resp = client.post("/v1/iris/webhook/evolution", json=_make_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body.get("message_id", "").startswith("msg-")
        pipeline_spy.assert_awaited_once()

    def test_idempotent_duplicate_skips_pipeline(
        self, client, supabase_fake, pipeline_spy
    ):
        payload = _make_payload(wamid="wamid-dup")
        first = client.post("/v1/iris/webhook/evolution", json=payload)
        assert first.status_code == 200
        assert first.json()["ok"] is True

        second = client.post("/v1/iris/webhook/evolution", json=payload)
        assert second.status_code == 200
        assert second.json().get("duplicate") is True

        assert pipeline_spy.await_count == 1

    def test_filters_skip_without_dispatch(self, client, supabase_fake, pipeline_spy):
        for kwargs in (
            {"from_me": True},
            {"remote_jid": GROUP_JID},
            {"remote_jid": "status@broadcast"},
            {"message": {}},
        ):
            resp = client.post("/v1/iris/webhook/evolution", json=_make_payload(**kwargs))
            assert resp.status_code == 200
            body = resp.json()
            assert body["ok"] is True and "skipped" in body

        pipeline_spy.assert_not_called()

    def test_unknown_instance_returns_200_and_skips(
        self, client, supabase_fake, pipeline_spy
    ):
        resp = client.post(
            "/v1/iris/webhook/evolution",
            json=_make_payload(instance="not-mapped"),
        )
        assert resp.status_code == 200
        assert resp.json()["skipped"] == "unknown_instance"
        pipeline_spy.assert_not_called()

    def test_jid_allowlist_filters_other_numbers(
        self, client, supabase_fake, pipeline_spy, monkeypatch
    ):
        monkeypatch.setenv(
            "IRIS_ALLOWED_JIDS", "5511555550000@s.whatsapp.net"
        )
        resp = client.post("/v1/iris/webhook/evolution", json=_make_payload())
        assert resp.status_code == 200
        assert resp.json()["skipped"] == "jid_not_allowed"
        pipeline_spy.assert_not_called()

    def test_invalid_json_returns_200_with_reason(
        self, client, supabase_fake, pipeline_spy
    ):
        resp = client.post(
            "/v1/iris/webhook/evolution",
            content="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is False
        pipeline_spy.assert_not_called()
