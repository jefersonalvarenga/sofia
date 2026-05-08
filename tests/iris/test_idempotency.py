"""
Iris C10 — webhook idempotency end-to-end.

POSTs the same Evolution payload twice and asserts:
  - first call: 200, ``ok: true``, pipeline invoked, sf_messages row inserted.
  - second call: 200, ``duplicate: true``, pipeline NOT invoked,
    sf_agent_activations count unchanged.

The webhook's idempotency boundary is the ``(clinic_id, wamid)`` UNIQUE
constraint on ``sf_messages`` (UPSERT with ignore_duplicates). On a duplicate
the inbound row is skipped *before* the pipeline runs — so no LLM calls,
no token spend, no extra audit rows.

[EASAA-31](../../EASAA/issues/EASAA-31).
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


CLINIC_ID = "57952a29-e228-4cac-b5fa-3d20ba478f5d"
PATIENT_JID = "5511999990000@s.whatsapp.net"


def _payload(*, wamid: str, text: str = "oi") -> Dict[str, Any]:
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


# ----------------------------------------------------------------------------
# Tiny supabase fake: enforces (clinic_id, wamid) uniqueness on sf_messages
# and counts inserts on sf_agent_activations.
# ----------------------------------------------------------------------------


class _SupabaseFake:
    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []
        self.activations: List[Dict[str, Any]] = []
        self._next_id = 1

    def table(self, name: str) -> "_Table":
        return _Table(self, name)


class _Table:
    def __init__(self, parent: _SupabaseFake, name: str) -> None:
        self._parent = parent
        self._name = name
        self._chain: Dict[str, Any] = {}

    def select(self, *_a: Any, **_kw: Any) -> "_Table":
        self._chain["op"] = "select"
        return self

    def insert(self, row: Dict[str, Any]) -> "_Table":
        self._chain["op"] = "insert"
        self._chain["row"] = row
        return self

    def upsert(
        self,
        row: Dict[str, Any],
        *,
        on_conflict: str,
        ignore_duplicates: bool = False,
    ) -> "_Table":
        self._chain["op"] = "upsert"
        self._chain["row"] = row
        self._chain["ignore_duplicates"] = ignore_duplicates
        return self

    def eq(self, key: str, value: Any) -> "_Table":
        self._chain.setdefault("filters", {})[key] = value
        return self

    def maybe_single(self) -> "_Table":
        self._chain["single"] = True
        return self

    def execute(self) -> Any:
        if self._name == "sf_instance_clinic_map" and self._chain.get("op") == "select":
            instance = (self._chain.get("filters") or {}).get("instance_name")
            data = {"clinic_id": CLINIC_ID} if instance == "Sofia-EasyScale" else None
            return MagicMock(data=data)

        if self._name == "sf_messages" and self._chain.get("op") == "upsert":
            row = self._chain["row"]
            existing = next(
                (
                    m for m in self._parent.messages
                    if m["clinic_id"] == row["clinic_id"]
                    and m["wamid"] == row["wamid"]
                ),
                None,
            )
            if existing is not None and self._chain.get("ignore_duplicates"):
                return MagicMock(data=[])
            stored = dict(row, id=f"msg-{self._parent._next_id:04d}")
            self._parent._next_id += 1
            self._parent.messages.append(stored)
            return MagicMock(data=[stored])

        if self._name == "sf_agent_activations" and self._chain.get("op") == "insert":
            self._parent.activations.append(self._chain["row"])
            return MagicMock(data=[self._chain["row"]])

        return MagicMock(data=None)


@pytest.fixture
def supabase() -> _SupabaseFake:
    return _SupabaseFake()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, supabase: _SupabaseFake):
    monkeypatch.delenv("IRIS_ALLOWED_JIDS", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    with patch("app.iris.webhook.get_supabase", return_value=supabase):
        from main import app

        yield TestClient(app)


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


class TestWebhookIdempotency:
    """The webhook MUST NOT invoke the pipeline on a duplicate wamid."""

    def test_duplicate_wamid_does_not_invoke_pipeline_or_grow_activations(
        self,
        client: TestClient,
        supabase: _SupabaseFake,
    ) -> None:
        wamid = "wamid-idempotency-001"
        payload = _payload(wamid=wamid)

        with patch(
            "app.iris.webhook.pipeline.invoke", new_callable=AsyncMock
        ) as pipeline_spy:
            # First POST — accept, dispatch pipeline.
            first = client.post("/v1/iris/webhook/evolution", json=payload)
            assert first.status_code == 200
            body = first.json()
            assert body["ok"] is True
            assert body.get("duplicate") is not True
            assert body.get("message_id", "").startswith("msg-")
            assert pipeline_spy.await_count == 1

            activations_before = len(supabase.activations)
            messages_before = len(supabase.messages)

            # Second POST — duplicate, must short-circuit before pipeline.
            second = client.post("/v1/iris/webhook/evolution", json=payload)
            assert second.status_code == 200
            second_body = second.json()
            assert second_body["ok"] is True
            assert second_body.get("duplicate") is True

            # No second LLM-bearing pipeline run.
            assert pipeline_spy.await_count == 1, (
                "pipeline.invoke must not be called on a duplicate wamid; "
                "got await_count="
                f"{pipeline_spy.await_count}"
            )
            # No additional sf_messages row.
            assert len(supabase.messages) == messages_before
            # No additional sf_agent_activations row.
            assert len(supabase.activations) == activations_before

    def test_distinct_wamids_each_invoke_pipeline(
        self,
        client: TestClient,
        supabase: _SupabaseFake,
    ) -> None:
        with patch(
            "app.iris.webhook.pipeline.invoke", new_callable=AsyncMock
        ) as pipeline_spy:
            for wamid in ("wamid-A", "wamid-B", "wamid-C"):
                resp = client.post(
                    "/v1/iris/webhook/evolution", json=_payload(wamid=wamid)
                )
                assert resp.status_code == 200
                assert resp.json()["ok"] is True

            assert pipeline_spy.await_count == 3
            assert len(supabase.messages) == 3
