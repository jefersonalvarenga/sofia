"""
Iris C10 — pipeline end-to-end smoke (mocked Anthropic + mocked Evolution).

Drives the C8 LangGraph pipeline ([EASAA-29](../../EASAA/issues/EASAA-29))
with a synthetic ``"oi"`` payload and asserts:

  - Webhook returns ``200 ok``.
  - Pipeline result lists ``GreetingAgent`` in ``agent_runs``.
  - Final ``conversation_stage`` is ``"greeting"``.
  - Outbound message is delivered through Evolution (HTTP mocked).

External services are stubbed end-to-end:
  - Anthropic SDK → fake tool_use Message returning ``GREETING``.
  - Supabase → in-memory fake table store.
  - Evolution HTTP → ``httpx.MockTransport`` returning a synthetic ``key.id``.

[EASAA-31](../../EASAA/issues/EASAA-31).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.iris import pipeline as iris_pipeline
from app.iris.schemas import ParsedMessage


CLINIC_ID = "57952a29-e228-4cac-b5fa-3d20ba478f5d"
PATIENT_JID = "5511999990000@s.whatsapp.net"


# ---------------------------------------------------------------------------
# Anthropic stub — emits the same shape IrisRouterAgent expects.
# ---------------------------------------------------------------------------


def _fake_anthropic_response(
    *,
    intents: List[Any],
    reasoning: str = "fake",
    scope: str = "oi",
) -> Any:
    """Build a fake router response.

    `intents` may be a list of macro_state strings (legacy convenience —
    `scope` is used as scope_text for each) or a list of ``(macro_state,
    scope_text)`` tuples for multi-intent fixtures.
    """
    parsed_intents: List[Dict[str, str]] = []
    for item in intents:
        if isinstance(item, tuple):
            macro, scope_text = item
        else:
            macro, scope_text = item, scope
        parsed_intents.append({"macro_state": macro, "scope_text": scope_text})

    block = MagicMock()
    block.type = "tool_use"
    block.name = "classify_intent"
    block.input = {
        "intents": parsed_intents,
        "language": "pt-BR",
        "reasoning": reasoning,
        "confidence": 0.95,
    }
    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock(input_tokens=42, output_tokens=11)
    return response


# ---------------------------------------------------------------------------
# Supabase fake — captures session/customer/profile reads + activation writes.
# ---------------------------------------------------------------------------


class _SupabaseFake:
    def __init__(self) -> None:
        self.activations: List[Dict[str, Any]] = []
        self.sessions_updated: List[Dict[str, Any]] = []
        self.outbound_messages: List[Dict[str, Any]] = []

    def table(self, name: str) -> "_Table":
        return _Table(self, name)


class _Table:
    def __init__(self, parent: _SupabaseFake, name: str) -> None:
        self._parent = parent
        self._name = name
        self._chain: Dict[str, Any] = {"filters": {}}

    # ------------- write paths -------------

    def insert(self, row: Dict[str, Any]) -> "_Table":
        self._chain["op"] = "insert"
        self._chain["row"] = row
        return self

    def upsert(self, row: Dict[str, Any], **_kw: Any) -> "_Table":
        self._chain["op"] = "upsert"
        self._chain["row"] = row
        return self

    def update(self, row: Dict[str, Any]) -> "_Table":
        self._chain["op"] = "update"
        self._chain["row"] = row
        return self

    # ------------- read paths -------------

    def select(self, *_a: Any, **_kw: Any) -> "_Table":
        self._chain["op"] = self._chain.get("op", "select")
        return self

    def eq(self, key: str, value: Any) -> "_Table":
        self._chain["filters"][key] = value
        return self

    def neq(self, *_a: Any, **_kw: Any) -> "_Table":
        return self

    def is_(self, *_a: Any, **_kw: Any) -> "_Table":
        return self

    def in_(self, *_a: Any, **_kw: Any) -> "_Table":
        return self

    def order(self, *_a: Any, **_kw: Any) -> "_Table":
        return self

    def limit(self, *_a: Any, **_kw: Any) -> "_Table":
        return self

    def maybe_single(self) -> "_Table":
        self._chain["single"] = True
        return self

    # ------------- terminator -------------

    def execute(self) -> Any:
        op = self._chain.get("op")

        if op == "insert" and self._name == "sf_agent_activations":
            self._parent.activations.append(self._chain["row"])
            return MagicMock(data=[self._chain["row"]])

        if op in ("insert", "upsert") and self._name == "sf_messages":
            self._parent.outbound_messages.append(self._chain["row"])
            return MagicMock(data=[self._chain["row"]])

        if op == "update" and self._name == "sf_sessions":
            self._parent.sessions_updated.append(self._chain["row"])
            return MagicMock(data=[self._chain["row"]])

        # Reads — return shapes load_session() expects.
        if self._name == "sf_customers":
            return MagicMock(data=[{"id": "customer-uuid-001"}])
        if self._name == "sf_clinic_profiles":
            return MagicMock(
                data={"clinic_name": "Clínica Vitória", "assistant_name": "Iris"}
            )
        if self._name == "sf_clinic_services":
            return MagicMock(data=[])
        if self._name == "sf_clinic_offers":
            return MagicMock(data=[])
        if self._name == "sf_clinic_business_rules":
            return MagicMock(data=[])
        if self._name == "la_blueprints":
            return MagicMock(data=None)
        if self._name == "sf_sessions":
            # Empty sessions row → load_session() inserts a new one.
            return MagicMock(data=[])
        if self._name == "sf_instance_clinic_map":
            return MagicMock(data={"clinic_id": CLINIC_ID})

        return MagicMock(data=None)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def _ok_evolution_response(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "key": {
                "id": "BAE5OUTBOUND001",
                "remoteJid": PATIENT_JID,
                "fromMe": True,
            },
            "status": "PENDING",
        },
    )


@pytest.fixture
def supabase_fake() -> _SupabaseFake:
    return _SupabaseFake()


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOLUTION_API_KEY", "test-key")
    monkeypatch.setenv("EVOLUTION_API_URL", "https://evo.test")
    monkeypatch.setenv("SOFIA_VERSION", "iris-test")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    # Ensure get_settings re-reads env (it caches in a module global).
    from app.core import config as core_config

    monkeypatch.setattr(core_config, "_settings", None, raising=False)


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


class TestPipelineE2E:
    def test_greeting_pipeline_end_to_end(
        self,
        supabase_fake: _SupabaseFake,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 1) Patch Anthropic client used inside IrisRouterAgent.
        fake_client = MagicMock()
        fake_client.messages.create = MagicMock(
            return_value=_fake_anthropic_response(intents=["GREETING"])
        )
        monkeypatch.setattr(
            iris_pipeline._router_agent, "client", fake_client, raising=True
        )

        # 2) Patch supabase used by session manager + evolution_client.
        # session.manager imports get_supabase at module level; evolution_client
        # imports it lazily inside persist_outbound_message — patch the source.
        monkeypatch.setattr(
            "app.core.supabase_client.get_supabase",
            lambda: supabase_fake,
        )
        monkeypatch.setattr(
            "app.session.manager.get_supabase",
            lambda: supabase_fake,
        )

        # 3) Patch Evolution HTTP transport.
        transport = httpx.MockTransport(_ok_evolution_response)
        original_send_text = iris_pipeline.send_text_message

        async def _patched_send(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            kwargs.setdefault("base_url", "https://evo.test")
            async with httpx.AsyncClient(transport=transport) as http:
                return await original_send_text(*args, client=http, **kwargs)

        monkeypatch.setattr(iris_pipeline, "send_text_message", _patched_send)

        # 4) Drive the pipeline directly with a synthetic ParsedMessage.
        parsed = ParsedMessage(
            instance_name="iris-prod",
            remote_jid=PATIENT_JID,
            phone="5511999990000",
            wamid="wamid-e2e-001",
            push_name="Maria",
            message_content="oi",
            message_type="text",
        )

        result = asyncio.run(
            iris_pipeline.invoke(
                clinic_id=CLINIC_ID,
                message_id="msg-e2e-001",
                parsed=parsed,
                trace_id="trace-e2e-001",
            )
        )

        # ---- assertions on pipeline result ----
        assert result["status"] == "ok"
        assert result["primary_intent"] == "GREETING"
        assert "GREETING" in result["detected_intents"]

        agent_names = [run.get("agent") for run in result["agent_runs"]]
        assert "GreetingAgent" in agent_names, (
            f"GreetingAgent missing from agent_runs={agent_names}"
        )

        # ---- assertions on persisted state ----
        # Conversation stage written to sf_sessions must be 'greeting'.
        stages_written = [
            row.get("conversation_stage") for row in supabase_fake.sessions_updated
        ]
        assert "greeting" in stages_written, (
            f"sf_sessions.update never wrote conversation_stage='greeting' "
            f"(got {stages_written!r})"
        )

        # ---- assertions on Evolution send ----
        assert result["outbound_wamid"] == "BAE5OUTBOUND001"
        assert any(
            m.get("wamid") == "BAE5OUTBOUND001" and m.get("direction") == "outbound"
            for m in supabase_fake.outbound_messages
        ), "outbound sf_messages row was not persisted"

        # ---- assertions on audit trail ----
        activated_agents = [
            row.get("agent_name") for row in supabase_fake.activations
        ]
        assert "GreetingAgent" in activated_agents
        assert "IrisRouterAgent" in activated_agents

    def test_multi_intent_fans_out_and_aggregates(
        self,
        supabase_fake: _SupabaseFake,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Smoke from [EASAA-140](../../EASAA/issues/EASAA-140):

        Input "Quanto custa o botox? Posso fazer estando grávida?" produces
        two intents (FAQ + HUMAN_ESCALATION). The pipeline fans out into one
        specialist run per intent (both fall back deterministically here —
        real FAQ/Escalation specialists ship in follow-ups) and the
        aggregator consolidates the two replies into a single outbound
        message.

        To prove aggregation is real and not a no-op, we register a
        temporary FAQ specialist that emits a price reply, so the two
        responses are distinct and we can assert both are present in the
        aggregated outbound message.
        """
        # 1) Router emits two intents with distinct scope_texts
        fake_client = MagicMock()
        fake_client.messages.create = MagicMock(
            return_value=_fake_anthropic_response(
                intents=[
                    ("FAQ", "Quanto custa o botox?"),
                    ("HUMAN_ESCALATION", "Posso fazer estando grávida?"),
                ],
                reasoning="preço + escalação médica",
            )
        )
        monkeypatch.setattr(
            iris_pipeline._router_agent, "client", fake_client, raising=True
        )

        # 2) Register a temporary FAQ specialist so the aggregator has two
        #    distinct, non-empty responses to consolidate.
        def _fake_faq(state: Any, scope_text: str) -> Dict[str, Any]:
            return {
                "messages": [
                    {
                        "type": "text",
                        "content": (
                            f"Botox custa entre R$800 e R$1.500 por região "
                            f"(scope: {scope_text})."
                        ),
                    }
                ],
                "conversation_stage": state.get("conversation_stage", "new"),
                "reasoning": "fake faq",
                "data": None,
            }

        original_registry = dict(iris_pipeline.SPECIALIST_REGISTRY)
        iris_pipeline.SPECIALIST_REGISTRY["FAQ"] = ("FAQResponder", _fake_faq)
        monkeypatch.setattr(
            iris_pipeline, "SPECIALIST_REGISTRY", iris_pipeline.SPECIALIST_REGISTRY
        )

        # 3) Patch supabase + Evolution like the single-intent smoke
        monkeypatch.setattr(
            "app.core.supabase_client.get_supabase",
            lambda: supabase_fake,
        )
        monkeypatch.setattr(
            "app.session.manager.get_supabase",
            lambda: supabase_fake,
        )

        outbound_payloads: List[str] = []

        def _capture_evolution(request: httpx.Request) -> httpx.Response:
            try:
                body = json.loads(request.content.decode("utf-8"))
            except Exception:  # pragma: no cover - defensive
                body = {}
            outbound_payloads.append(body.get("text") or body.get("textMessage", {}).get("text") or "")
            return httpx.Response(
                200,
                json={
                    "key": {
                        "id": "BAE5OUTBOUND002",
                        "remoteJid": PATIENT_JID,
                        "fromMe": True,
                    },
                    "status": "PENDING",
                },
            )

        transport = httpx.MockTransport(_capture_evolution)
        original_send_text = iris_pipeline.send_text_message

        async def _patched_send(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            kwargs.setdefault("base_url", "https://evo.test")
            async with httpx.AsyncClient(transport=transport) as http:
                return await original_send_text(*args, client=http, **kwargs)

        monkeypatch.setattr(iris_pipeline, "send_text_message", _patched_send)

        # 4) Drive the pipeline
        parsed = ParsedMessage(
            instance_name="iris-prod",
            remote_jid=PATIENT_JID,
            phone="5511999990000",
            wamid="wamid-e2e-002",
            push_name="Maria",
            message_content="Quanto custa o botox? Posso fazer estando grávida?",
            message_type="text",
        )

        try:
            result = asyncio.run(
                iris_pipeline.invoke(
                    clinic_id=CLINIC_ID,
                    message_id="msg-e2e-002",
                    parsed=parsed,
                    trace_id="trace-e2e-002",
                )
            )
        finally:
            iris_pipeline.SPECIALIST_REGISTRY.clear()
            iris_pipeline.SPECIALIST_REGISTRY.update(original_registry)

        # ---- router produced the multi-intent shape ----
        assert result["status"] == "ok"
        assert [i["macro_state"] for i in result["intents"]] == [
            "FAQ",
            "HUMAN_ESCALATION",
        ]
        assert result["primary_intent"] == "HUMAN_ESCALATION"

        # ---- fan-out produced one run per intent ----
        specialist_runs = [
            run
            for run in result["agent_runs"]
            if run.get("agent") != "IrisRouterAgent"
        ]
        assert len(specialist_runs) == 2, (
            f"expected one specialist run per intent, got {specialist_runs!r}"
        )
        # First intent (FAQ informational) wired to the fake FAQResponder;
        # second intent (HUMAN_ESCALATION) falls back deterministically until
        # the escalation specialist ships.
        run_agents = [run.get("agent") for run in specialist_runs]
        assert run_agents == ["FAQResponder", "UnknownFallback"]

        # ---- aggregator emitted one outbound message containing both replies ----
        assert len(outbound_payloads) == 1, (
            f"aggregator must collapse N intents into 1 send, got "
            f"{len(outbound_payloads)} sends"
        )
        aggregated = outbound_payloads[0]
        assert "Botox" in aggregated, aggregated
        assert iris_pipeline.UNKNOWN_FALLBACK_TEXT in aggregated, aggregated

        # ---- audit trail captures both specialists ----
        activated = [row.get("agent_name") for row in supabase_fake.activations]
        assert "IrisRouterAgent" in activated
        assert "FAQResponder" in activated
        assert "UnknownFallback" in activated
