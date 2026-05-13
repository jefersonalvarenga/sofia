"""
Escalation end-to-end smoke test — EASAA-216.

Validates the full escalation path:

  1. Explicit handoff request → HUMAN_ESCALATION, welcome response, paused flag
  2. Clinical urgency → HUMAN_ESCALATION, welcome response, paused flag
  3. Paused conversation → pipeline short-circuits, no processing

Mocks Anthropic (router) + Supabase (session/activations) + Evolution (outbound).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import httpx
import pytest

from app.iris import pipeline as iris_pipeline
from app.iris.schemas import ParsedMessage

CLINIC_ID = "57952a29-e228-4cac-b5fa-3d20ba478f5d"
PATIENT_JID = "5511999990000@s.whatsapp.net"


# ---------------------------------------------------------------------------
# Anthropic stub — emits HUMAN_ESCALATION intent
# ---------------------------------------------------------------------------

def _fake_escalation_response(
    scope_text: str,
    reasoning: str = "escalation trigger",
    language: str = "pt-BR",
    confidence: float = 0.95,
) -> Any:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "classify_intent"
    block.input = {
        "intents": [{"macro_state": "HUMAN_ESCALATION", "scope_text": scope_text}],
        "language": language,
        "reasoning": reasoning,
        "confidence": confidence,
    }
    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock(input_tokens=42, output_tokens=11)
    return response


# ---------------------------------------------------------------------------
# Supabase fake — configurable paused state, captures writes
# ---------------------------------------------------------------------------

class _SupabaseFake:
    def __init__(self, initial_paused: bool = False) -> None:
        self._initial_paused = initial_paused
        self.activations: List[Dict[str, Any]] = []
        self.sessions_updated: List[Dict[str, Any]] = []
        self.sessions_inserted: List[Dict[str, Any]] = []
        self.outbound_messages: List[Dict[str, Any]] = []

    def table(self, name: str) -> "_Table":
        return _Table(self, name)


class _Table:
    def __init__(self, parent: _SupabaseFake, name: str) -> None:
        self._parent = parent
        self._name = name
        self._chain: Dict[str, Any] = {"filters": {}}

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

    def execute(self) -> Any:
        op = self._chain.get("op")

        if op == "insert" and self._name == "sf_sessions":
            self._parent.sessions_inserted.append(self._chain["row"])
            return MagicMock(data=[self._chain["row"]])

        if op == "insert" and self._name == "sf_agent_activations":
            self._parent.activations.append(self._chain["row"])
            return MagicMock(data=[self._chain["row"]])

        if op in ("insert", "upsert") and self._name == "sf_messages":
            self._parent.outbound_messages.append(self._chain["row"])
            return MagicMock(data=[self._chain["row"]])

        if op == "update" and self._name == "sf_sessions":
            self._parent.sessions_updated.append(self._chain["row"])
            return MagicMock(data=[self._chain["row"]])

        if self._name == "sf_customers":
            return MagicMock(data=[{"id": "customer-uuid-001"}])
        if self._name == "sf_clinic_profiles":
            return MagicMock(
                data={"clinic_name": u"Cl\u00ednica Vit\u00f3ria", "assistant_name": "Iris"}
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
            # Return configured paused state
            return MagicMock(data=[{
                "session_id": f"{PATIENT_JID}:{CLINIC_ID}",
                "history": [],
                "conversation_stage": "new",
                "paused": self._parent._initial_paused,
            }])
        if self._name == "sf_instance_clinic_map":
            return MagicMock(data={"clinic_id": CLINIC_ID})

        return MagicMock(data=None)


# ---------------------------------------------------------------------------
# Evolution transport — captures outgoing payloads
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOLUTION_API_KEY", "test-key")
    monkeypatch.setenv("EVOLUTION_API_URL", "https://evo.test")
    monkeypatch.setenv("SOFIA_VERSION", "iris-test")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    from app.core import config as core_config
    monkeypatch.setattr(core_config, "_settings", None, raising=False)


def _setup_mocks(
    monkeypatch: pytest.MonkeyPatch,
    supabase_fake: _SupabaseFake,
    scope_text: str,
    reasoning: str = "escalation",
) -> None:
    fake_client = MagicMock()
    fake_client.messages.create = MagicMock(
        return_value=_fake_escalation_response(
            scope_text=scope_text, reasoning=reasoning
        )
    )
    monkeypatch.setattr(
        iris_pipeline._router_agent, "client", fake_client, raising=True
    )

    monkeypatch.setattr(
        "app.core.supabase_client.get_supabase",
        lambda: supabase_fake,
    )
    monkeypatch.setattr(
        "app.session.manager.get_supabase",
        lambda: supabase_fake,
    )

    transport = httpx.MockTransport(_ok_evolution_response)
    original_send_text = iris_pipeline.send_text_message

    async def _patched_send(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("base_url", "https://evo.test")
        async with httpx.AsyncClient(transport=transport) as http:
            return await original_send_text(*args, client=http, **kwargs)

    monkeypatch.setattr(iris_pipeline, "send_text_message", _patched_send)


def _run_pipeline(message_content: str, push_name: str = "Maria") -> Dict[str, Any]:
    parsed = ParsedMessage(
        instance_name="iris-prod",
        remote_jid=PATIENT_JID,
        phone="5511999990000",
        wamid="wamid-smoke-001",
        push_name=push_name,
        message_content=message_content,
        message_type="text",
    )
    return asyncio.run(
        iris_pipeline.invoke(
            clinic_id=CLINIC_ID,
            message_id="msg-smoke-001",
            parsed=parsed,
            trace_id="trace-smoke-001",
        )
    )


# ============================================================================
# Tests
# ============================================================================


class TestExplicitHandoffRequest:
    """Scenario 1: Patient types "quero falar com uma pessoa"."""

    def test_router_detects_human_escalation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        supabase_fake = _SupabaseFake(initial_paused=False)
        _setup_mocks(monkeypatch, supabase_fake, "quero falar com uma pessoa",
                     reasoning="Paciente pediu atendente")

        result = _run_pipeline("quero falar com uma pessoa")

        assert result["status"] == "ok"
        assert "HUMAN_ESCALATION" in result["detected_intents"]
        assert result["primary_intent"] == "HUMAN_ESCALATION"

    def test_specialist_generates_welcome_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        supabase_fake = _SupabaseFake(initial_paused=False)
        _setup_mocks(monkeypatch, supabase_fake, "quero falar com uma pessoa")

        result = _run_pipeline("quero falar com uma pessoa")

        agent_names = [run.get("agent") for run in result["agent_runs"]]
        assert "HumanEscalation" in agent_names, (
            f"HumanEscalation missing from agent_runs={agent_names}"
        )

        # Check specialist response contains acolhimento content
        for resp in result.get("specialist_responses", []):
            if resp.get("macro_state") == "HUMAN_ESCALATION":
                assert "recepcionista" in resp["response_text"], (
                    f"Expected 'recepcionista' in response: {resp['response_text']}"
                )

    def test_paused_flag_set_on_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        supabase_fake = _SupabaseFake(initial_paused=False)
        _setup_mocks(monkeypatch, supabase_fake, "quero falar com uma pessoa")

        _run_pipeline("quero falar com uma pessoa")

        paused_updates = [
            row for row in supabase_fake.sessions_updated
            if row.get("paused") is True
        ]
        assert len(paused_updates) > 0, (
            "sf_sessions update must include paused=True after escalation"
        )

    def test_escalation_data_logged_in_activation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        supabase_fake = _SupabaseFake(initial_paused=False)
        _setup_mocks(monkeypatch, supabase_fake, "quero falar com uma pessoa")

        _run_pipeline("quero falar com uma pessoa")

        escalation_activations = [
            row for row in supabase_fake.activations
            if row.get("agent_name") == "HumanEscalation"
        ]
        assert len(escalation_activations) > 0, (
            "No HumanEscalation activation row found"
        )

        activation = escalation_activations[0]
        data = activation.get("data")
        assert data is not None, "Activation data is missing"
        assert data.get("type") == "escalation", (
            f"Expected data.type='escalation', got {data.get('type')}"
        )


class TestClinicalUrgency:
    """Scenario 2: Patient reports post-procedure bleeding."""

    def test_router_detects_escalation_for_medical_concern(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        supabase_fake = _SupabaseFake(initial_paused=False)
        _setup_mocks(
            monkeypatch,
            supabase_fake,
            "estou com sangramento apos o procedimento",
            reasoning="Relato clinico requer avaliacao profissional",
        )

        result = _run_pipeline("estou com sangramento apos o procedimento")

        assert result["status"] == "ok"
        assert "HUMAN_ESCALATION" in result["detected_intents"]
        assert result["primary_intent"] == "HUMAN_ESCALATION"

    def test_welcome_message_with_medical_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        supabase_fake = _SupabaseFake(initial_paused=False)
        _setup_mocks(
            monkeypatch,
            supabase_fake,
            "estou com sangramento apos o procedimento",
        )

        result = _run_pipeline("estou com sangramento apos o procedimento")

        agent_names = [run.get("agent") for run in result["agent_runs"]]
        assert "HumanEscalation" in agent_names

    def test_paused_flag_and_escalation_data(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        supabase_fake = _SupabaseFake(initial_paused=False)
        _setup_mocks(
            monkeypatch,
            supabase_fake,
            "estou com sangramento apos o procedimento",
        )

        _run_pipeline("estou com sangramento apos o procedimento")

        # paused=True written
        paused_updates = [
            row for row in supabase_fake.sessions_updated
            if row.get("paused") is True
        ]
        assert len(paused_updates) > 0, "paused flag not set"

        # escalation type in activation
        escalation_activations = [
            row for row in supabase_fake.activations
            if row.get("agent_name") == "HumanEscalation"
        ]
        assert len(escalation_activations) > 0
        assert escalation_activations[0].get("data", {}).get("type") == "escalation"


class TestPausedConversationIgnored:
    """Scenario 3: Message on a paused conversation is silently dropped."""

    def test_pipeline_short_circuits_on_paused(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        supabase_fake = _SupabaseFake(initial_paused=True)
        _setup_mocks(monkeypatch, supabase_fake, "alguem vai me responder?")

        result = _run_pipeline("alguem vai me responder?")

        assert result["status"] == "ok"
        # No intents should be detected because pipeline ends at load_context
        assert result.get("detected_intents") == []
        assert result.get("primary_intent") is None

    def test_no_specialist_runs_on_paused(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        supabase_fake = _SupabaseFake(initial_paused=True)
        _setup_mocks(monkeypatch, supabase_fake, "alguem vai me responder?")

        result = _run_pipeline("alguem vai me responder?")

        agent_runs = result.get("agent_runs", [])
        assert len(agent_runs) == 0, (
            f"Expected 0 agent runs on paused conversation, got {len(agent_runs)}"
        )

    def test_no_outbound_message_sent_on_paused(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        supabase_fake = _SupabaseFake(initial_paused=True)
        _setup_mocks(monkeypatch, supabase_fake, "alguem vai me responder?")

        result = _run_pipeline("alguem vai me responder?")

        assert result.get("response_text") is None, (
            f"Expected no response on paused, got: {result.get('response_text')}"
        )
        assert result.get("outbound_wamid") is None

    def test_no_agent_activations_on_paused(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        supabase_fake = _SupabaseFake(initial_paused=True)
        _setup_mocks(monkeypatch, supabase_fake, "alguem vai me responder?")

        _run_pipeline("alguem vai me responder?")

        assert len(supabase_fake.activations) == 0, (
            f"Expected 0 activations on paused, got {len(supabase_fake.activations)}"
        )
