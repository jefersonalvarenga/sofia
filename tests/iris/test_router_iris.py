"""
Unit tests for IrisRouterAgent (Anthropic SDK + tool use Pydantic).

The Anthropic client is mocked end-to-end — no network calls, no real API key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from app.agents.router.agent_iris import (
    CLASSIFY_TOOL,
    IRIS_ROUTER_MODEL,
    Intent,
    IrisRouterAgent,
    IrisRouterOutput,
)
from app.agents.router.signatures import SofiaIntentType


EVAL_CASES_PATH = Path(__file__).resolve().parents[1] / "eval_cases.json"


def _intent(macro: str, scope: str) -> Dict[str, str]:
    return {"macro_state": macro, "scope_text": scope}


def _make_tool_use_response(tool_input: Dict[str, Any]) -> Any:
    """Build a fake Anthropic Message with one tool_use block named classify_intent."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "classify_intent"
    block.input = tool_input

    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock(input_tokens=120, output_tokens=30)
    return response


def _make_text_only_response(text: str = "no tool here") -> Any:
    block = MagicMock()
    block.type = "text"
    block.text = text

    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock(input_tokens=10, output_tokens=5)
    return response


@pytest.fixture
def router() -> IrisRouterAgent:
    client = MagicMock()
    client.messages.create = MagicMock()
    return IrisRouterAgent(client=client)


# ─── Tool schema sanity ───────────────────────────────────────────────────────

class TestClassifyToolSchema:

    def test_tool_name(self):
        assert CLASSIFY_TOOL["name"] == "classify_intent"

    def test_required_fields(self):
        required = set(CLASSIFY_TOOL["input_schema"]["required"])
        assert {"intents", "language", "reasoning", "confidence"} <= required

    def test_pydantic_round_trip(self):
        out = IrisRouterOutput(
            intents=[Intent(macro_state=SofiaIntentType.GREETING, scope_text="oi")],
            language="pt-BR",
            reasoning="ok",
            confidence=0.9,
        )
        assert out.intents[0].macro_state == SofiaIntentType.GREETING
        assert out.intents[0].scope_text == "oi"

    def test_intent_requires_scope_text(self):
        # Empty scope_text is rejected — every intent must point at the slice
        # of the message that triggered it.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Intent(macro_state=SofiaIntentType.FAQ, scope_text="")


# ─── _format_history ──────────────────────────────────────────────────────────

class TestFormatHistory:

    def test_empty_history(self, router):
        assert router._format_history([]) == "Sem histórico anterior."

    def test_human_label(self, router):
        result = router._format_history([{"role": "human", "content": "oi"}])
        assert "Paciente: oi" in result

    def test_agent_label_preserved(self, router):
        result = router._format_history([{"role": "FAQResponder", "content": "Olá!"}])
        assert "FAQResponder: Olá!" in result

    def test_truncates_to_last_20(self, router):
        history = [{"role": "human", "content": f"msg {i}"} for i in range(25)]
        result = router._format_history(history)
        lines = result.strip().split("\n")
        assert len(lines) == 20
        assert "msg 24" in result
        assert "msg 4" not in result


# ─── _normalize_intents ───────────────────────────────────────────────────────

class TestNormalizeIntents:

    def test_priority_sort_cta_last(self, router):
        # FAQ informational, SCHEDULE CTA → CTA last
        result = router._normalize_intents(
            [_intent("SCHEDULE", "quero agendar"), _intent("FAQ", "aceitam Unimed?")],
            latest_message="quero agendar, aceitam Unimed?",
        )
        assert [i["macro_state"] for i in result] == ["FAQ", "SCHEDULE"]
        # scope_text travels with each intent through the sort
        assert result[0]["scope_text"] == "aceitam Unimed?"
        assert result[1]["scope_text"] == "quero agendar"

    def test_human_escalation_always_last(self, router):
        result = router._normalize_intents(
            [
                _intent("HUMAN_ESCALATION", "quero atendente"),
                _intent("FAQ", "preço?"),
                _intent("SCHEDULE", "marcar"),
            ],
            latest_message="...",
        )
        assert result[-1]["macro_state"] == "HUMAN_ESCALATION"

    def test_dedups_preserving_first_scope(self, router):
        # When the LLM emits the same macro_state twice, keep the first scope.
        result = router._normalize_intents(
            [
                _intent("FAQ", "preço?"),
                _intent("FAQ", "horário?"),
                _intent("SCHEDULE", "marcar"),
            ],
            latest_message="...",
        )
        assert [i["macro_state"] for i in result] == ["FAQ", "SCHEDULE"]
        faq = next(i for i in result if i["macro_state"] == "FAQ")
        assert faq["scope_text"] == "preço?"

    def test_empty_falls_back_to_unclassified_with_full_message(self, router):
        result = router._normalize_intents([], latest_message="blá blá")
        assert result == [{"macro_state": "UNCLASSIFIED", "scope_text": "blá blá"}]

    def test_unknown_intent_dropped(self, router):
        # only FAQ survives → list is non-empty so no UNCLASSIFIED fallback
        result = router._normalize_intents(
            [_intent("BANANA", "?"), _intent("FAQ", "preço?")],
            latest_message="preço?",
        )
        assert [i["macro_state"] for i in result] == ["FAQ"]

    def test_accepts_intent_pydantic(self, router):
        # _normalize_intents must accept Intent models, not just dicts —
        # `forward()` hands it parsed Intent instances directly.
        result = router._normalize_intents(
            [
                Intent(macro_state=SofiaIntentType.GREETING, scope_text="oi"),
                Intent(macro_state=SofiaIntentType.FAQ, scope_text="preço?"),
            ],
            latest_message="oi, preço?",
        )
        # GREETING priority 5, FAQ priority 4 → GREETING first, FAQ last
        assert [i["macro_state"] for i in result] == ["GREETING", "FAQ"]

    def test_missing_scope_text_falls_back_to_message(self, router):
        result = router._normalize_intents(
            [{"macro_state": "FAQ", "scope_text": ""}],
            latest_message="preço?",
        )
        assert result[0]["scope_text"] == "preço?"


# ─── forward() ────────────────────────────────────────────────────────────────

class TestForward:

    def test_calls_anthropic_with_correct_params(self, router):
        router.client.messages.create.return_value = _make_tool_use_response({
            "intents": [_intent("GREETING", "oi")],
            "language": "pt-BR",
            "reasoning": "Saudação simples.",
            "confidence": 0.95,
        })
        router.forward("oi", [], "new")
        kwargs = router.client.messages.create.call_args.kwargs
        assert kwargs["model"] == IRIS_ROUTER_MODEL
        assert kwargs["tool_choice"] == {"type": "tool", "name": "classify_intent"}
        assert kwargs["tools"][0]["name"] == "classify_intent"
        assert "system" in kwargs and "Iris" in kwargs["system"]
        assert kwargs["messages"][0]["role"] == "user"
        assert "oi" in kwargs["messages"][0]["content"]
        assert "new" in kwargs["messages"][0]["content"]

    def test_normal_greeting(self, router):
        router.client.messages.create.return_value = _make_tool_use_response({
            "intents": [_intent("GREETING", "oi")],
            "language": "pt-BR",
            "reasoning": "Saudação.",
            "confidence": 0.9,
        })
        result = router.forward("oi", [], "new")
        assert result["intents"] == [{"macro_state": "GREETING", "scope_text": "oi"}]
        assert result["detected_intents"] == ["GREETING"]
        assert result["language"] == "pt-BR"
        assert result["confidence"] == pytest.approx(0.9)
        assert "reasoning" in result

    def test_multi_intent_priority_order(self, router):
        # LLM may return in any order — normalizer must sort CTA last
        router.client.messages.create.return_value = _make_tool_use_response({
            "intents": [
                _intent("SCHEDULE", "quero agendar limpeza"),
                _intent("FAQ", "aceitam Unimed?"),
            ],
            "language": "pt-BR",
            "reasoning": "Quer agendar e pergunta convênio.",
            "confidence": 0.88,
        })
        result = router.forward("quero agendar limpeza, aceitam Unimed?", [], "new")
        assert result["detected_intents"] == ["FAQ", "SCHEDULE"]
        assert result["intents"][0]["scope_text"] == "aceitam Unimed?"
        assert result["intents"][1]["scope_text"] == "quero agendar limpeza"

    def test_human_escalation_last(self, router):
        router.client.messages.create.return_value = _make_tool_use_response({
            "intents": [
                _intent("FAQ", "preço?"),
                _intent("HUMAN_ESCALATION", "quero falar com atendente"),
            ],
            "language": "pt-BR",
            "reasoning": "Pergunta + pedir atendente.",
            "confidence": 0.95,
        })
        result = router.forward("preço? quero falar com atendente", [], "new")
        assert result["detected_intents"][-1] == "HUMAN_ESCALATION"
        # scope_text on the escalation intent points at the escalation slice
        escalation = next(
            i for i in result["intents"] if i["macro_state"] == "HUMAN_ESCALATION"
        )
        assert "atendente" in escalation["scope_text"]

    def test_botox_pregnancy_smoke_shape(self, router):
        """Smoke from [EASAA-140](../../EASAA/issues/EASAA-140):

        Multi-question message yields FAQ (preço) + HUMAN_ESCALATION
        (gravidez = pergunta clínica). Each intent carries the originating
        scope so downstream specialists answer the right slice.
        """
        router.client.messages.create.return_value = _make_tool_use_response({
            "intents": [
                _intent("FAQ", "Quanto custa o botox?"),
                _intent("HUMAN_ESCALATION", "Posso fazer estando grávida?"),
            ],
            "language": "pt-BR",
            "reasoning": "Preço + escalação médica.",
            "confidence": 0.92,
        })
        result = router.forward(
            "Quanto custa o botox? Posso fazer estando grávida?", [], "new"
        )
        assert [i["macro_state"] for i in result["intents"]] == [
            "FAQ",
            "HUMAN_ESCALATION",
        ]
        assert result["intents"][0]["scope_text"] == "Quanto custa o botox?"
        assert result["intents"][1]["scope_text"] == "Posso fazer estando grávida?"

    def test_anthropic_exception_propagates(self, router):
        # Router must NOT swallow API failures — telemetry needs the real error
        # so build_agent_run records status="error" with the cause.
        router.client.messages.create.side_effect = RuntimeError("API timeout")
        with pytest.raises(RuntimeError, match="API timeout"):
            router.forward("oi", [], "new")
        assert router.last_response is None

    def test_missing_tool_use_propagates(self, router):
        router.client.messages.create.return_value = _make_text_only_response()
        with pytest.raises(ValueError, match="classify_intent tool call missing"):
            router.forward("oi", [], "new")

    def test_invalid_intent_payload_propagates(self, router):
        router.client.messages.create.return_value = _make_tool_use_response({
            "intents": [_intent("BANANA", "hmm")],
            "language": "pt-BR",
            "reasoning": "?",
            "confidence": 0.5,
        })
        with pytest.raises(Exception):  # pydantic ValidationError
            router.forward("hmm", [], "new")

    def test_confidence_out_of_range_propagates(self, router):
        router.client.messages.create.return_value = _make_tool_use_response({
            "intents": [_intent("GREETING", "oi")],
            "language": "pt-BR",
            "reasoning": "ok",
            "confidence": 1.5,
        })
        with pytest.raises(Exception):  # pydantic ValidationError on confidence
            router.forward("oi", [], "new")

    def test_history_passed_in_user_prompt(self, router):
        router.client.messages.create.return_value = _make_tool_use_response({
            "intents": [_intent("SCHEDULE", "o primeiro horário")],
            "language": "pt-BR",
            "reasoning": "Mid scheduling.",
            "confidence": 0.9,
        })
        history = [{"role": "human", "content": "quero agendar"}]
        router.forward("o primeiro horário", history, "presenting_slots")
        prompt = router.client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "presenting_slots" in prompt
        assert "Paciente: quero agendar" in prompt

    def test_last_response_captured_for_telemetry(self, router):
        router.client.messages.create.return_value = _make_tool_use_response({
            "intents": [_intent("GREETING", "oi")],
            "language": "pt-BR",
            "reasoning": "ok",
            "confidence": 0.9,
        })
        router.forward("oi", [], "new")
        assert router.last_response is not None
        assert router.last_response.usage.input_tokens == 120


# ─── eval_cases parity ────────────────────────────────────────────────────────

def _load_router_eval_subset(limit: int = 6) -> List[Dict[str, Any]]:
    """Pick a deterministic subset of eval_cases.router with known expected_intent."""
    raw = json.loads(EVAL_CASES_PATH.read_text(encoding="utf-8"))
    cases = [c for c in raw.get("router", []) if c.get("expected_intent")]
    return cases[:limit]


class TestEvalCasesParity:
    """When the LLM returns the canonical answer, IrisRouterAgent must match Sofia output."""

    @pytest.mark.parametrize("case", _load_router_eval_subset(6), ids=lambda c: c["id"])
    def test_case(self, case):
        client = MagicMock()
        client.messages.create.return_value = _make_tool_use_response({
            "intents": [_intent(case["expected_intent"], case["message"])],
            "language": "pt-BR",
            "reasoning": case["description"][:200],
            "confidence": 0.9,
        })
        agent = IrisRouterAgent(client=client)
        result = agent.forward(case["message"], case["history"], case["stage"])
        assert case["expected_intent"] in result["detected_intents"]
        assert result["confidence"] == pytest.approx(0.9)


# ─── extract_tokens_anthropic ─────────────────────────────────────────────────

class TestExtractTokensAnthropic:

    def test_normal_response(self):
        from app.core.telemetry import extract_tokens_anthropic

        response = MagicMock()
        response.usage = MagicMock(input_tokens=100, output_tokens=50)
        tokens = extract_tokens_anthropic(response)
        assert tokens == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    def test_none_response(self):
        from app.core.telemetry import extract_tokens_anthropic

        assert extract_tokens_anthropic(None) == {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }

    def test_missing_usage(self):
        from app.core.telemetry import extract_tokens_anthropic

        response = MagicMock(spec=[])  # no usage attr
        assert extract_tokens_anthropic(response)["total_tokens"] == 0


# ─── C10 deterministic intent table ────────────────────────────────────────────
#
# C10 ([EASAA-31](../../EASAA/issues/EASAA-31)) — fixed table proving the
# IrisRouter contract for the greeting smoke. The Anthropic call is mocked so
# the assertion is on the agent's normalization + handling, not on a live LLM.


class TestC10IntentTable:
    """'oi' → GREETING, 'agendar' → SCHEDULE, 'blá blá' → UNCLASSIFIED."""

    @pytest.mark.parametrize(
        "message, llm_intents, expected",
        [
            ("oi", [("GREETING", "oi")], ["GREETING"]),
            (
                "quero agendar limpeza",
                [("SCHEDULE", "quero agendar limpeza")],
                ["SCHEDULE"],
            ),
            # When the LLM produces a valid UNCLASSIFIED, surface it directly.
            ("blá blá", [("UNCLASSIFIED", "blá blá")], ["UNCLASSIFIED"]),
        ],
        ids=["greeting", "schedule", "unclassified"],
    )
    def test_router_intent_table(self, message, llm_intents, expected):
        client = MagicMock()
        client.messages.create.return_value = _make_tool_use_response({
            "intents": [_intent(m, s) for m, s in llm_intents],
            "language": "pt-BR",
            "reasoning": "c10 fixture",
            "confidence": 0.91,
        })
        agent = IrisRouterAgent(client=client)
        result = agent.forward(message, [], "new")
        assert result["detected_intents"] == expected
