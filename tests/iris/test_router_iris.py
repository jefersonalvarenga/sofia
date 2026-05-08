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
    IrisRouterAgent,
    IrisRouterOutput,
)
from app.agents.router.signatures import SofiaIntentType


EVAL_CASES_PATH = Path(__file__).resolve().parents[1] / "eval_cases.json"


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
        assert {"detected_intents", "language", "reasoning", "confidence"} <= required

    def test_pydantic_round_trip(self):
        out = IrisRouterOutput(
            detected_intents=[SofiaIntentType.GREETING],
            language="pt-BR",
            reasoning="ok",
            confidence=0.9,
        )
        assert out.detected_intents[0] == SofiaIntentType.GREETING


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
        result = router._normalize_intents(["SCHEDULE", "FAQ"])
        assert result == ["FAQ", "SCHEDULE"]

    def test_human_escalation_always_last(self, router):
        result = router._normalize_intents(["HUMAN_ESCALATION", "FAQ", "SCHEDULE"])
        assert result[-1] == "HUMAN_ESCALATION"

    def test_dedups_preserving_order(self, router):
        result = router._normalize_intents(["FAQ", "FAQ", "SCHEDULE"])
        assert result == ["FAQ", "SCHEDULE"]

    def test_empty_falls_back_to_unclassified(self, router):
        assert router._normalize_intents([]) == ["UNCLASSIFIED"]

    def test_unknown_intent_dropped(self, router):
        # only FAQ survives → list is non-empty so no UNCLASSIFIED fallback
        assert router._normalize_intents(["BANANA", "FAQ"]) == ["FAQ"]

    def test_accepts_enum_values(self, router):
        result = router._normalize_intents([SofiaIntentType.GREETING, SofiaIntentType.FAQ])
        # GREETING priority 5, FAQ priority 4 → GREETING first (informational), FAQ last
        assert result == ["GREETING", "FAQ"]


# ─── forward() ────────────────────────────────────────────────────────────────

class TestForward:

    def test_calls_anthropic_with_correct_params(self, router):
        router.client.messages.create.return_value = _make_tool_use_response({
            "detected_intents": ["GREETING"],
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
            "detected_intents": ["GREETING"],
            "language": "pt-BR",
            "reasoning": "Saudação.",
            "confidence": 0.9,
        })
        result = router.forward("oi", [], "new")
        assert result["detected_intents"] == ["GREETING"]
        assert result["language"] == "pt-BR"
        assert result["confidence"] == pytest.approx(0.9)
        assert "reasoning" in result

    def test_multi_intent_priority_order(self, router):
        # LLM may return in any order — normalizer must sort CTA last
        router.client.messages.create.return_value = _make_tool_use_response({
            "detected_intents": ["SCHEDULE", "FAQ"],
            "language": "pt-BR",
            "reasoning": "Quer agendar e pergunta convênio.",
            "confidence": 0.88,
        })
        result = router.forward("quero agendar limpeza, aceitam Unimed?", [], "new")
        assert result["detected_intents"] == ["FAQ", "SCHEDULE"]

    def test_human_escalation_last(self, router):
        router.client.messages.create.return_value = _make_tool_use_response({
            "detected_intents": ["FAQ", "HUMAN_ESCALATION"],
            "language": "pt-BR",
            "reasoning": "Pergunta + pedir atendente.",
            "confidence": 0.95,
        })
        result = router.forward("preço? quero falar com atendente", [], "new")
        assert result["detected_intents"][-1] == "HUMAN_ESCALATION"

    def test_anthropic_exception_returns_unclassified(self, router):
        router.client.messages.create.side_effect = RuntimeError("API timeout")
        result = router.forward("oi", [], "new")
        assert result["detected_intents"] == ["UNCLASSIFIED"]
        assert result["language"] == "pt-BR"
        assert result["confidence"] == pytest.approx(0.0)
        assert "Erro" in result["reasoning"]

    def test_missing_tool_use_returns_unclassified(self, router):
        router.client.messages.create.return_value = _make_text_only_response()
        result = router.forward("oi", [], "new")
        assert result["detected_intents"] == ["UNCLASSIFIED"]
        assert result["confidence"] == pytest.approx(0.0)

    def test_invalid_intent_payload_dropped(self, router):
        # BANANA is not a valid IntentEnum → pydantic validation fails → fallback
        router.client.messages.create.return_value = _make_tool_use_response({
            "detected_intents": ["BANANA"],
            "language": "pt-BR",
            "reasoning": "?",
            "confidence": 0.5,
        })
        result = router.forward("hmm", [], "new")
        assert result["detected_intents"] == ["UNCLASSIFIED"]

    def test_confidence_out_of_range_clamped(self, router):
        # Pydantic enforces 0..1 — out-of-range raises and the agent returns fallback.
        router.client.messages.create.return_value = _make_tool_use_response({
            "detected_intents": ["GREETING"],
            "language": "pt-BR",
            "reasoning": "ok",
            "confidence": 1.5,
        })
        result = router.forward("oi", [], "new")
        assert result["detected_intents"] == ["UNCLASSIFIED"]

    def test_history_passed_in_user_prompt(self, router):
        router.client.messages.create.return_value = _make_tool_use_response({
            "detected_intents": ["SCHEDULE"],
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
            "detected_intents": ["GREETING"],
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
            "detected_intents": [case["expected_intent"]],
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
