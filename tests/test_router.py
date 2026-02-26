"""
Testes unitários — SofiaRouterAgent

Cobre:
  - _parse_intent: valores válidos, inválidos, embedded, case
  - _parse_confidence: float, string numérica, embedded, inválido
  - _format_history: vazio, único turno, multi-turno, truncamento
  - forward(): resposta normal, intent inválido, exceção do LLM
"""

import pytest
from unittest.mock import patch, MagicMock

from app.agents.router.agent import SofiaRouterAgent
from app.agents.router.signatures import SofiaIntentType

from tests.conftest import mock_prediction


@pytest.fixture
def agent():
    return SofiaRouterAgent()


# ─── _parse_intent ────────────────────────────────────────────────────────────

class TestParseIntent:

    def test_valid_greeting(self, agent):
        assert agent._parse_intent("GREETING") == "GREETING"

    def test_valid_faq(self, agent):
        assert agent._parse_intent("FAQ") == "FAQ"

    def test_valid_schedule(self, agent):
        assert agent._parse_intent("SCHEDULE") == "SCHEDULE"

    def test_valid_human_escalation(self, agent):
        assert agent._parse_intent("HUMAN_ESCALATION") == "HUMAN_ESCALATION"

    def test_valid_reengage(self, agent):
        assert agent._parse_intent("REENGAGE") == "REENGAGE"

    def test_valid_unclassified(self, agent):
        assert agent._parse_intent("UNCLASSIFIED") == "UNCLASSIFIED"

    def test_lowercase_normalized(self, agent):
        assert agent._parse_intent("schedule") == "SCHEDULE"

    def test_mixed_case_normalized(self, agent):
        assert agent._parse_intent("Schedule") == "SCHEDULE"

    def test_embedded_in_phrase(self, agent):
        # LLM às vezes responde "The intent is SCHEDULE"
        assert agent._parse_intent("The intent is SCHEDULE here") == "SCHEDULE"

    def test_embedded_human_escalation(self, agent):
        assert agent._parse_intent("Intent: HUMAN_ESCALATION") == "HUMAN_ESCALATION"

    def test_empty_string_returns_unclassified(self, agent):
        assert agent._parse_intent("") == "UNCLASSIFIED"

    def test_garbage_returns_unclassified(self, agent):
        assert agent._parse_intent("banana_fofoca_123") == "UNCLASSIFIED"

    def test_none_returns_unclassified(self, agent):
        assert agent._parse_intent(None) == "UNCLASSIFIED"

    def test_integer_returns_unclassified(self, agent):
        assert agent._parse_intent(42) == "UNCLASSIFIED"


# ─── _parse_confidence ────────────────────────────────────────────────────────

class TestParseConfidence:

    def test_float_passthrough(self, agent):
        assert agent._parse_confidence(0.9) == pytest.approx(0.9)

    def test_int_converted(self, agent):
        assert agent._parse_confidence(1) == pytest.approx(1.0)

    def test_string_float(self, agent):
        assert agent._parse_confidence("0.85") == pytest.approx(0.85)

    def test_string_embedded(self, agent):
        # LLM pode responder "Confidence: 0.75"
        assert agent._parse_confidence("Confidence: 0.75") == pytest.approx(0.75)

    def test_string_with_trailing_text(self, agent):
        assert agent._parse_confidence("0.6 (high)") == pytest.approx(0.6)

    def test_empty_string_returns_zero(self, agent):
        assert agent._parse_confidence("") == pytest.approx(0.0)

    def test_none_returns_zero(self, agent):
        assert agent._parse_confidence(None) == pytest.approx(0.0)

    def test_garbage_returns_zero(self, agent):
        assert agent._parse_confidence("muito confiante") == pytest.approx(0.0)


# ─── _format_history ─────────────────────────────────────────────────────────

class TestFormatHistory:

    def test_empty_history(self, agent):
        result = agent._format_history([])
        assert result == "Sem histórico anterior."

    def test_human_turn_label(self, agent):
        history = [{"role": "human", "content": "oi"}]
        result = agent._format_history(history)
        assert "Paciente: oi" in result

    def test_agent_turn_uses_agent_name(self, agent):
        history = [{"role": "FAQResponder", "content": "Olá!"}]
        result = agent._format_history(history)
        assert "FAQResponder: Olá!" in result

    def test_multi_turn_ordered(self, agent):
        history = [
            {"role": "human", "content": "quero agendar"},
            {"role": "Scheduler", "content": "Qual serviço?"},
        ]
        result = agent._format_history(history)
        assert result.index("Paciente") < result.index("Scheduler")

    def test_truncates_to_last_20_turns(self, agent):
        # 25 turnos → deve retornar apenas os últimos 20
        history = [{"role": "human", "content": f"msg {i}"} for i in range(25)]
        result = agent._format_history(history)
        lines = result.strip().split("\n")
        assert len(lines) == 20
        assert "msg 24" in result   # último turno presente
        assert "msg 4" not in result  # turno 4 (fora dos 20 últimos) ausente


# ─── forward() ───────────────────────────────────────────────────────────────

class TestRouterForward:

    def test_normal_schedule_intent(self, agent):
        agent.process = MagicMock(return_value=mock_prediction(
            intent="SCHEDULE",
            confidence="0.95",
            reasoning="Paciente quer agendar.",
        ))
        result = agent.forward("quero marcar", [], "new")
        assert result["intent"] == "SCHEDULE"
        assert result["confidence"] == pytest.approx(0.95)
        assert "reasoning" in result

    def test_normal_faq_intent(self, agent):
        agent.process = MagicMock(return_value=mock_prediction(
            intent="FAQ",
            confidence="0.88",
            reasoning="Pergunta sobre preço.",
        ))
        result = agent.forward("quanto custa?", [], "active")
        assert result["intent"] == "FAQ"

    def test_human_escalation_intent(self, agent):
        agent.process = MagicMock(return_value=mock_prediction(
            intent="HUMAN_ESCALATION",
            confidence="0.99",
            reasoning="Quer atendente.",
        ))
        result = agent.forward("quero falar com atendente", [], "new")
        assert result["intent"] == "HUMAN_ESCALATION"

    def test_invalid_intent_falls_back_to_unclassified(self, agent):
        agent.process = MagicMock(return_value=mock_prediction(
            intent="BANANA",
            confidence="0.5",
            reasoning="?",
        ))
        result = agent.forward("hmm", [], "new")
        assert result["intent"] == "UNCLASSIFIED"

    def test_llm_exception_returns_unclassified(self, agent):
        agent.process = MagicMock(side_effect=Exception("OpenAI timeout"))
        result = agent.forward("oi", [], "new")
        assert result["intent"] == "UNCLASSIFIED"
        assert result["confidence"] == pytest.approx(0.0)

    def test_passes_history_and_stage(self, agent):
        agent.process = MagicMock(return_value=mock_prediction(
            intent="SCHEDULE", confidence="0.9", reasoning="ok",
        ))
        history = [{"role": "human", "content": "quero agendar"}]
        agent.forward("o primeiro horário", history, "presenting_slots")
        call_kwargs = agent.process.call_args.kwargs
        assert "history_str" in call_kwargs
        assert "presenting_slots" in call_kwargs["conversation_stage"]
