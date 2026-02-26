"""
Testes unitários — HumanEscalationAgent

Agente determinístico (sem LLM) — testável diretamente, sem mocking.

Cobre:
  - Mensagem inclui nome do paciente quando disponível
  - Mensagem não inclui "Paciente," quando nome é genérico
  - Mensagem inclui nome da clínica
  - Campos obrigatórios do retorno: requires_human, agent_name, conversation_stage
  - Combinações de patient_name nulo, vazio, "Paciente"
"""

import pytest

from app.agents.human_escalation.agent import HumanEscalationAgent


@pytest.fixture
def agent():
    return HumanEscalationAgent()


# ─── Campos obrigatórios ──────────────────────────────────────────────────────

class TestHumanEscalationFields:

    def test_requires_human_always_true(self, agent):
        result = agent.forward("Maria", "Sofia", "Clínica Teste")
        assert result["requires_human"] is True

    def test_agent_name(self, agent):
        result = agent.forward("Maria", "Sofia", "Clínica Teste")
        assert result["agent_name"] == "HumanEscalation"

    def test_conversation_stage(self, agent):
        result = agent.forward("Maria", "Sofia", "Clínica Teste")
        assert result["conversation_stage"] == "human_escalation"

    def test_response_message_not_empty(self, agent):
        result = agent.forward("Maria", "Sofia", "Clínica Teste")
        assert result["response_message"].strip() != ""

    def test_reasoning_present(self, agent):
        result = agent.forward("Maria", "Sofia", "Clínica Teste")
        assert "reasoning" in result


# ─── Mensagem com nome do paciente ────────────────────────────────────────────

class TestHumanEscalationMessage:

    def test_patient_name_included_in_message(self, agent):
        result = agent.forward("João", "Sofia", "Sorriso Da Gente")
        assert "João" in result["response_message"]

    def test_clinic_name_included_in_message(self, agent):
        result = agent.forward("João", "Sofia", "Sorriso Da Gente")
        assert "Sorriso Da Gente" in result["response_message"]

    def test_generic_name_paciente_not_included_as_greeting(self, agent):
        result = agent.forward("Paciente", "Sofia", "Clínica Teste")
        # "Paciente," não deve aparecer como saudação no início
        assert not result["response_message"].startswith("Paciente,")

    def test_empty_name_no_greeting(self, agent):
        result = agent.forward("", "Sofia", "Clínica Teste")
        # Mensagem não deve começar com ","
        assert not result["response_message"].startswith(",")

    def test_none_name_no_greeting(self, agent):
        result = agent.forward(None, "Sofia", "Clínica Teste")
        assert not result["response_message"].startswith(",")

    def test_real_name_starts_message(self, agent):
        result = agent.forward("Ana Paula", "Sofia", "Clínica Saúde")
        assert result["response_message"].startswith("Ana Paula,")

    def test_different_clinic_names(self, agent):
        clinics = ["Sorriso Da Gente", "Clínica Bella", "OdontoVida"]
        for clinic in clinics:
            result = agent.forward("Maria", "Sofia", clinic)
            assert clinic in result["response_message"]
