"""
Testes unitários — SchedulerAgent

Cobre:
  - _parse_stage: valores válidos, inválidos, fallback para stage atual
  - _parse_slot: 4 formatos ISO, regex, variantes de null
  - _parse_service: string válida, null, none, vazio
  - _humanize_slot: formatação para o paciente
  - forward(): guard booked-sem-slot, guard presenting-sem-slots, exceção LLM
"""

import pytest
from unittest.mock import MagicMock

from app.agents.scheduler.agent import SchedulerAgent

from tests.conftest import mock_prediction


@pytest.fixture
def agent():
    return SchedulerAgent()


# ─── _parse_stage ─────────────────────────────────────────────────────────────

class TestParseStage:

    def test_collecting_service(self, agent):
        assert agent._parse_stage("collecting_service", "new") == "collecting_service"

    def test_presenting_slots(self, agent):
        assert agent._parse_stage("presenting_slots", "collecting_service") == "presenting_slots"

    def test_confirming(self, agent):
        assert agent._parse_stage("confirming", "presenting_slots") == "confirming"

    def test_booked(self, agent):
        assert agent._parse_stage("booked", "confirming") == "booked"

    def test_invalid_returns_current_stage(self, agent):
        assert agent._parse_stage("invalid_stage", "collecting_service") == "collecting_service"

    def test_empty_string_returns_current_stage(self, agent):
        assert agent._parse_stage("", "presenting_slots") == "presenting_slots"

    def test_none_returns_current_stage(self, agent):
        assert agent._parse_stage(None, "confirming") == "confirming"

    def test_uppercase_normalized(self, agent):
        assert agent._parse_stage("BOOKED", "confirming") == "booked"


# ─── _parse_slot ──────────────────────────────────────────────────────────────

class TestParseSlot:

    def test_iso_with_seconds(self, agent):
        assert agent._parse_slot("2026-03-10T09:00:00") == "2026-03-10 09:00"

    def test_iso_without_seconds(self, agent):
        assert agent._parse_slot("2026-03-10T09:00") == "2026-03-10 09:00"

    def test_space_separator_with_seconds(self, agent):
        assert agent._parse_slot("2026-03-10 09:00:00") == "2026-03-10 09:00"

    def test_space_separator_without_seconds(self, agent):
        assert agent._parse_slot("2026-03-10 09:00") == "2026-03-10 09:00"

    def test_regex_fallback_mixed_format(self, agent):
        # LLM pode retornar texto como "Horário: 2026-03-10 09:00 confirmado"
        assert agent._parse_slot("Horário: 2026-03-10 09:00 confirmado") == "2026-03-10 09:00"

    def test_null_string_returns_none(self, agent):
        assert agent._parse_slot("null") is None

    def test_none_returns_none(self, agent):
        assert agent._parse_slot(None) is None

    def test_none_string_returns_none(self, agent):
        assert agent._parse_slot("none") is None

    def test_empty_string_returns_none(self, agent):
        assert agent._parse_slot("") is None

    def test_garbage_returns_none(self, agent):
        assert agent._parse_slot("nenhum horário disponível") is None


# ─── _parse_service ───────────────────────────────────────────────────────────

class TestParseService:

    def test_valid_service_name(self, agent):
        assert agent._parse_service("Clareamento Dental") == "Clareamento Dental"

    def test_trims_whitespace(self, agent):
        assert agent._parse_service("  Limpeza  ") == "Limpeza"

    def test_null_string_returns_none(self, agent):
        assert agent._parse_service("null") is None

    def test_none_returns_none(self, agent):
        assert agent._parse_service(None) is None

    def test_none_string_returns_none(self, agent):
        assert agent._parse_service("none") is None

    def test_empty_returns_none(self, agent):
        assert agent._parse_service("") is None


# ─── _humanize_slot ───────────────────────────────────────────────────────────

class TestHumanizeSlot:

    def test_format_thursday(self, agent):
        # 2026-02-26 é quinta-feira
        result = agent._humanize_slot("2026-02-26 09:00")
        assert "Qui" in result
        assert "26/02" in result
        assert "09h" in result
        assert "(2026-02-26 09:00)" in result  # ISO preservado para o LLM extrair

    def test_format_monday(self, agent):
        # 2026-03-02 é segunda-feira
        result = agent._humanize_slot("2026-03-02 14:00")
        assert "Seg" in result
        assert "02/03" in result
        assert "14h" in result

    def test_invalid_slot_returns_original(self, agent):
        assert agent._humanize_slot("not-a-date") == "not-a-date"


# ─── forward() — guards e fallbacks ──────────────────────────────────────────

class TestSchedulerForwardGuards:

    def test_booked_without_chosen_slot_downgraded_to_confirming(self, agent):
        """Guard crítico: não pode bookar sem slot — deve regredir para confirming."""
        agent.process = MagicMock(return_value=mock_prediction(
            response_message="Agendado!",
            stage="booked",
            chosen_slot="null",   # LLM não extraiu o slot
            service_requested="Clareamento Dental",
            reasoning="ok",
        ))
        result = agent.forward(
            patient_message="sim confirmo",
            history=[],
            available_slots=["2026-03-10 09:00"],
            clinic_name="Clínica",
            patient_name="Maria",
            stage="confirming",
        )
        assert result["conversation_stage"] == "confirming"
        assert result["chosen_slot"] is None

    def test_presenting_slots_without_available_downgraded_to_collecting(self, agent):
        """Guard: não pode apresentar slots se não houver nenhum."""
        agent.process = MagicMock(return_value=mock_prediction(
            response_message="Aqui os horários:",
            stage="presenting_slots",
            chosen_slot="null",
            service_requested="Clareamento Dental",
            reasoning="ok",
        ))
        result = agent.forward(
            patient_message="quero agendar clareamento",
            history=[],
            available_slots=[],   # sem slots disponíveis
            clinic_name="Clínica",
            patient_name="Maria",
            stage="collecting_service",
        )
        assert result["conversation_stage"] == "collecting_service"

    def test_valid_booked_with_slot(self, agent):
        """Fluxo feliz: booked com slot válido."""
        agent.process = MagicMock(return_value=mock_prediction(
            response_message="Agendado!",
            stage="booked",
            chosen_slot="2026-03-10 09:00",
            service_requested="Clareamento Dental",
            reasoning="ok",
        ))
        result = agent.forward(
            patient_message="sim confirmo",
            history=[],
            available_slots=["2026-03-10 09:00"],
            clinic_name="Clínica",
            patient_name="Maria",
            stage="confirming",
        )
        assert result["conversation_stage"] == "booked"
        assert result["chosen_slot"] == "2026-03-10 09:00"
        assert result["service_requested"] == "Clareamento Dental"

    def test_stage_collecting_to_presenting(self, agent):
        agent.process = MagicMock(return_value=mock_prediction(
            response_message="Aqui os horários disponíveis:",
            stage="presenting_slots",
            chosen_slot="null",
            service_requested="Limpeza Dental",
            reasoning="Serviço identificado.",
        ))
        result = agent.forward(
            patient_message="quero limpeza",
            history=[],
            available_slots=["2026-03-10 09:00", "2026-03-10 10:00"],
            clinic_name="Clínica",
            patient_name="Maria",
            stage="collecting_service",
        )
        assert result["conversation_stage"] == "presenting_slots"
        assert result["service_requested"] == "Limpeza Dental"

    def test_non_scheduling_stage_mapped_to_collecting(self, agent):
        """Stage 'new' ou 'active' deve ser mapeado para collecting_service."""
        agent.process = MagicMock(return_value=mock_prediction(
            response_message="Qual serviço?",
            stage="collecting_service",
            chosen_slot="null",
            service_requested="null",
            reasoning="Iniciando agendamento.",
        ))
        result = agent.forward(
            patient_message="quero marcar",
            history=[],
            available_slots=["2026-03-10 09:00"],
            clinic_name="Clínica",
            patient_name="Maria",
            stage="new",  # stage inválido para scheduler
        )
        assert result["agent_name"] == "Scheduler"
        assert result["requires_human"] is False

    def test_llm_exception_returns_fallback(self, agent):
        agent.process = MagicMock(side_effect=Exception("Timeout"))
        result = agent.forward(
            patient_message="quero agendar",
            history=[],
            available_slots=[],
            clinic_name="Clínica",
            patient_name="Maria",
            stage="collecting_service",
        )
        assert result["agent_name"] == "Scheduler"
        assert result["response_message"] != ""
        assert result["conversation_stage"] == "collecting_service"  # mantém stage atual
