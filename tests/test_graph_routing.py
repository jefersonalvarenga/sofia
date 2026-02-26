"""
Testes unitários — Lógica de roteamento do LangGraph

Cobre funções puras do sofia_graph.py sem precisar de Supabase ou LLM:
  - _route_after_intent: todos os intents, prioridade HUMAN_ESCALATION,
    fallback por baixa confiança
  - _extract_service_names: JSON válido, vazio, malformado
  - node_human_escalation: integração determinística via state
"""

import pytest

from app.graph.sofia_graph import _route_after_intent, _extract_service_names, node_human_escalation


# ─── _route_after_intent ─────────────────────────────────────────────────────

class TestRouteAfterIntent:

    def _state(self, intent, confidence=0.9):
        return {"intent": intent, "confidence": confidence}

    # Roteamento padrão
    def test_schedule_high_confidence_routes_to_scheduler(self):
        assert _route_after_intent(self._state("SCHEDULE", 0.9)) == "scheduler"

    def test_faq_routes_to_faq_responder(self):
        assert _route_after_intent(self._state("FAQ", 0.9)) == "faq_responder"

    def test_greeting_routes_to_faq_responder(self):
        assert _route_after_intent(self._state("GREETING", 0.95)) == "faq_responder"

    def test_reengage_routes_to_faq_responder(self):
        assert _route_after_intent(self._state("REENGAGE", 0.8)) == "faq_responder"

    def test_unclassified_routes_to_faq_responder(self):
        assert _route_after_intent(self._state("UNCLASSIFIED", 0.3)) == "faq_responder"

    # Prioridade HUMAN_ESCALATION — sempre vence, qualquer confiança
    def test_human_escalation_always_wins(self):
        assert _route_after_intent(self._state("HUMAN_ESCALATION", 0.99)) == "human_escalation"

    def test_human_escalation_even_with_zero_confidence(self):
        assert _route_after_intent(self._state("HUMAN_ESCALATION", 0.0)) == "human_escalation"

    def test_human_escalation_even_with_none_confidence(self):
        assert _route_after_intent({"intent": "HUMAN_ESCALATION", "confidence": None}) == "human_escalation"

    # Fallback por baixa confiança
    def test_schedule_low_confidence_falls_back_to_faq(self):
        # confidence < 0.5 deve cair no faq_responder por segurança
        assert _route_after_intent(self._state("SCHEDULE", 0.3)) == "faq_responder"

    def test_schedule_confidence_exactly_05_goes_to_scheduler(self):
        # 0.5 é o limiar — exatamente 0.5 ainda deve ir pro scheduler
        assert _route_after_intent(self._state("SCHEDULE", 0.5)) == "scheduler"

    def test_schedule_confidence_049_falls_back(self):
        assert _route_after_intent(self._state("SCHEDULE", 0.49)) == "faq_responder"

    def test_missing_confidence_falls_back(self):
        # confidence None → 0.0 → fallback
        assert _route_after_intent({"intent": "SCHEDULE", "confidence": None}) == "faq_responder"

    def test_missing_intent_routes_to_faq(self):
        assert _route_after_intent({"intent": None, "confidence": 0.9}) == "faq_responder"


# ─── _extract_service_names ───────────────────────────────────────────────────

class TestExtractServiceNames:

    def test_extracts_names_from_valid_json(self):
        ctx = '{"services": [{"name": "Clareamento"}, {"name": "Limpeza"}], "offers": []}'
        result = _extract_service_names(ctx)
        assert "Clareamento" in result
        assert "Limpeza" in result

    def test_returns_empty_for_empty_services(self):
        ctx = '{"services": [], "offers": []}'
        assert _extract_service_names(ctx) == []

    def test_returns_empty_for_empty_json(self):
        assert _extract_service_names("{}") == []

    def test_returns_empty_for_invalid_json(self):
        assert _extract_service_names("não é json") == []

    def test_returns_empty_for_empty_string(self):
        assert _extract_service_names("") == []

    def test_skips_services_without_name(self):
        ctx = '{"services": [{"price": 100}, {"name": "Botox"}], "offers": []}'
        result = _extract_service_names(ctx)
        assert result == ["Botox"]

    def test_many_services(self):
        services = [{"name": f"Serviço {i}"} for i in range(10)]
        import json
        ctx = json.dumps({"services": services, "offers": []})
        result = _extract_service_names(ctx)
        assert len(result) == 10


# ─── node_human_escalation ────────────────────────────────────────────────────

class TestNodeHumanEscalation:
    """Testa o nó do grafo diretamente com um state dict."""

    def test_returns_requires_human_true(self, base_state):
        result = node_human_escalation(base_state)
        assert result["requires_human"] is True

    def test_returns_correct_agent_name(self, base_state):
        result = node_human_escalation(base_state)
        assert result["agent_name"] == "HumanEscalation"

    def test_uses_patient_name_from_state(self, base_state):
        base_state["patient_name"] = "Carlos"
        result = node_human_escalation(base_state)
        assert "Carlos" in result["response_message"]

    def test_uses_clinic_name_from_state(self, base_state):
        base_state["clinic_name"] = "OdontoVida"
        result = node_human_escalation(base_state)
        assert "OdontoVida" in result["response_message"]

    def test_falls_back_to_push_name_when_patient_name_missing(self, base_state):
        base_state["patient_name"] = None
        base_state["push_name"] = "Roberto"
        result = node_human_escalation(base_state)
        assert "Roberto" in result["response_message"]
