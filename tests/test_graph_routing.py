"""
Testes unitários — Funções puras do `app.graph.sofia_graph`.

Cobre `_extract_service_names`. As suítes anteriores (`_route_after_intent`,
`node_human_escalation`) ficaram órfãs após o refactor que substituiu a
máquina de roteamento por `node_detect_intents` + `node_execute_agents`.
"""

import json

from app.graph.sofia_graph import _extract_service_names


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
        ctx = json.dumps({"services": services, "offers": []})
        result = _extract_service_names(ctx)
        assert len(result) == 10
