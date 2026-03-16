"""
Tests for load_resource_for_service in app/session/manager.py
"""
import pytest
from unittest.mock import patch, MagicMock
from app.session.manager import load_resource_for_service


def _make_supabase_mock(resources: list):
    """Helper: retorna mock do supabase que devolve `resources` para sf_resources query."""
    mock_result = MagicMock()
    mock_result.data = resources

    mock_table = MagicMock()
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.execute.return_value = mock_result

    mock_sb = MagicMock()
    mock_sb.table.return_value = mock_table
    return mock_sb


CLINIC_ID = "0d6d8eaf-6efa-4aaf-9845-de4b0d0f608c"


class TestLoadResourceForService:

    def test_returns_none_when_no_resources(self):
        """Fallback gracioso: tabela vazia nao quebra o fluxo."""
        with patch("app.session.manager.get_supabase", return_value=_make_supabase_mock([])):
            result = load_resource_for_service(CLINIC_ID)
        assert result is None

    def test_returns_single_resource_directly(self):
        """Caso mais comum: uma clinica tem apenas 1 resource."""
        resource = {"id": "res-001", "name": "Agenda Geral", "type": "generic"}
        with patch("app.session.manager.get_supabase", return_value=_make_supabase_mock([resource])):
            result = load_resource_for_service(CLINIC_ID)
        assert result == resource

    def test_prefers_generic_when_multiple(self):
        """Com multiplos resources, retorna o generic primeiro."""
        resources = [
            {"id": "res-001", "name": "Dra. Ana", "type": "professional"},
            {"id": "res-002", "name": "Agenda Geral", "type": "generic"},
        ]
        with patch("app.session.manager.get_supabase", return_value=_make_supabase_mock(resources)):
            result = load_resource_for_service(CLINIC_ID)
        assert result["type"] == "generic"
        assert result["id"] == "res-002"

    def test_returns_first_when_no_generic(self):
        """Sem generic, retorna o primeiro da lista."""
        resources = [
            {"id": "res-001", "name": "Dra. Ana", "type": "professional"},
            {"id": "res-002", "name": "Sala A", "type": "room"},
        ]
        with patch("app.session.manager.get_supabase", return_value=_make_supabase_mock(resources)):
            result = load_resource_for_service(CLINIC_ID)
        assert result["id"] == "res-001"

    def test_returns_none_on_exception(self):
        """Excecao no Supabase nao propaga — retorna None."""
        mock_sb = MagicMock()
        mock_sb.table.side_effect = Exception("connection error")
        with patch("app.session.manager.get_supabase", return_value=mock_sb):
            result = load_resource_for_service(CLINIC_ID)
        assert result is None
