"""
Shared fixtures para os testes unitários da Sofia.
Todos os testes rodam sem OpenAI e sem Supabase.
"""

import pytest
from unittest.mock import MagicMock


def mock_prediction(**kwargs):
    """Cria um objeto de prediction DSPy falso com atributos arbitrários."""
    pred = MagicMock()
    for key, value in kwargs.items():
        setattr(pred, key, value)
    return pred


@pytest.fixture
def base_state():
    """SofiaState mínimo válido para testes de graph e agentes."""
    return {
        "instance_id": "test-instance",
        "clinic_id": "test-clinic-id",
        "remote_jid": "5511999990000@s.whatsapp.net",
        "push_name": "Maria",
        "message": "oi",
        "message_type": "text",
        "wamid": "test-wamid-001",
        "available_slots": [],
        "session_id": "5511999990000@s.whatsapp.net:test-clinic-id",
        "clinic_name": "Clínica Teste",
        "assistant_name": "Sofia",
        "services_context": (
            '{"services": [{"name": "Clareamento Dental", "price": 350.0},'
            ' {"name": "Limpeza Dental", "price": 120.0}], "offers": []}'
        ),
        "business_rules": "[]",
        "history": [],
        "conversation_stage": "new",
        "patient_name": "Maria",
        "customer_id": "customer-uuid-123",
        "intent": None,
        "confidence": None,
        "response_message": None,
        "agent_name": None,
        "requires_human": False,
        "appointment_created": None,
        "reasoning": None,
        "processing_time_ms": 0.0,
    }
