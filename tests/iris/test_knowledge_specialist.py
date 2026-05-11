"""
Smoke tests — KnowledgeSpecialist (EASAA-143).

Tests run without live Supabase or Anthropic. Anthropic SDK is stubbed via
MagicMock; retrieval is stubbed to return deterministic KB chunks.

Smoke scenario (issue spec): "posso fazer botox grávida?" →
  - sensitive_flag = True
  - requires_consultation = True
  - answer contains "avaliação presencial"
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("DSPY_CACHEDIR", os.path.join(tempfile.gettempdir(), "iris_dspy_cache"))

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app.agents.knowledge.agent import KnowledgeSpecialist, _is_sensitive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOTOX_CHUNKS = [
    {
        "procedure": "Botox",
        "title": "O que é Botox?",
        "body": "Botox (toxina botulínica) é aplicado para relaxar músculos e suavizar rugas.",
    },
    {
        "procedure": "Botox",
        "title": "Quem pode fazer Botox?",
        "body": (
            "Contraindicado em gestantes, lactantes, pessoas com doenças neuromusculares "
            "e pessoas em uso de anticoagulantes — nesses casos é necessária avaliação presencial."
        ),
    },
]


def _fake_anthropic_response(
    answer: str,
    sources: List[str],
    requires_consultation: bool,
    sensitive_flag: bool,
) -> Any:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "knowledge_answer"
    block.input = {
        "answer": answer,
        "sources": sources,
        "requires_consultation": requires_consultation,
        "sensitive_flag": sensitive_flag,
    }
    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock(input_tokens=80, output_tokens=120)
    return response


# ---------------------------------------------------------------------------
# Unit: sensitive regex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    ("posso fazer botox grávida?", True),
    ("quanto tempo dura o botox?", False),
    ("tenho alergia a látex, posso fazer harmonização?", True),
    ("uso anticoagulante, posso fazer preenchimento?", True),
    ("o que é harmonização facial?", False),
    ("tenho diabetes, botox é seguro?", True),
])
def test_is_sensitive(question: str, expected: bool):
    assert _is_sensitive(question) == expected


# ---------------------------------------------------------------------------
# Smoke: "posso fazer botox grávida?" → sensitive escalation
# ---------------------------------------------------------------------------

def test_smoke_botox_gravida():
    """Issue spec: 'posso fazer botox grávida?' → resposta clara + escalação médica."""
    fake_response = _fake_anthropic_response(
        answer=(
            "Olá! O Botox é contraindicado durante a gravidez. "
            "A toxina botulínica não é recomendada para gestantes. "
            "Isso depende de uma avaliação presencial com a nossa equipe médica "
            "— quer agendar uma consulta?"
        ),
        sources=["Quem pode fazer Botox?"],
        requires_consultation=True,
        sensitive_flag=True,
    )

    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response

    with patch("app.agents.knowledge.agent._retrieve", return_value=BOTOX_CHUNKS):
        agent = KnowledgeSpecialist(client=mock_client)
        result = agent.forward(
            question="posso fazer botox grávida?",
            clinic_name="Clínica Vitória",
            tenant_id="57952a29-e228-4cac-b5fa-3d20ba478f5d",
        )

    assert result["messages"], "deve retornar mensagem"
    answer_text = result["messages"][0]["content"]
    assert "avaliação presencial" in answer_text.lower(), (
        f"resposta deve mencionar avaliação presencial; got: {answer_text!r}"
    )
    assert "[SCHEDULE_NEXT]" not in answer_text, (
        "routing hint não deve aparecer no texto da mensagem"
    )

    data = result["data"]
    assert data["sensitive_flag"] is True, "sensitive_flag deve ser True para gestante"
    assert data["requires_consultation"] is True, "requires_consultation deve ser True"
    assert data["routing_hint"] == "SCHEDULE_NEXT", (
        f"routing_hint deve ser SCHEDULE_NEXT para query sensível; got: {data.get('routing_hint')!r}"
    )
    assert result["conversation_stage"] == "knowledge"


# ---------------------------------------------------------------------------
# Non-sensitive: "quanto tempo dura o efeito do botox?"
# ---------------------------------------------------------------------------

def test_non_sensitive_botox_duration():
    fake_response = _fake_anthropic_response(
        answer="O efeito do Botox dura em média 4 a 6 meses, variando por área e metabolismo.",
        sources=["O que é Botox?"],
        requires_consultation=False,
        sensitive_flag=False,
    )

    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response

    with patch("app.agents.knowledge.agent._retrieve", return_value=BOTOX_CHUNKS):
        agent = KnowledgeSpecialist(client=mock_client)
        result = agent.forward(
            question="quanto tempo dura o efeito do botox?",
            clinic_name="Clínica Vitória",
            tenant_id="57952a29-e228-4cac-b5fa-3d20ba478f5d",
        )

    data = result["data"]
    assert data["sensitive_flag"] is False
    assert data["requires_consultation"] is False
    assert data["routing_hint"] is None, (
        f"routing_hint deve ser None para query não-sensível; got: {data.get('routing_hint')!r}"
    )
    assert "4 a 6 meses" in result["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Sensitive regex guard: LLM returns sensitive_flag=False but question triggers
# ---------------------------------------------------------------------------

def test_sensitive_guard_overrides_llm():
    """Even if LLM forgets to set sensitive_flag, regex guard enforces it."""
    fake_response = _fake_anthropic_response(
        answer="Botox tem contraindicações.",
        sources=["Quem pode fazer Botox?"],
        requires_consultation=False,
        sensitive_flag=False,  # LLM forgot to flag
    )

    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response

    with patch("app.agents.knowledge.agent._retrieve", return_value=BOTOX_CHUNKS):
        agent = KnowledgeSpecialist(client=mock_client)
        result = agent.forward(
            question="posso fazer botox grávida?",
            clinic_name="Clínica Vitória",
            tenant_id="57952a29-e228-4cac-b5fa-3d20ba478f5d",
        )

    data = result["data"]
    assert data["sensitive_flag"] is True, "guard deve forçar sensitive_flag=True"
    assert data["requires_consultation"] is True
    assert data["routing_hint"] == "SCHEDULE_NEXT", (
        f"guard deve forçar routing_hint=SCHEDULE_NEXT; got: {data.get('routing_hint')!r}"
    )
    assert "[SCHEDULE_NEXT]" not in result["messages"][0]["content"], (
        "routing hint não deve aparecer no texto da mensagem"
    )


# ---------------------------------------------------------------------------
# Fallback on LLM error
# ---------------------------------------------------------------------------

def test_fallback_on_llm_error():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("timeout")

    with patch("app.agents.knowledge.agent._retrieve", return_value=[]):
        agent = KnowledgeSpecialist(client=mock_client)
        result = agent.forward(
            question="o que é harmonização?",
            clinic_name="Clínica Vitória",
            tenant_id="57952a29-e228-4cac-b5fa-3d20ba478f5d",
        )

    assert result["messages"][0]["content"]
    assert result["data"]["sensitive_flag"] is False


# ---------------------------------------------------------------------------
# Pipeline registry: FAQ intent routes to KnowledgeSpecialist
# ---------------------------------------------------------------------------

def test_pipeline_registry_faq():
    from app.iris.pipeline import SPECIALIST_REGISTRY
    assert "FAQ" in SPECIALIST_REGISTRY, "FAQ deve estar no SPECIALIST_REGISTRY"
    agent_name, _ = SPECIALIST_REGISTRY["FAQ"]
    assert agent_name == "KnowledgeSpecialist"
