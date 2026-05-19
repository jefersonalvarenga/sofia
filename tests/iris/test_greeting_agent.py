"""
Unit tests for GreetingAgent v14.

Spec: kb/07-MVP/Tech/03-Discussoes/03 - Greeting Agent Spec v0.14.md

LM mocked end-to-end. v14 contract uses patient_intents, session_summary,
recent_relevant_messages, time_gap_hours; clinic_name and assistant_name
remain mandatory.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import dspy
import pytest

from app.agents.greeting.agent import (
    GREETING_MAX_TOKENS,
    GREETING_MODEL,
    GREETING_TEMPERATURE,
    GreetingAgent,
    TECHNICAL_FALLBACK,
    _clean_llm_output,
    _coerce_few_shot,
    _normalize_contact_name,
)


import json as _json


def _make_lm(return_value=None, side_effect=None):
    lm = MagicMock()
    if side_effect is not None:
        lm.side_effect = side_effect
    else:
        lm.return_value = return_value if return_value is not None else [""]
    return lm


def _lm_json(response: str, reasoning: str = "ok"):
    """Helper: build an LM mock returning a v14 JSON payload."""
    payload = _json.dumps({"reasoning": reasoning, "response": response})
    return _make_lm(return_value=[payload])


@pytest.fixture
def agent_factory():
    def _factory(lm: MagicMock) -> GreetingAgent:
        return GreetingAgent(lm=lm)

    return _factory


class TestConstants:
    def test_model(self):
        # v24.1: rolled back from gpt-5-nano to deepseek-v4-flash.
        assert GREETING_MODEL == "deepseek/deepseek-v4-flash"

    def test_temperature(self):
        # v26: 0.3 -> 0.0 (deterministic single-shot reproduction).
        assert GREETING_TEMPERATURE == 0.0

    def test_max_tokens(self):
        # v17: raised 96 -> 192 because DeepSeek pt-BR reasoning is longer.
        assert GREETING_MAX_TOKENS == 192

    def test_technical_fallback(self):
        assert TECHNICAL_FALLBACK == "Olá! Tudo bem?"


class TestHelpers:
    def test_clean_strips_whitespace(self):
        assert _clean_llm_output("  Olá!  ") == "Olá!"

    def test_clean_strips_quotes(self):
        assert _clean_llm_output('"Olá!"') == "Olá!"

    def test_clean_strips_brackets(self):
        assert _clean_llm_output("[Olá!]") == "Olá!"

    def test_clean_empty(self):
        assert _clean_llm_output("") == ""

    def test_normalize_paciente_sentinel(self):
        assert _normalize_contact_name("Paciente") is None

    def test_normalize_empty(self):
        assert _normalize_contact_name("") is None
        assert _normalize_contact_name(None) is None

    def test_normalize_real_name(self):
        assert _normalize_contact_name("Camila") == "Camila"

    def test_coerce_few_shot_singular_wins(self):
        # v26: signature is _coerce_few_shot(few_shot, few_shots, initial_greetings, greeting_example).
        assert _coerce_few_shot("a", ["b", "c"], ["x"], "legacy") == "a"

    def test_coerce_few_shots_list_fallback(self):
        # Legacy callers passing a list: first non-empty element wins.
        assert _coerce_few_shot(None, ["a", "b"], ["x"], "legacy") == "a"

    def test_coerce_initial_greetings_fallback(self):
        assert _coerce_few_shot(None, None, ["x", "y"], "legacy") == "x"

    def test_coerce_greeting_example_fallback(self):
        assert _coerce_few_shot(None, None, None, "legacy") == "legacy"

    def test_coerce_empty(self):
        assert _coerce_few_shot(None, None, None, None) == ""
        assert _coerce_few_shot("", [], [], "") == ""

    def test_coerce_filters_blanks(self):
        # Blank entries are skipped; first non-blank wins.
        assert _coerce_few_shot(None, ["", "   ", "real"], None, None) == "real"


class TestForwardAlwaysResponds:
    def test_first_contact(self, agent_factory):
        agent = agent_factory(_lm_json("Olá Camila! Aqui é da Lumina."))
        out = agent.forward(
            patient_message="oi",
            patient_intents=[],
            patient_name="Camila",
            clinic_name="Lumina",
            assistant_name="Iris",
            few_shots=["Olá! Aqui é da Lumina. Como posso te ajudar?"],
            recent_relevant_messages=[],
            time_gap_hours=None,
        )
        assert out["messages"][0]["content"] == "Olá Camila! Aqui é da Lumina."
        assert out["conversation_stage"] == "greeting"
        assert "source=llm" in out["reasoning"]

    def test_with_intencao(self, agent_factory):
        agent = agent_factory(_lm_json("Olá Pedro. Aqui é da Lumina."))
        out = agent.forward(
            patient_message="oi, queria saber preço",
            patient_intents=["BUSINESS_INFO"],
            patient_name="Pedro",
            clinic_name="Lumina",
            assistant_name="Iris",
            few_shots=["Olá! Aqui é da Lumina. Como posso te ajudar?"],
            time_gap_hours=None,
        )
        assert out["messages"][0]["content"] == "Olá Pedro. Aqui é da Lumina."

    def test_llm_reasoning_propagated_to_data(self, agent_factory):
        agent = agent_factory(_lm_json("Olá!", reasoning="só cumprimento, mantive CTA"))
        out = agent.forward(
            patient_message="oi", patient_name="X", clinic_name="X", assistant_name="X"
        )
        # data dict carries the model's reasoning + confidence (None when omitted)
        assert out["data"]["llm_reasoning"] == "só cumprimento, mantive CTA"
        assert "llm_reasoning=só cumprimento, mantive CTA" in out["reasoning"]

    def test_silence_when_response_is_empty_string(self, agent_factory):
        # v15: model intentionally emits response="" → agent returns messages=[]
        agent = agent_factory(_lm_json("", reasoning="paciente fechou ritual"))
        out = agent.forward(
            patient_message="tudo bem",
            patient_name="Camila",
            clinic_name="X",
            assistant_name="X",
        )
        assert out["messages"] == []
        assert "source=llm_silence" in out["reasoning"]
        assert out["data"]["silence"] is True

    def test_confidence_ignored_when_present(self, agent_factory):
        # v17: confidence was dropped from the schema. If the model still
        # returns it, we ignore it silently and do not expose it.
        import json
        payload = json.dumps(
            {"reasoning": "ok", "response": "Olá!", "confidence": 0.87}
        )
        agent = agent_factory(_make_lm(return_value=[payload]))
        out = agent.forward(
            patient_message="oi", patient_name="X", clinic_name="X", assistant_name="X"
        )
        assert "confidence" not in (out.get("data") or {})
        assert "confidence" not in out["reasoning"]


class TestForwardPostprocessing:
    def test_strips_quotes(self, agent_factory):
        agent = agent_factory(_lm_json('"Olá!"'))
        out = agent.forward(
            patient_message="oi",
            patient_name="X",
            clinic_name="X",
            assistant_name="X",
        )
        assert out["messages"][0]["content"] == "Olá!"

    def test_strips_whitespace(self, agent_factory):
        agent = agent_factory(_lm_json("  Olá Camila  \n"))
        out = agent.forward(
            patient_message="oi",
            patient_name="Camila",
            clinic_name="X",
            assistant_name="X",
        )
        assert out["messages"][0]["content"] == "Olá Camila"


class TestForwardTechnicalFallback:
    def test_missing_response_field_falls_back(self, agent_factory):
        # v15: JSON without "response" key (or null/non-string) → fallback
        import json
        payload = json.dumps({"reasoning": "ok", "confidence": 0.5})
        agent = agent_factory(_make_lm(return_value=[payload]))
        out = agent.forward(
            patient_message="oi", patient_name="X", clinic_name="X", assistant_name="X"
        )
        assert out["messages"][0]["content"] == TECHNICAL_FALLBACK
        assert "source=fallback" in out["reasoning"]

    def test_invalid_json_recovers_raw_text(self, agent_factory):
        # v14: LM returned non-JSON → agent tries to recover raw text
        agent = agent_factory(_make_lm(return_value=["Olá Camila"]))
        out = agent.forward(
            patient_message="oi", patient_name="Camila", clinic_name="X", assistant_name="X"
        )
        # Recovery path keeps the raw text as the response
        assert out["messages"][0]["content"] == "Olá Camila"
        assert "source=llm_no_json" in out["reasoning"]

    def test_exception_falls_back(self, agent_factory):
        agent = agent_factory(_make_lm(side_effect=RuntimeError("OpenAI 503")))
        out = agent.forward(
            patient_message="oi", patient_name="X", clinic_name="X", assistant_name="X"
        )
        assert out["messages"][0]["content"] == TECHNICAL_FALLBACK
        assert "source=fallback" in out["reasoning"]

    def test_no_lm_configured_falls_back(self):
        old_lm = dspy.settings.lm
        try:
            dspy.settings.configure(lm=None)
            agent = GreetingAgent()
            out = agent.forward(
                patient_message="oi", patient_name="X", clinic_name="X", assistant_name="X"
            )
            assert out["messages"][0]["content"] == TECHNICAL_FALLBACK
            assert "source=fallback" in out["reasoning"]
        finally:
            dspy.settings.configure(lm=old_lm)

    def test_empty_outputs_list_falls_back(self, agent_factory):
        agent = agent_factory(_make_lm(return_value=[]))
        out = agent.forward(
            patient_message="oi", patient_name="X", clinic_name="X", assistant_name="X"
        )
        assert out["messages"][0]["content"] == TECHNICAL_FALLBACK


class TestForwardLMInvocation:
    def test_temperature_and_max_tokens(self, agent_factory):
        mock_lm = _lm_json("Olá!")
        agent = agent_factory(mock_lm)
        agent.forward(
            patient_message="oi", patient_name="X", clinic_name="X", assistant_name="X"
        )
        kwargs = mock_lm.call_args.kwargs
        assert kwargs["temperature"] == GREETING_TEMPERATURE
        assert kwargs["max_tokens"] == GREETING_MAX_TOKENS

    def test_messages_system_plus_user(self, agent_factory):
        mock_lm = _lm_json("Olá!")
        agent = agent_factory(mock_lm)
        agent.forward(
            patient_message="oi, quero saber preço",
            patient_intents=["BUSINESS_INFO"],
            patient_name="Camila",
            clinic_name="Lumina",
            assistant_name="Iris",
            few_shots=["Oi! Aqui é da Lumina."],
            recent_relevant_messages=[],
            time_gap_hours=None,
        )
        msgs = mock_lm.call_args.kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        system = msgs[0]["content"]
        # v20: prompt restaurado pro tamanho completo após v19 colapsar.
        # Marcadores estáveis: ROLE, OBJETIVO, ENTRADAS, PADRÃO, etc.
        assert "agente de saudação" in system
        assert "OBJETIVO" in system
        assert "patient_intents" in system
        user = msgs[1]["content"]
        assert "Camila" in user
        assert "Lumina" in user
        assert "Oi! Aqui é da Lumina." in user
        assert "oi, quero saber preço" in user
        assert "BUSINESS_INFO" in user
        assert "time_gap_hours" in user

    def test_first_contact_renders_null_gap(self, agent_factory):
        mock_lm = _lm_json("Olá!")
        agent = agent_factory(mock_lm)
        agent.forward(
            patient_message="oi",
            patient_name="Camila",
            clinic_name="Lumina",
            assistant_name="Iris",
            time_gap_hours=None,
        )
        user = mock_lm.call_args.kwargs["messages"][1]["content"]
        assert "primeiro contato" in user

    def test_returning_user_renders_gap(self, agent_factory):
        mock_lm = _lm_json("Olá!")
        agent = agent_factory(mock_lm)
        agent.forward(
            patient_message="oi",
            patient_name="Camila",
            clinic_name="Lumina",
            assistant_name="Iris",
            time_gap_hours=48,
        )
        user = mock_lm.call_args.kwargs["messages"][1]["content"]
        assert "time_gap_hours: 48" in user

    def test_paciente_sentinel_becomes_null_in_prompt(self, agent_factory):
        mock_lm = _lm_json("Olá!")
        agent = agent_factory(mock_lm)
        agent.forward(
            patient_message="oi",
            patient_name="Paciente",
            clinic_name="X",
            assistant_name="X",
        )
        user = mock_lm.call_args.kwargs["messages"][1]["content"]
        assert "patient_name: null" in user
        line = [l for l in user.split("\n") if l.startswith("- patient_name")][0]
        assert "Paciente" not in line

    def test_few_shot_block_when_empty(self, agent_factory):
        # v26: few_shot is singular; user prompt shows "(não fornecido)" when missing.
        mock_lm = _lm_json("Olá!")
        agent = agent_factory(mock_lm)
        agent.forward(
            patient_message="oi",
            patient_name="Camila",
            clinic_name="X",
            assistant_name="X",
            few_shot="",
        )
        user = mock_lm.call_args.kwargs["messages"][1]["content"]
        assert "(não fornecido)" in user

    def test_recent_messages_rendered(self, agent_factory):
        mock_lm = _lm_json("Olá!")
        agent = agent_factory(mock_lm)
        agent.forward(
            patient_message="oi",
            patient_name="Camila",
            clinic_name="Lumina",
            assistant_name="Iris",
            recent_relevant_messages=[
                {"role": "patient", "content": "boa tarde"},
                {"role": "greeting", "content": "Olá Camila, aqui é da Lumina."},
            ],
        )
        user = mock_lm.call_args.kwargs["messages"][1]["content"]
        assert "boa tarde" in user
        assert "Olá Camila, aqui é da Lumina." in user


class TestForwardLegacyCompat:
    def test_legacy_call_shape(self, agent_factory):
        # pipeline.py passes (patient_name, clinic_name, assistant_name,
        # history_length, greeting_example) — agent must still accept it.
        mock_lm = _lm_json("Olá Camila")
        agent = agent_factory(mock_lm)
        out = agent.forward(
            patient_name="Camila",
            clinic_name="Lumina",
            assistant_name="Iris",
            history_length=0,
            greeting_example="Oi! Aqui é da Lumina",
            scope_text="oi",
        )
        assert out["messages"][0]["content"] == "Olá Camila"

    def test_legacy_greeting_example_becomes_few_shot(self, agent_factory):
        mock_lm = _lm_json("Olá!")
        agent = agent_factory(mock_lm)
        agent.forward(
            patient_name="Lucas",
            clinic_name="Lumina",
            assistant_name="Iris",
            greeting_example="E aí! Aqui é da Lumina",
            scope_text="oi",
        )
        user = mock_lm.call_args.kwargs["messages"][1]["content"]
        assert "E aí! Aqui é da Lumina" in user

    def test_legacy_history_becomes_recent_messages(self, agent_factory):
        mock_lm = _lm_json("Olá!")
        agent = agent_factory(mock_lm)
        agent.forward(
            patient_name="Pedro",
            clinic_name="X",
            assistant_name="X",
            scope_text="oi",
            history=[
                {"role": "patient", "content": "boa noite"},
            ],
        )
        user = mock_lm.call_args.kwargs["messages"][1]["content"]
        assert "boa noite" in user
