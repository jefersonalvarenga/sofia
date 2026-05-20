"""Tests for SCHEDULE_INTAKE wiring inside ``app.iris.pipeline._call_scheduler``.

Validates that when ``ScheduleRouter`` decides ``SCHEDULE_INTAKE``, the
pipeline:
  1. Resolves the service from ``schedule_session_data`` (creating an
     ``evaluation`` entry on cold start if missing).
  2. Resolves ``service_id`` via Supabase (or passes ``None`` when it can't).
  3. Loads intake questions via ``load_intake_questions``.
  4. Loads ``contraindications`` from the ``sf_clinic_services`` row (empty
     when the service couldn't be resolved).
  5. Calls ``ScheduleIntakeAgent.forward`` with the right kwargs.
  6. Returns an envelope where ``messages[0].content`` is the agent's text
     and ``data`` propagates ``schedule_sub_intent``,
     ``schedule_is_deviation`` and the updated ``schedule_session_data``.

Other ``next_intent`` values (CASHIER, EVALUATION, COMPLETION, FALLBACK) still
fall back to the deterministic UNKNOWN response — verified by a regression
test so we don't accidentally short-circuit them.

LM/Supabase fully mocked; this is a unit test of the wiring, not an E2E.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


CLINIC = "clinic-uuid-1"
SERVICE_ID = "service-uuid-botox"
SERVICE_NAME = "botox"


def _base_state(
    *,
    message: str = "quero agendar botox",
    schedule_session_data: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "trace_id": "trace-abc",
        "clinic_id": CLINIC,
        "session_id": "sess-1",
        "remote_jid": "5511999990001@s.whatsapp.net",
        "message": message,
        "history": [{"role": "human", "content": message}],
        "conversation_stage": "new",
        "clinic_name": "Clínica Bloom",
        "assistant_name": "Iris",
        "language": "pt-BR",
    }
    if schedule_session_data is not None:
        state["schedule_session_data"] = schedule_session_data
    return state


def _mk_questions(n: int = 5) -> List[Dict[str, Any]]:
    return [
        {
            "id": f"q-{i}",
            "order": i,
            "question_text": f"Pergunta {i}?",
            "category": "medicamentos",
            "is_required": True,
            "source": "clinic",
        }
        for i in range(1, n + 1)
    ]


def _agent_envelope(
    *,
    text: str = "Você toma algum medicamento regularmente?",
    session_data: List[Dict[str, Any]] | None = None,
    sub_intent_complete: bool = False,
    next_hint: str | None = None,
) -> Dict[str, Any]:
    """Return the shape an actual ``ScheduleIntakeAgent.forward`` call returns."""
    return {
        "messages": [{"type": "text", "content": text}],
        "conversation_stage": "schedule_intake",
        "reasoning": "test stub",
        "data": {
            "session_data": session_data
            or [
                {
                    "name": "evaluation",
                    "data": {"service": SERVICE_NAME, "intake_answers": []},
                }
            ],
            "sub_intent_complete": sub_intent_complete,
            "next_hint": next_hint,
            "intake_answers": [],
            "next_question_id": "q-1",
            "escalation_reason": None,
        },
    }


def _mk_supabase_mock(
    *,
    service_row: Dict[str, Any] | None = {
        "id": SERVICE_ID,
        "contraindications": ["gravidez", "anticoagulante"],
    },
):
    """Return a MagicMock that emulates the supabase-py builder chain.

    ``sb.table("sf_clinic_services").select(...).eq(...).ilike(...).order(...).limit(1).execute().data``
    returns ``[service_row]`` (or ``[]`` when ``service_row is None``).
    """
    rows = [service_row] if service_row is not None else []

    builder = MagicMock()
    builder.select.return_value = builder
    builder.eq.return_value = builder
    builder.ilike.return_value = builder
    builder.order.return_value = builder
    builder.limit.return_value = builder
    builder.execute.return_value = MagicMock(data=rows)

    sb = MagicMock()
    sb.table.return_value = builder
    return sb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScheduleIntakeWiring:
    def test_invokes_intake_agent_with_resolved_kwargs(self):
        """Router → SCHEDULE_INTAKE: agent is called with the right kwargs."""
        from app.iris import pipeline

        state = _base_state(
            schedule_session_data=[
                {
                    "name": "evaluation",
                    "data": {"service": SERVICE_NAME, "intake_answers": []},
                }
            ],
        )

        questions = _mk_questions(5)
        envelope = _agent_envelope(text="Pergunta 1?")

        with (
            patch.object(
                pipeline._schedule_router,
                "forward",
                return_value={
                    "next_intent": "SCHEDULE_INTAKE",
                    "is_deviation": False,
                    "session_data": state["schedule_session_data"],
                    "confidence": 0.95,
                    "reasoning": "router test",
                },
            ),
            patch.object(
                pipeline._schedule_intake_agent,
                "forward",
                return_value=envelope,
            ) as intake_spy,
            patch.object(
                pipeline,
                "load_intake_questions",
                return_value=questions,
            ) as load_questions_spy,
            patch.object(
                pipeline,
                "get_supabase",
                return_value=_mk_supabase_mock(),
            ),
        ):
            result = pipeline._call_scheduler(state, scope_text=state["message"])

        # Intake agent was called once
        assert intake_spy.call_count == 1

        # Questions were loaded with the resolved service_id
        load_questions_spy.assert_called_once_with(CLINIC, SERVICE_ID)

        kwargs = intake_spy.call_args.kwargs
        assert kwargs["latest_message"] == state["message"]
        assert kwargs["history"] == state["history"]
        assert kwargs["clinic_id"] == CLINIC
        assert kwargs["service"] == SERVICE_NAME
        assert kwargs["questions"] == questions
        assert kwargs["contraindications"] == ["gravidez", "anticoagulante"]
        assert kwargs["clinic_name"] == "Clínica Bloom"
        assert kwargs["assistant_name"] == "Iris"
        # session_data passed in must contain the evaluation entry
        passed_sd = kwargs["session_data"]
        assert any(e.get("name") == "evaluation" for e in passed_sd)

        # The envelope returned by _call_scheduler surfaces the agent's text
        msg = result["messages"][0]
        assert msg["type"] == "text"
        assert msg["content"] == "Pergunta 1?"

        # And carries the sub-router decision in data for the dispatcher
        data = result["data"]
        assert data["schedule_sub_intent"] == "SCHEDULE_INTAKE"
        assert data["schedule_is_deviation"] is False
        # Session_data must be the post-intake list (propagated forward)
        assert data["schedule_session_data"] == envelope["data"]["session_data"]

    def test_cold_start_creates_evaluation_entry_with_null_service(self):
        """When session_data has no evaluation entry, pipeline creates one
        with ``service=None`` so the agent doesn't blow up — the agent
        decides what to do next.
        """
        from app.iris import pipeline

        state = _base_state(schedule_session_data=[])

        questions: List[Dict[str, Any]] = []  # no service → no questions
        envelope = _agent_envelope(
            text="Pronto! Não preciso de mais nada nesta etapa, podemos seguir.",
            session_data=[
                {
                    "name": "evaluation",
                    "data": {"service": None, "intake_answers": []},
                }
            ],
            sub_intent_complete=True,
        )

        with (
            patch.object(
                pipeline._schedule_router,
                "forward",
                return_value={
                    "next_intent": "SCHEDULE_INTAKE",
                    "is_deviation": False,
                    "session_data": [],
                    "confidence": 0.9,
                    "reasoning": "cold start",
                },
            ),
            patch.object(
                pipeline._schedule_intake_agent,
                "forward",
                return_value=envelope,
            ) as intake_spy,
            patch.object(
                pipeline,
                "load_intake_questions",
                return_value=questions,
            ) as load_questions_spy,
            patch.object(
                pipeline,
                "get_supabase",
                return_value=_mk_supabase_mock(service_row=None),
            ),
        ):
            result = pipeline._call_scheduler(state, scope_text=state["message"])

        # No service => service_id=None passed to load_intake_questions
        load_questions_spy.assert_called_once_with(CLINIC, None)

        kwargs = intake_spy.call_args.kwargs
        assert kwargs["service"] in (None, "")
        # Pipeline must have synthesized the evaluation entry
        sd = kwargs["session_data"]
        eval_entry = next(e for e in sd if e["name"] == "evaluation")
        assert "service" in eval_entry["data"]
        assert kwargs["contraindications"] == []

        # Output still well-formed
        assert result["data"]["schedule_sub_intent"] == "SCHEDULE_INTAKE"
        assert result["messages"][0]["content"].startswith("Pronto!")

    def test_router_chooses_non_intake_returns_unknown_fallback(self):
        """Regression: CASHIER/EVALUATION/etc still fall back to UNKNOWN.

        Only INTAKE is wired in this PR; the rest stays deterministic.
        """
        from app.iris import pipeline

        state = _base_state(
            schedule_session_data=[
                {
                    "name": "evaluation",
                    "data": {"service": SERVICE_NAME, "intake_answers": []},
                }
            ],
        )

        with (
            patch.object(
                pipeline._schedule_router,
                "forward",
                return_value={
                    "next_intent": "SCHEDULE_CASHIER",
                    "is_deviation": False,
                    "session_data": state["schedule_session_data"],
                    "confidence": 0.9,
                    "reasoning": "post intake",
                },
            ),
            patch.object(
                pipeline._schedule_intake_agent,
                "forward",
            ) as intake_spy,
        ):
            result = pipeline._call_scheduler(state, scope_text=state["message"])

        assert intake_spy.call_count == 0
        # Deterministic UNKNOWN text — keeps prior behaviour for unwired flows
        assert result["messages"][0]["content"] == pipeline.UNKNOWN_FALLBACK_TEXT
        assert result["data"]["schedule_sub_intent"] == "SCHEDULE_CASHIER"

    def test_supabase_failure_loading_service_falls_back_to_no_metadata(self):
        """If Supabase query raises, the pipeline still calls the agent with
        ``service_id=None`` and ``contraindications=[]`` rather than crashing.
        """
        from app.iris import pipeline

        state = _base_state(
            schedule_session_data=[
                {
                    "name": "evaluation",
                    "data": {"service": SERVICE_NAME, "intake_answers": []},
                }
            ],
        )

        sb = MagicMock()
        sb.table.side_effect = RuntimeError("supabase down")

        envelope = _agent_envelope()

        with (
            patch.object(
                pipeline._schedule_router,
                "forward",
                return_value={
                    "next_intent": "SCHEDULE_INTAKE",
                    "is_deviation": False,
                    "session_data": state["schedule_session_data"],
                    "confidence": 0.9,
                    "reasoning": "router",
                },
            ),
            patch.object(
                pipeline._schedule_intake_agent,
                "forward",
                return_value=envelope,
            ) as intake_spy,
            patch.object(
                pipeline,
                "load_intake_questions",
                return_value=[],
            ) as load_questions_spy,
            patch.object(pipeline, "get_supabase", return_value=sb),
        ):
            result = pipeline._call_scheduler(state, scope_text=state["message"])

        # Service couldn't be resolved → service_id=None passed downstream
        load_questions_spy.assert_called_once_with(CLINIC, None)
        kwargs = intake_spy.call_args.kwargs
        assert kwargs["contraindications"] == []
        assert result["data"]["schedule_sub_intent"] == "SCHEDULE_INTAKE"
