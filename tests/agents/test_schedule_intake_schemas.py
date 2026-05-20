"""Schema-level tests for SCHEDULE_INTAKE Pydantic models.

Spec: kb/07-MVP/Tech/03-Discussoes/schedule/01 - Spec SCHEDULE_INTAKE.md (§3).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestIntakeAnswerSchema:
    def test_intake_answer_requires_question_id(self):
        from app.agents.schedule_intake.schemas import IntakeAnswer

        with pytest.raises(ValidationError):
            IntakeAnswer(
                question_text="Você toma remédio?",
                category="medicamentos",
                answer="não",
            )

    def test_intake_answer_requires_question_text(self):
        from app.agents.schedule_intake.schemas import IntakeAnswer

        with pytest.raises(ValidationError):
            IntakeAnswer(
                question_id="q1",
                category="medicamentos",
                answer="não",
            )

    def test_intake_answer_requires_category(self):
        from app.agents.schedule_intake.schemas import IntakeAnswer

        with pytest.raises(ValidationError):
            IntakeAnswer(
                question_id="q1",
                question_text="Você toma remédio?",
                answer="não",
            )

    def test_intake_answer_requires_answer(self):
        from app.agents.schedule_intake.schemas import IntakeAnswer

        with pytest.raises(ValidationError):
            IntakeAnswer(
                question_id="q1",
                question_text="Você toma remédio?",
                category="medicamentos",
            )

    def test_intake_answer_matched_contraindication_defaults_to_none(self):
        from app.agents.schedule_intake.schemas import IntakeAnswer

        a = IntakeAnswer(
            question_id="q1",
            question_text="Você toma remédio?",
            category="medicamentos",
            answer="não",
        )
        assert a.matched_contraindication is None

    def test_intake_answer_accepts_matched_contraindication(self):
        from app.agents.schedule_intake.schemas import IntakeAnswer

        a = IntakeAnswer(
            question_id="q3",
            question_text="Está grávida?",
            category="gestacao",
            answer="sim, 12 semanas",
            matched_contraindication="gravidez",
        )
        assert a.matched_contraindication == "gravidez"


class TestIntakeDataSchema:
    def test_intake_data_requires_session_data(self):
        from app.agents.schedule_intake.schemas import IntakeData

        with pytest.raises(ValidationError):
            IntakeData(sub_intent_complete=False, intake_answers=[])

    def test_intake_data_minimal_valid(self):
        from app.agents.schedule_intake.schemas import IntakeData

        d = IntakeData(
            session_data=[],
            sub_intent_complete=False,
            intake_answers=[],
        )
        assert d.next_hint is None
        assert d.next_question_id is None
        assert d.escalation_reason is None

    def test_intake_data_with_escalation(self):
        from app.agents.schedule_intake.schemas import IntakeAnswer, IntakeData

        ans = IntakeAnswer(
            question_id="q3",
            question_text="Está grávida?",
            category="gestacao",
            answer="sim",
            matched_contraindication="gravidez",
        )
        d = IntakeData(
            session_data=[],
            sub_intent_complete=False,
            next_hint="ESCALATE_TO_HUMAN",
            intake_answers=[ans],
            next_question_id=None,
            escalation_reason="Paciente grávida. Botox contraindicado.",
        )
        assert d.next_hint == "ESCALATE_TO_HUMAN"
        assert d.escalation_reason.startswith("Paciente")


class TestIntakeOutputSchema:
    def test_intake_output_minimal(self):
        from app.agents.schedule_intake.schemas import IntakeData, IntakeOutput

        out = IntakeOutput(
            messages=[{"type": "text", "content": "Olá!"}],
            reasoning="cold start",
            data={
                "session_data": [],
                "sub_intent_complete": False,
                "intake_answers": [],
            },
        )
        assert out.conversation_stage == "schedule_intake"

    def test_intake_output_reasoning_max_length(self):
        from app.agents.schedule_intake.schemas import IntakeOutput

        long_reason = "x" * 401
        with pytest.raises(ValidationError):
            IntakeOutput(
                messages=[{"type": "text", "content": "hi"}],
                reasoning=long_reason,
                data={
                    "session_data": [],
                    "sub_intent_complete": False,
                    "intake_answers": [],
                },
            )
