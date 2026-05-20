"""Pydantic schemas for the SCHEDULE_INTAKE sub-agent.

Spec: kb/07-MVP/Tech/03-Discussoes/schedule/01 - Spec SCHEDULE_INTAKE.md (§3).

Three layers:
    - ``IntakeAnswer``: one collected answer (with optional contraindication match).
    - ``IntakeData``: ``data`` payload of the output envelope.
    - ``IntakeOutput``: full output envelope (messages + conversation_stage +
      reasoning + data) following the Greeting / Knowledge / ScheduleRouter
      convention.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IntakeAnswer(BaseModel):
    """A single collected intake answer.

    ``question_text`` is snapshotted at answer time so editing the question
    later does not retroactively rewrite the audit trail.
    """

    question_id: str = Field(..., min_length=1)
    question_text: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    matched_contraindication: Optional[str] = Field(
        default=None,
        description=(
            "Term from contraindications[] that the LLM identified semantically "
            "in the answer. None when no contraindication matched."
        ),
    )


class IntakeData(BaseModel):
    """``data`` payload returned by the intake sub-agent."""

    session_data: List[Dict[str, Any]] = Field(
        ...,
        description="Cross-sub-agent state. Always present, append-only.",
    )
    sub_intent_complete: bool = Field(
        ...,
        description=(
            "True when all is_required=true questions have been answered and "
            "no contraindication was detected."
        ),
    )
    next_hint: Optional[str] = Field(
        default=None,
        description='"ESCALATE_TO_HUMAN" when a contraindication was detected.',
    )
    intake_answers: List[IntakeAnswer] = Field(
        default_factory=list,
        description="All answers collected so far (append-only).",
    )
    next_question_id: Optional[str] = Field(
        default=None,
        description="ID of the question that will be asked next. None on completion or escalation.",
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Human-readable reason when next_hint=ESCALATE_TO_HUMAN.",
    )


class IntakeOutput(BaseModel):
    """Full output envelope for the SCHEDULE_INTAKE sub-agent."""

    messages: List[Dict[str, str]] = Field(
        ...,
        description='List of {"type":"text", "content":"..."} payloads.',
    )
    conversation_stage: str = Field(default="schedule_intake")
    reasoning: str = Field(..., max_length=400)
    data: Dict[str, Any] = Field(...)
