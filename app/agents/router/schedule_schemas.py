"""
Pydantic schemas for the Iris schedule sub-router structured output.

The sub-router classifies the patient's latest message into exactly ONE
``ScheduleIntent`` sub-type, given the session sequence handed off by the
upstream Manager agent.

Output shape (intentionally distinct from RouterOutput because:
  - the primary router emits multi-intent; this sub-router emits single
  - this sub-router carries `session_data` forward across hand-offs
  - `is_deviation` is a first-class signal so the pipeline can branch
):

    {
        "next_intent": "SCHEDULE_INTAKE",
        "is_deviation": False,
        "session_data": [{"name": "evaluation", "data": {...}}],
        "confidence": 0.92,
        "reasoning": "Paciente respondeu pergunta anterior, segue sequencia."
    }
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .schedule_intents import ScheduleIntent


class SessionDataEntry(BaseModel):
    """Cross-sub-agent payload carried by the sub-router.

    ``name`` identifies the sub-flow (e.g. ``"evaluation"``, ``"service"``),
    ``data`` is sub-flow-specific (service of interest, slot id, etc).
    """

    name: str = Field(..., min_length=1)
    data: Dict[str, Any] = Field(default_factory=dict)


class ScheduleRouterOutput(BaseModel):
    """Structured output of the schedule sub-router LLM call."""

    next_intent: ScheduleIntent = Field(
        ...,
        description=(
            "Sub-tipo a despachar agora. Pode ser o proximo da sequencia "
            "(happy-path) ou um intent de desvio (CANCEL/CHANGE/FALLBACK)."
        ),
    )
    is_deviation: bool = Field(
        ...,
        description=(
            "True quando a mensagem do paciente quebra a sequencia esperada. "
            "FALLBACK conta como desvio."
        ),
    )
    session_data: List[SessionDataEntry] = Field(
        default_factory=list,
        description=(
            "Payload de sessao a propagar adiante. O sub-router pode acrescentar "
            "entradas com base no que percebeu na mensagem; nunca remove."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Confianca da classificacao [0,1]. Abaixo do threshold do agente, "
            "o caller deve forcar SCHEDULE_FALLBACK."
        ),
    )
    reasoning: str = Field(
        default="",
        max_length=400,
        description="Justificativa curta da decisao (<= 400 chars).",
    )
