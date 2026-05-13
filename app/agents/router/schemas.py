"""
Pydantic schemas for the Iris primary router structured output.

Spec: kb/07-MVP/Tech/03-Discussoes/02 - Spec Router Primario.md (§6)

`RouterOutput` is what the LLM tool-call returns. After parsing,
``RouterAgent.forward()`` normalizes the intents (dedup, ordering, fallback)
and projects them into a dict shape suitable for the pipeline state.

The ``language`` field from the legacy ``IrisRouterOutput`` is intentionally
absent — MVP is pt-BR only (§4).
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from .intents import IntentType


class IntentClassification(BaseModel):
    """A single intent detected in the patient message, with its originating scope.

    ``scope_text`` is the substring (or close paraphrase) of the latest message
    that triggered this intent. Downstream specialists receive ``scope_text``
    rather than the full message so each specialist answers only its own slice
    of a multi-intent message.
    """

    intent: IntentType = Field(
        ...,
        description=(
            "Intent detected. One of: BUSINESS_INFO, TOPIC_KNOWLEDGE, REENGAGE, "
            "GREETING, UNCLASSIFIED, INTAKE, SCHEDULE, HUMAN_ESCALATION."
        ),
    )
    scope_text: str = Field(
        ...,
        min_length=1,
        description=(
            "Trecho da mensagem do paciente que originou essa intent. Prefira "
            "uma substring literal. Quando ha so uma intent, pode ser a "
            "mensagem inteira."
        ),
    )


class RouterOutput(BaseModel):
    """Structured output schema returned by the router LLM tool call."""

    intents: List[IntentClassification] = Field(
        ...,
        min_length=1,
        description=(
            "Lista de intents detectadas (1..N). O router emitira uma ou mais "
            "entradas {intent, scope_text}. A ordenacao final por categoria "
            "semantica (informacional -> CTA -> terminal) e aplicada pelo "
            "_normalize_intents apos o parse."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Confianca agregada do roteamento, entre 0.0 e 1.0. Valores abaixo "
            "do threshold (default 0.70) fazem o agente emitir UNCLASSIFIED."
        ),
    )
    reasoning: str = Field(
        ...,
        max_length=400,
        description="Explicacao curta da decisao de roteamento (<= 400 chars).",
    )
