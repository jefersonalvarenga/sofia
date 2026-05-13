"""
IntentType vocabulary for the Iris primary router.

Spec: kb/07-MVP/Tech/03-Discussoes/02 - Spec Router Primario.md (§1)

The router classifies the patient's latest message into one or more intents.
Each intent belongs to a semantic category that drives ordering and downstream
dispatch:

- INFORMATIONAL: answers the patient; does not end with a question expecting a reply.
- CTA: ends the turn with a question/action expecting a reply from the patient.
- TERMINAL: ends the automation. Next patient message bypasses Iris (goes to a human).

Ordering rule (used by ``_normalize_intents``): informational first (in order of
appearance), then CTAs (in order of appearance), then terminal last.
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet


class IntentType(str, Enum):
    # Informational
    BUSINESS_INFO = "BUSINESS_INFO"
    TOPIC_KNOWLEDGE = "TOPIC_KNOWLEDGE"
    REENGAGE = "REENGAGE"
    GREETING = "GREETING"
    UNCLASSIFIED = "UNCLASSIFIED"
    # CTAs
    INTAKE = "INTAKE"
    SCHEDULE = "SCHEDULE"
    # Terminal
    HUMAN_ESCALATION = "HUMAN_ESCALATION"


class IntentCategory(str, Enum):
    INFORMATIONAL = "informational"
    CTA = "cta"
    TERMINAL = "terminal"


INFORMATIONAL_INTENTS: FrozenSet[IntentType] = frozenset(
    {
        IntentType.BUSINESS_INFO,
        IntentType.TOPIC_KNOWLEDGE,
        IntentType.REENGAGE,
        IntentType.GREETING,
        IntentType.UNCLASSIFIED,
    }
)

CTA_INTENTS: FrozenSet[IntentType] = frozenset(
    {
        IntentType.INTAKE,
        IntentType.SCHEDULE,
    }
)

TERMINAL_INTENTS: FrozenSet[IntentType] = frozenset(
    {
        IntentType.HUMAN_ESCALATION,
    }
)


def category_of(intent: IntentType) -> IntentCategory:
    """Return the semantic category of an intent.

    Raises ``ValueError`` for values outside the enum — callers should not
    reach this function with unvalidated input.
    """
    if intent in INFORMATIONAL_INTENTS:
        return IntentCategory.INFORMATIONAL
    if intent in CTA_INTENTS:
        return IntentCategory.CTA
    if intent in TERMINAL_INTENTS:
        return IntentCategory.TERMINAL
    raise ValueError(f"Intent {intent!r} has no category mapping")


VALID_INTENT_VALUES: FrozenSet[str] = frozenset(item.value for item in IntentType)
