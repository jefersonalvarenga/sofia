"""
ScheduleIntent vocabulary for the Iris schedule sub-router.

Sub-classifies a SCHEDULE intent (from the primary RouterAgent) into one of
10 sub-types covering the scheduling life cycle of an aesthetic clinic.

The sub-router is invoked by the pipeline whenever the primary router emits
``SCHEDULE``. It receives the latest message + a session-scoped sequence (from
an upstream Manager agent) and decides which sub-agent should act next.

Two categories drive deviation detection:

- SEQUENCE: sub-types that are part of a normal (happy-path) flow.
  Examples: INTAKE -> CASHIER -> EVALUATION -> COMPLETION.
- DEVIATION: sub-types that BREAK the expected sequence (patient cancels,
  changes, asks for human, or sends an ambiguous message that the router
  cannot place).

The sub-router decides for itself whether the patient is following the
sequence or deviating, so neither category is hard-coded into the prompt as a
filter. They are used by ``ScheduleRouterOutput`` consumers (and the eval
heuristic) to discriminate ``is_deviation``.
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet


class ScheduleIntent(str, Enum):
    # Sequence sub-types (happy-path; part of a flow handed off by Manager)
    SCHEDULE_INTAKE = "SCHEDULE_INTAKE"
    SCHEDULE_CASHIER = "SCHEDULE_CASHIER"
    SCHEDULE_EVALUATION = "SCHEDULE_EVALUATION"
    SCHEDULE_SERVICE = "SCHEDULE_SERVICE"
    SCHEDULE_SERVICE_PROTOCOL = "SCHEDULE_SERVICE_PROTOCOL"
    SCHEDULE_CONFIRMATION = "SCHEDULE_CONFIRMATION"
    SCHEDULE_REMINDER = "SCHEDULE_REMINDER"
    SCHEDULE_COMPLETION = "SCHEDULE_COMPLETION"
    # Deviation sub-types (patient breaks the expected sequence)
    SCHEDULE_CANCEL = "SCHEDULE_CANCEL"
    SCHEDULE_CHANGE = "SCHEDULE_CHANGE"
    # Fallback (router cannot identify; pipeline asks a recalibration question)
    SCHEDULE_FALLBACK = "SCHEDULE_FALLBACK"


SEQUENCE_INTENTS: FrozenSet[ScheduleIntent] = frozenset(
    {
        ScheduleIntent.SCHEDULE_INTAKE,
        ScheduleIntent.SCHEDULE_CASHIER,
        ScheduleIntent.SCHEDULE_EVALUATION,
        ScheduleIntent.SCHEDULE_SERVICE,
        ScheduleIntent.SCHEDULE_SERVICE_PROTOCOL,
        ScheduleIntent.SCHEDULE_CONFIRMATION,
        ScheduleIntent.SCHEDULE_REMINDER,
        ScheduleIntent.SCHEDULE_COMPLETION,
    }
)

DEVIATION_INTENTS: FrozenSet[ScheduleIntent] = frozenset(
    {
        ScheduleIntent.SCHEDULE_CANCEL,
        ScheduleIntent.SCHEDULE_CHANGE,
    }
)

FALLBACK_INTENT: ScheduleIntent = ScheduleIntent.SCHEDULE_FALLBACK


VALID_SCHEDULE_INTENT_VALUES: FrozenSet[str] = frozenset(
    item.value for item in ScheduleIntent
)


def is_deviation(intent: ScheduleIntent) -> bool:
    """Return True when the intent breaks the expected sequence.

    FALLBACK counts as deviation because the sequence cannot proceed.
    """
    if intent in DEVIATION_INTENTS:
        return True
    if intent == FALLBACK_INTENT:
        return True
    return False
