"""
Pure normalization for router-emitted intents.

Spec: kb/07-MVP/Tech/03-Discussoes/02 - Spec Router Primario.md (§1.5, §2)

``_normalize_intents`` takes the raw intents emitted by the LLM (either parsed
``IntentClassification`` models or plain dicts from a tool-call payload) and
applies the deterministic post-processing rules:

  1. Drop entries with intent values outside the enum (silent).
  2. Dedup by intent value — first occurrence wins.
  3. Empty ``scope_text`` (after strip) falls back to the full latest message.
  4. Sort stably: informational first (in input order), then CTAs (in input
     order), then HUMAN_ESCALATION last. Within each bucket, the LLM's order
     is preserved — it reflects the order the intents appeared in the message.
  5. If the resulting list is empty, return a single
     ``{"intent": "UNCLASSIFIED", "scope_text": <latest_message>}``.

This module is intentionally LLM-agnostic and side-effect-free so it can be
unit-tested without any external dependency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

from .intents import (
    CTA_INTENTS,
    INFORMATIONAL_INTENTS,
    IntentType,
    TERMINAL_INTENTS,
    VALID_INTENT_VALUES,
)
from .schemas import IntentClassification


RawIntent = Union[IntentClassification, Dict[str, Any]]


_CATEGORY_RANK: Dict[IntentType, int] = {
    **{intent: 0 for intent in INFORMATIONAL_INTENTS},
    **{intent: 1 for intent in CTA_INTENTS},
    **{intent: 2 for intent in TERMINAL_INTENTS},
}


def _coerce(item: RawIntent) -> Union[Dict[str, str], None]:
    """Extract ``{intent, scope_text}`` from an item, or None if unusable."""
    if isinstance(item, IntentClassification):
        return {"intent": item.intent.value, "scope_text": item.scope_text}

    if not isinstance(item, dict):
        return None

    raw_intent = item.get("intent")
    if isinstance(raw_intent, IntentType):
        intent_value = raw_intent.value
    else:
        intent_value = str(raw_intent or "").strip().upper()

    if intent_value not in VALID_INTENT_VALUES:
        return None

    scope_text = str(item.get("scope_text") or "")
    return {"intent": intent_value, "scope_text": scope_text}


def normalize_intents(
    raw_intents: List[RawIntent],
    latest_message: str,
) -> List[Dict[str, str]]:
    """Apply the post-parse normalization rules from the spec.

    Returns a list of ``{"intent": str, "scope_text": str}`` dicts, always
    non-empty (falls back to a single UNCLASSIFIED entry when no valid intent
    survives).
    """
    seen: set[str] = set()
    parsed: List[Dict[str, str]] = []

    for item in raw_intents or []:
        coerced = _coerce(item)
        if coerced is None:
            continue
        intent_value = coerced["intent"]
        if intent_value in seen:
            continue

        scope_text = coerced["scope_text"].strip()
        if not scope_text:
            scope_text = latest_message

        seen.add(intent_value)
        parsed.append({"intent": intent_value, "scope_text": scope_text})

    if not parsed:
        return [
            {
                "intent": IntentType.UNCLASSIFIED.value,
                "scope_text": latest_message,
            }
        ]

    parsed.sort(key=lambda x: _CATEGORY_RANK[IntentType(x["intent"])])
    return parsed
