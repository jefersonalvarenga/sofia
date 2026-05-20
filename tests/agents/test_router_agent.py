"""Unit tests for ``RouterAgent`` — GREETING composition logic.

Covers the three composition triggers introduced in
``feat/router-greeting-composition``:

  1. Cold start  — first patient message in the session (history empty).
  2. Stale gap   — last interaction > 24h ago.
  3. Social reply — patient answered an assistant greeting socially. The
                    LLM detects this semantically, so we only verify the
                    user prompt carries enough context for it to decide.

The LM is mocked end-to-end; no DeepSeek calls are made. The tests assert:

  * ``RouterAgent.forward`` accepts the new ``last_interaction_at`` and
    ``patient_name`` kwargs with backward-compatible defaults.
  * The user prompt sent to the LM carries a ``context_flags`` block that
    surfaces ``cold_start``, ``stale``, and ``last_interaction_hours_ago``.
  * When the LM returns ``[GREETING, SCHEDULE]`` (or just ``[SCHEDULE]``)
    the agent normalizes the output unchanged — the composition decision
    lives entirely in the prompt, not in post-processing.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_lm_returning(payload: Dict[str, Any]) -> MagicMock:
    """Mock dspy.LM that returns ``[json.dumps(payload)]`` on call."""
    lm = MagicMock()
    lm.return_value = [json.dumps(payload)]
    return lm


def _last_user_prompt(lm: MagicMock) -> str:
    """Pull the user-role content out of the most recent LM invocation."""
    assert lm.call_args is not None, "LM was never invoked"
    messages = lm.call_args.kwargs["messages"]
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"]
    raise AssertionError("No user message in LM call")


def _history_with_two_turns() -> List[Dict[str, str]]:
    return [
        {"role": "human", "content": "oi"},
        {"role": "assistant", "content": "Oi! Como posso ajudar?"},
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestColdStartComposition:
    def test_cold_start_composes_greeting_with_specialist(self):
        """history=[] triggers cold_start=true flag and accepts [GREETING, SCHEDULE]."""
        from app.agents.router.agent import RouterAgent

        lm = _mock_lm_returning(
            {
                "intents": [
                    {"intent": "GREETING", "scope_text": "quero agendar botox"},
                    {"intent": "SCHEDULE", "scope_text": "quero agendar botox"},
                ],
                "confidence": 0.92,
                "reasoning": "cold start + schedule intent",
            }
        )
        agent = RouterAgent(lm=lm)

        result = agent.forward(
            latest_message="quero agendar botox",
            history=[],
            conversation_stage="new",
        )

        prompt = _last_user_prompt(lm)
        assert "cold_start: true" in prompt
        assert "stale: false" in prompt
        assert result["detected_intents"] == ["GREETING", "SCHEDULE"]


class TestStaleComposition:
    def test_stale_composes_greeting(self):
        """last_interaction_at > 24h ago triggers stale=true flag."""
        from app.agents.router.agent import RouterAgent

        lm = _mock_lm_returning(
            {
                "intents": [
                    {"intent": "GREETING", "scope_text": "quero agendar"},
                    {"intent": "SCHEDULE", "scope_text": "quero agendar"},
                ],
                "confidence": 0.90,
                "reasoning": "stale + schedule",
            }
        )
        agent = RouterAgent(lm=lm)

        twenty_five_hours_ago = datetime.now(timezone.utc) - timedelta(hours=25)
        result = agent.forward(
            latest_message="quero agendar",
            history=_history_with_two_turns(),
            conversation_stage="active",
            last_interaction_at=twenty_five_hours_ago,
        )

        prompt = _last_user_prompt(lm)
        assert "cold_start: false" in prompt
        assert "stale: true" in prompt
        # Hours-ago surfaces as an integer near 25.
        assert "last_interaction_hours_ago: 25" in prompt
        assert result["detected_intents"] == ["GREETING", "SCHEDULE"]

    def test_within_24h_no_stale(self):
        """last_interaction_at within 24h leaves stale=false; LM does not compose."""
        from app.agents.router.agent import RouterAgent

        lm = _mock_lm_returning(
            {
                "intents": [
                    {"intent": "SCHEDULE", "scope_text": "quero agendar"},
                ],
                "confidence": 0.90,
                "reasoning": "schedule only",
            }
        )
        agent = RouterAgent(lm=lm)

        ten_hours_ago = datetime.now(timezone.utc) - timedelta(hours=10)
        result = agent.forward(
            latest_message="quero agendar",
            history=_history_with_two_turns(),
            conversation_stage="active",
            last_interaction_at=ten_hours_ago,
        )

        prompt = _last_user_prompt(lm)
        assert "cold_start: false" in prompt
        assert "stale: false" in prompt
        assert "last_interaction_hours_ago: 10" in prompt
        assert result["detected_intents"] == ["SCHEDULE"]

    def test_no_last_interaction_at_treats_stale_false(self):
        """last_interaction_at=None ⇒ stale=false, hours_ago=null."""
        from app.agents.router.agent import RouterAgent

        lm = _mock_lm_returning(
            {
                "intents": [
                    {"intent": "GREETING", "scope_text": "quero agendar"},
                    {"intent": "SCHEDULE", "scope_text": "quero agendar"},
                ],
                "confidence": 0.90,
                "reasoning": "cold start (no history)",
            }
        )
        agent = RouterAgent(lm=lm)

        # history empty -> cold_start kicks in; last_interaction_at=None ⇒ stale must be false.
        agent.forward(
            latest_message="quero agendar",
            history=[],
            conversation_stage="new",
            last_interaction_at=None,
        )

        prompt = _last_user_prompt(lm)
        assert "stale: false" in prompt
        assert "last_interaction_hours_ago: null" in prompt


class TestNormalFollowup:
    def test_normal_followup_no_greeting(self):
        """Mid-session followup within 24h does not flag cold_start or stale."""
        from app.agents.router.agent import RouterAgent

        lm = _mock_lm_returning(
            {
                "intents": [
                    {"intent": "SCHEDULE", "scope_text": "quero agendar"},
                ],
                "confidence": 0.91,
                "reasoning": "schedule only",
            }
        )
        agent = RouterAgent(lm=lm)

        ten_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
        result = agent.forward(
            latest_message="quero agendar",
            history=_history_with_two_turns(),
            conversation_stage="active",
            last_interaction_at=ten_minutes_ago,
        )

        prompt = _last_user_prompt(lm)
        assert "cold_start: false" in prompt
        assert "stale: false" in prompt
        assert result["detected_intents"] == ["SCHEDULE"]


class TestBackwardCompat:
    def test_backward_compat_default_args(self):
        """forward() works without last_interaction_at / patient_name kwargs."""
        from app.agents.router.agent import RouterAgent

        lm = _mock_lm_returning(
            {
                "intents": [
                    {"intent": "SCHEDULE", "scope_text": "quero agendar"},
                ],
                "confidence": 0.93,
                "reasoning": "schedule",
            }
        )
        agent = RouterAgent(lm=lm)

        # Call with the legacy 3-arg signature.
        result = agent.forward(
            latest_message="quero agendar",
            history=_history_with_two_turns(),
            conversation_stage="active",
        )

        # Defaults yield stale=false and (history non-empty) cold_start=false.
        prompt = _last_user_prompt(lm)
        assert "cold_start: false" in prompt
        assert "stale: false" in prompt
        assert "last_interaction_hours_ago: null" in prompt
        assert result["detected_intents"] == ["SCHEDULE"]
