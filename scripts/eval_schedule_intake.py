"""Evaluation harness — SCHEDULE_INTAKE sub-agent (LLM-as-judge ready).

Spec: kb/07-MVP/Tech/03-Discussoes/schedule/01 - Spec SCHEDULE_INTAKE.md §10

This script runs the 12 canonical cases from Spec §13 against the REAL
DeepSeek V4 Flash LM (no mocks) and reports pass/fail per dimension:

  - parse_pydantic_valid   (>= 97% target)
  - latency_p99            (<= 2s target)
  - sensitivity_recall     (>= 90% target)
  - precision_multi        (>= 85% target)

Status: SKELETON. The unit-test suite (tests/agents/test_schedule_intake.py)
exercises the same 12 cases against a mocked LM and is what gates CI. This
eval is for periodic LLM-quality checks against real DeepSeek; it requires
``DEEPSEEK_API_KEY`` in the environment.

Usage:
    cd easyscale-sofia
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_schedule_intake.py

TODO(intake-eval): wire actual LLM-as-judge once we have the eval cases
materialized in fixtures/intake_eval.json. For now this script just exits
with a clear message so it's discoverable but does not silently no-op in CI.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

# Eval cases (mirrors tests/agents/test_schedule_intake.py)
# Schema:
#   {
#     "name": str,
#     "latest_message": str,
#     "history": List[Dict[str, str]],
#     "questions": List[Dict[str, Any]],
#     "contraindications": List[str],
#     "expected": {
#       "next_hint": Optional[str],   # "ESCALATE_TO_HUMAN" or None
#       "sub_intent_complete": bool,
#       "matched_contraindication": Optional[str],
#       "next_question_id": Optional[str],
#       "new_answers_count": int,
#     }
#   }
EVAL_CASES: List[Dict[str, Any]] = [
    # Populated from the 12 canonical cases. See test_schedule_intake.py
    # for the structured-data shapes; the eval here mirrors them but reaches
    # for the real LM.
]


def main() -> int:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print(
            "DEEPSEEK_API_KEY not set — refusing to run the LLM-real eval. "
            "Set the key and retry, or run `pytest tests/agents/test_schedule_intake.py` "
            "for the mocked CI suite.",
            file=sys.stderr,
        )
        return 2

    if not EVAL_CASES:
        print(
            "SKELETON: EVAL_CASES is empty. Materialize the 12 cases from "
            "tests/agents/test_schedule_intake.py before running this eval. "
            "TODO(intake-eval).",
            file=sys.stderr,
        )
        return 3

    # TODO(intake-eval): instantiate ScheduleIntakeAgent without mock, iterate
    # cases, measure latency, Pydantic validity, sensitivity (escalation
    # recall), and multi-answer precision. Emit a summary table at the end.
    print("Not implemented yet — see TODO(intake-eval).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
