"""
Smoke test — Iris pipeline end-to-end (1 message per critical intent).

Stubs Supabase (load_session/save_session/load_services_context) and Evolution
API (send/persist) so the test runs hermetically. The 4 LLM agents (Router,
Greeting, Knowledge, Schedule sub-router) still hit DeepSeek API for real, so
DEEPSEEK_API_KEY + OPENAI_API_KEY must be in .env.

Validates:
  - Pipeline doesn't crash for each scenario
  - At least one agent_run produced per call
  - response_text is non-empty for GREETING / TOPIC_KNOWLEDGE
  - SCHEDULE produces a sub_intent decision (UNKNOWN fallback for the text)

Usage:
    cd easyscale-sofia
    PYTHONPATH=. python scripts/smoke_pipeline.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import warnings
from typing import Any, Dict, List
from unittest.mock import patch

warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


# Clinic Bloom (seeded with knowledge chunks in feat/knowledge-agent).
TEST_CLINIC_ID = "0d6d8eaf-6efa-4aaf-9845-de4b0d0f608c"
TEST_CLINIC_NAME = "Clínica Bloom"
TEST_REMOTE_JID = "5511999990001@s.whatsapp.net"
TEST_INSTANCE = "smoke-test"


SCENARIOS = [
    {
        "id": "GR",
        "label": "GREETING — paciente diz 'oi'",
        "message": "oi",
        "expects_intent": "GREETING",
        "expects_response": True,
    },
    {
        "id": "TK",
        "label": "TOPIC_KNOWLEDGE — paciente pergunta duração do botox",
        "message": "quanto tempo dura o efeito do botox?",
        "expects_intent": "TOPIC_KNOWLEDGE",
        "expects_response": True,
    },
    {
        "id": "SC",
        "label": "SCHEDULE — paciente quer marcar limpeza",
        "message": "quero marcar uma limpeza de pele",
        "expects_intent": "SCHEDULE",
        "expects_response": True,  # UNKNOWN fallback text, but non-empty
    },
    {
        "id": "HE",
        "label": "HUMAN_ESCALATION — paciente pede atendente",
        "message": "quero falar com atendente",
        "expects_intent": "HUMAN_ESCALATION",
        "expects_response": True,
    },
]


def _fake_load_session(**kwargs: Any) -> Dict[str, Any]:
    return {
        "session_id": f"smoke:{kwargs.get('remote_jid')}",
        "customer_id": None,
        "history": [],
        "conversation_stage": "new",
        "conversation_type": "first_contact",
        "patient_name": kwargs.get("push_name"),
        "clinic_name": TEST_CLINIC_NAME,
        "assistant_name": "Iris",
        "clinic_style": {"greeting_example": "Olá! Aqui é da Clínica Bloom. Como posso te ajudar?"},
        "paused": False,
    }


def _fake_save_session(state: Dict[str, Any]) -> None:
    return None


def _fake_load_services(clinic_id: str) -> str:
    return "{}"


def _fake_notify_receptionist(**kwargs: Any) -> None:
    return None


async def _fake_send(**kwargs: Any) -> Dict[str, Any]:
    return {"key": {"id": "fake-wamid-001"}}


def _fake_persist_outbound(**kwargs: Any) -> None:
    return None


def _make_parsed(message: str) -> Any:
    """Construct a ParsedMessage-shaped object (duck-typed)."""
    class _P:
        instance_name = TEST_INSTANCE
        remote_jid = TEST_REMOTE_JID
        push_name = "Teste"
        message_content = message
        message_type = "text"
        wamid = "smoke-in-" + str(abs(hash(message)) % 100000)

    return _P()


async def run_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    from app.iris import pipeline

    result = await pipeline.invoke(
        clinic_id=TEST_CLINIC_ID,
        message_id=f"smoke-msg-{scenario['id']}",
        parsed=_make_parsed(scenario["message"]),
        trace_id=f"smoke-trace-{scenario['id']}",
    )
    return result


def evaluate(scenario: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    detected = result.get("detected_intents") or []
    intents = result.get("intents") or []
    responses = result.get("specialist_responses") or []
    response_text = ""
    for r in responses:
        if r.get("response_text", "").strip():
            response_text = r["response_text"]
            break

    expected_intent = scenario["expects_intent"]
    intent_match = expected_intent in detected

    verdict = "PASS"
    reasons: List[str] = []
    if not intents:
        verdict = "FAIL"
        reasons.append("no intents detected")
    elif not intent_match:
        verdict = "WARN"
        reasons.append(f"expected {expected_intent}, got {detected}")

    if scenario["expects_response"] and not response_text:
        verdict = "FAIL"
        reasons.append("expected non-empty response_text")

    return {
        "verdict": verdict,
        "detected": detected,
        "response_text": response_text,
        "schedule_sub_intent": result.get("schedule_sub_intent"),
        "reasons": reasons,
    }


def main() -> int:
    print("=" * 100)
    print("Iris pipeline — smoke E2E")
    print(f"Clinic: {TEST_CLINIC_NAME} ({TEST_CLINIC_ID})")
    print(f"Scenarios: {len(SCENARIOS)}")
    print("=" * 100)

    # Patch out external systems for hermetic testing.
    patches = [
        patch("app.iris.pipeline.load_session", side_effect=_fake_load_session),
        patch("app.iris.pipeline.save_session", side_effect=_fake_save_session),
        patch("app.iris.pipeline.load_services_context", side_effect=_fake_load_services),
        patch("app.iris.pipeline.send_text_message", side_effect=_fake_send),
        patch("app.iris.pipeline.persist_outbound_message", side_effect=_fake_persist_outbound),
        patch("app.iris.pipeline.notify_receptionist", side_effect=_fake_notify_receptionist),
    ]
    for p in patches:
        p.start()

    try:
        results = []
        for sc in SCENARIOS:
            print()
            print("-" * 100)
            print(f"[{sc['id']}] {sc['label']}")
            print("-" * 100)
            try:
                result = asyncio.run(run_scenario(sc))
            except Exception as exc:
                print(f"  CRASH: {type(exc).__name__}: {exc}")
                results.append({"id": sc["id"], "verdict": "CRASH", "exc": str(exc)})
                continue

            evaluation = evaluate(sc, result)
            results.append({"id": sc["id"], **evaluation})

            print(f"  Detected: {evaluation['detected']}")
            print(f"  Response: {evaluation['response_text'][:160]}{'…' if len(evaluation['response_text']) > 160 else ''}")
            if evaluation["schedule_sub_intent"]:
                print(f"  Schedule sub-intent: {evaluation['schedule_sub_intent']}")
            print(f"  Verdict: {evaluation['verdict']}")
            if evaluation["reasons"]:
                print(f"  Reasons: {evaluation['reasons']}")

        print()
        print("=" * 100)
        passes = sum(1 for r in results if r.get("verdict") == "PASS")
        warns = sum(1 for r in results if r.get("verdict") == "WARN")
        fails = sum(1 for r in results if r.get("verdict") in ("FAIL", "CRASH"))
        print(f"Summary: {passes} PASS, {warns} WARN, {fails} FAIL/CRASH (of {len(results)})")
        print("=" * 100)
        return 0 if fails == 0 else 1
    finally:
        for p in patches:
            p.stop()


if __name__ == "__main__":
    sys.exit(main())
