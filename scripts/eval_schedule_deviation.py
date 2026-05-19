"""
Evaluation harness — ScheduleRouter DEVIATION dimension.

Tests classification of messages that BREAK the expected sequence:
  - DEV.1: patient cancels in the middle of intake
  - DEV.2: patient asks to reschedule during reminder
  - DEV.3: ambiguous/unclassifiable message -> FALLBACK
  - DEV.4: patient cancels during confirmation flow

Verdict (STRICT):
  - next_intent MUST equal expected_next_intent (CANCEL/CHANGE/FALLBACK)
  - is_deviation MUST be True
  - confidence >= 0.70 (when expected is CANCEL/CHANGE)
  - confidence may be < 0.70 when expected is FALLBACK (forced fallback)

Usage:
    cd easyscale-sofia
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_schedule_deviation.py
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.agents.router.schedule_router import (  # noqa: E402
    ScheduleRouter,
    SYSTEM_PROMPT,
    SCHEDULE_ROUTER_MODEL,
    SCHEDULE_ROUTER_TEMPERATURE,
    SCHEDULE_ROUTER_MAX_TOKENS,
    DEFAULT_SCHEDULE_CONFIDENCE_THRESHOLD,
)


SEQ_EVALUATION = ["SCHEDULE_INTAKE", "SCHEDULE_CASHIER", "SCHEDULE_EVALUATION", "SCHEDULE_COMPLETION"]
SEQ_REMINDER = ["SCHEDULE_REMINDER", "SCHEDULE_COMPLETION"]
SEQ_CONFIRMATION = ["SCHEDULE_CONFIRMATION", "SCHEDULE_COMPLETION"]


CASES: List[Dict[str, Any]] = [
    {
        "id": "DEV.1",
        "label": "DEV.1 — paciente cancela no meio do intake",
        "expected_next_intent": "SCHEDULE_CANCEL",
        "expect_fallback": False,
        "kwargs": {
            "latest_message": "esquece, quero cancelar minha consulta",
            "history": [
                {"role": "ai", "content": "Voce toma algum medicamento de uso continuo?"},
            ],
            "sequence": SEQ_EVALUATION,
            "current_stage": "SCHEDULE_INTAKE",
            "session_data": [{"name": "evaluation", "data": {"service": "fotona"}}],
        },
    },
    {
        "id": "DEV.2",
        "label": "DEV.2 — paciente pede remarcar durante reminder",
        "expected_next_intent": "SCHEDULE_CHANGE",
        "expect_fallback": False,
        "kwargs": {
            "latest_message": "preciso remarcar pra semana que vem, nao consigo ir hoje",
            "history": [
                {"role": "ai", "content": "Lembrando do seu horario hoje as 16h."},
            ],
            "sequence": SEQ_REMINDER,
            "current_stage": "SCHEDULE_REMINDER",
            "session_data": [{"name": "reminder", "data": {"appointment_id": "apt_999"}}],
        },
    },
    {
        "id": "DEV.3",
        "label": "DEV.3 — mensagem sem sentido -> FALLBACK",
        "expected_next_intent": "SCHEDULE_FALLBACK",
        "expect_fallback": True,
        "kwargs": {
            "latest_message": "kkkkkk",
            "history": [],
            "sequence": SEQ_EVALUATION,
            "current_stage": "new",
            "session_data": [{"name": "evaluation", "data": {"service": "fotona"}}],
        },
    },
    {
        "id": "DEV.4",
        "label": "DEV.4 — paciente cancela durante confirmation",
        "expected_next_intent": "SCHEDULE_CANCEL",
        "expect_fallback": False,
        "kwargs": {
            "latest_message": "na verdade, cancela esse agendamento, nao posso ir mesmo",
            "history": [
                {"role": "ai", "content": "Voce confirma sua presenca na consulta de amanha?"},
            ],
            "sequence": SEQ_CONFIRMATION,
            "current_stage": "SCHEDULE_CONFIRMATION",
            "session_data": [{"name": "confirmation", "data": {"appointment_id": "apt_555"}}],
        },
    },
]


def auto_verdict(case: Dict[str, Any], result: Dict[str, Any]) -> str:
    expected = case["expected_next_intent"]
    next_intent = result.get("next_intent")
    is_deviation = bool(result.get("is_deviation", False))
    confidence = float(result.get("confidence") or 0.0)
    expect_fallback = case.get("expect_fallback", False)

    if next_intent != expected:
        return "NO"
    if not is_deviation:
        # Deviation eval requires is_deviation=True.
        return "NO"
    if expect_fallback:
        # FALLBACK can come from confidence-below-threshold forcing; accept any
        # confidence value as long as the intent matched.
        return "YES"
    if confidence < DEFAULT_SCHEDULE_CONFIDENCE_THRESHOLD:
        return "NO"
    return "YES"


def _render_markdown(cases: List[Dict[str, Any]], latencies: List[float]) -> str:
    lines: List[str] = []
    lines.append("# Avaliação DEVIATION — ScheduleRouter")
    lines.append("")
    lines.append(f"- **Modelo:** `{SCHEDULE_ROUTER_MODEL}`")
    lines.append(f"- **Temperature:** `{SCHEDULE_ROUTER_TEMPERATURE}`")
    lines.append(f"- **max_tokens:** `{SCHEDULE_ROUTER_MAX_TOKENS}`")
    lines.append(f"- **Threshold:** `{DEFAULT_SCHEDULE_CONFIDENCE_THRESHOLD}`")
    lines.append(f"- **Casos:** {len(cases)} (desvios pontuais)")
    sorted_lats = sorted(latencies)
    p50 = sorted_lats[len(sorted_lats) // 2]
    lines.append(
        f"- **Latência:** min={min(latencies):.0f}ms  "
        f"p50={p50:.0f}ms  max={max(latencies):.0f}ms"
    )
    lines.append("")
    lines.append("## Critério (estrito)")
    lines.append("")
    lines.append("- `next_intent` igual ao esperado.")
    lines.append("- `is_deviation` deve ser `True`.")
    lines.append("- `confidence >= 0.70` (exceto quando esperado é FALLBACK — pode vir de força).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## System prompt")
    lines.append("")
    lines.append("```")
    lines.append(SYSTEM_PROMPT)
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")

    for case in cases:
        lines.append(f"### {case['label']}")
        lines.append("")
        lines.append(f"_Latência: {case['_elapsed_ms']:.0f}ms_")
        lines.append("")
        lines.append(f"**Esperado:** `{case['expected_next_intent']}` (fallback={case['expect_fallback']})")
        lines.append("")
        lines.append(f"**Sequence:** `{case['kwargs']['sequence']}`")
        lines.append(f"**Current stage:** `{case['kwargs']['current_stage']}`")
        lines.append(f"**Input:** `{case['kwargs']['latest_message']!r}`")
        lines.append("")
        lines.append(f"**Detected:** `{case['_next_intent']}`  is_deviation=`{case['_is_deviation']}`  conf=`{case['_confidence']}`")
        lines.append(f"**Reasoning:** {case['_reasoning']!r}")
        lines.append("")
        lines.append(f"**Veredito automático:** `{case['_auto_verdict']}`")
        lines.append("")
        lines.append("**Veredito humano:**")
        lines.append("")
        lines.append("- [ ] YES")
        lines.append("- [ ] NO — motivo: ___")
        lines.append("")
        lines.append("---")
        lines.append("")

    auto_yes = sum(1 for c in cases if c["_auto_verdict"] == "YES")
    lines.append("## Score automático")
    lines.append("")
    lines.append(f"- **Auto-YES:** {auto_yes}/{len(cases)}")
    lines.append("")
    return "\n".join(lines)


def run_eval() -> None:
    router = ScheduleRouter()
    _lm = router._get_lm()
    if _lm is not None:
        _lm.cache = False

    print("=" * 100)
    print(f"ScheduleRouter — DEVIATION eval ({len(CASES)} cases)")
    print(f"Model: {router.model}, temp={router.temperature}, max_tokens={router.max_tokens}")
    print("=" * 100)

    latencies: List[float] = []

    for i, case in enumerate(CASES, 1):
        kw = case["kwargs"]
        t0 = time.perf_counter()
        try:
            result = router.forward(**kw)
        except Exception as exc:
            result = {
                "next_intent": "SCHEDULE_FALLBACK",
                "is_deviation": True,
                "session_data": [],
                "confidence": 0.0,
                "reasoning": f"EXCEPTION: {type(exc).__name__}: {exc}",
            }
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        case["_next_intent"] = result.get("next_intent")
        case["_is_deviation"] = result.get("is_deviation")
        case["_confidence"] = result.get("confidence") or 0.0
        case["_reasoning"] = result.get("reasoning") or ""
        case["_session_data"] = result.get("session_data") or []
        case["_elapsed_ms"] = elapsed
        case["_auto_verdict"] = auto_verdict(case, result)

        print()
        print("-" * 100)
        print(f"[{i:>2}] {case['label']}  ({elapsed:.0f}ms)")
        print("-" * 100)
        print(f"  EXPECT:    {case['expected_next_intent']} (fallback={case['expect_fallback']})")
        print(f"  SEQUENCE:  {kw['sequence']}")
        print(f"  STAGE:     {kw['current_stage']}")
        print(f"  INPUT:     {kw['latest_message']!r}")
        print(f"  DETECTED:  {case['_next_intent']}  dev={case['_is_deviation']}  conf={case['_confidence']}")
        print(f"  REASONING: {case['_reasoning'][:200]}")
        print(f"  AUTO_VER:  {case['_auto_verdict']}")

    print()
    print("=" * 100)
    sorted_lats = sorted(latencies)
    p50 = sorted_lats[len(sorted_lats) // 2]
    print(f"Latency: min={min(latencies):.0f}ms  p50={p50:.0f}ms  max={max(latencies):.0f}ms")
    auto_yes = sum(1 for c in CASES if c["_auto_verdict"] == "YES")
    print(f"Auto-score: {auto_yes}/{len(CASES)} YES")
    print("=" * 100)

    folder = os.path.expanduser(
        "~/Documents/easyscale/kb/07-MVP/Tech/Tests/Schedule Router"
    )
    os.makedirs(folder, exist_ok=True)
    date_tag = time.strftime("%Y-%m-%d")
    round_label = os.environ.get("EVAL_ROUND_LABEL", f"run {time.strftime('%H%M%S')}")
    default_path = os.path.join(folder, f"deviation - {round_label} ({date_tag}).md")
    out_path = os.environ.get("EVAL_REPORT_PATH", default_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(CASES, latencies))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    run_eval()
