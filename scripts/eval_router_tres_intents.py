"""
Evaluation harness — RouterAgent TRES INTENTS dimension (edge cases).

Tests classification of messages that carry THREE simultaneous intents.
Production distribution: very rare (~5% das mensagens), mas existem casos
genuinos onde paciente despeja varios intents em uma mensagem.

Foco do eval: garantir que o router NAO PERDE nenhum intent quando ha
3, e mantem o ordering correto (informacional -> CTA -> terminal).

Verdict (STRICT):
  - intents list == expected_intents (mesma ordem + mesmos valores)
  - confidence >= 0.70
  - scope_text nao-vazio em cada intent

Usage:
    cd easyscale-sofia
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_router_tres_intents.py
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.agents.router.agent import (  # noqa: E402
    RouterAgent,
    SYSTEM_PROMPT,
    ROUTER_MODEL,
    ROUTER_TEMPERATURE,
    ROUTER_MAX_TOKENS,
    DEFAULT_CONFIDENCE_THRESHOLD,
)


# ============================================================================
# Test matrix — 4 edge cases
# ============================================================================

CASES: List[Dict[str, Any]] = [
    {
        "id": "T.1",
        "label": "T.1 — GREETING + BUSINESS_INFO + SCHEDULE (paciente apressado)",
        "expected_intents": ["GREETING", "BUSINESS_INFO", "SCHEDULE"],
        "kwargs": {
            "latest_message": "oi! quanto custa a limpeza de pele? queria marcar pra essa semana",
            "history": [],
            "conversation_stage": "new",
        },
    },
    {
        "id": "T.2",
        "label": "T.2 — BUSINESS_INFO + TOPIC_KNOWLEDGE + SCHEDULE",
        "expected_intents": ["BUSINESS_INFO", "TOPIC_KNOWLEDGE", "SCHEDULE"],
        "kwargs": {
            "latest_message": "quanto custa o botox e como funciona? tem horario amanha?",
            "history": [],
            "conversation_stage": "new",
        },
    },
    {
        "id": "T.3",
        "label": "T.3 — GREETING + INTAKE + HUMAN_ESCALATION (lead frustrado desde o inicio)",
        "expected_intents": ["GREETING", "INTAKE", "HUMAN_ESCALATION"],
        "kwargs": {
            "latest_message": "oi, tenho marcas de expressao e queria fazer botox, mas prefiro falar com um atendente",
            "history": [],
            "conversation_stage": "new",
        },
    },
    {
        "id": "T.4",
        "label": "T.4 — BUSINESS_INFO + SCHEDULE + HUMAN_ESCALATION (ordering critico)",
        "expected_intents": ["BUSINESS_INFO", "SCHEDULE", "HUMAN_ESCALATION"],
        "kwargs": {
            "latest_message": "quanto custa? quero marcar amanha. depois quero falar com um atendente",
            "history": [],
            "conversation_stage": "new",
        },
    },
]


def auto_verdict(case: Dict[str, Any], result: Dict[str, Any]) -> str:
    expected = case["expected_intents"]
    detected = result.get("detected_intents") or []
    confidence = float(result.get("confidence") or 0.0)

    if detected != expected:
        return "NO"
    if confidence < DEFAULT_CONFIDENCE_THRESHOLD:
        return "NO"
    for item in result.get("intents") or []:
        if not (item.get("scope_text") or "").strip():
            return "NO"

    return "YES"


def _render_markdown(cases: List[Dict[str, Any]], latencies: List[float]) -> str:
    lines: List[str] = []
    lines.append("# Avaliação TRES INTENTS — RouterAgent (edge cases)")
    lines.append("")
    lines.append(f"- **Modelo:** `{ROUTER_MODEL}`")
    lines.append(f"- **Temperature:** `{ROUTER_TEMPERATURE}`")
    lines.append(f"- **max_tokens:** `{ROUTER_MAX_TOKENS}`")
    lines.append(f"- **Threshold:** `{DEFAULT_CONFIDENCE_THRESHOLD}`")
    lines.append(f"- **Casos:** {len(cases)} (edge cases multi-intent)")
    lines.append(
        f"- **Latência:** min={min(latencies):.0f}ms  "
        f"p50={sorted(latencies)[len(latencies)//2]:.0f}ms  "
        f"max={max(latencies):.0f}ms"
    )
    lines.append("")
    lines.append("## Critério (estrito)")
    lines.append("")
    lines.append("- `detected_intents` == `expected_intents` (3 intents, mesma ordem).")
    lines.append("- Ordering: INFORMATIONAL -> CTA -> TERMINAL.")
    lines.append("- `confidence >= 0.70`. `scope_text` não-vazio.")
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
        lines.append(f"**Esperado:** `{case['expected_intents']}`")
        lines.append("")
        lines.append(f"**Input:** `{case['kwargs']['latest_message']!r}`")
        lines.append("")
        lines.append(f"**Detected:** `{case['_detected']}`")
        lines.append(f"**Confidence:** `{case['_confidence']}`")
        lines.append(f"**Reasoning:** {case['_reasoning']!r}")
        lines.append("")
        lines.append("**Intents (full):**")
        lines.append("")
        lines.append("```")
        for item in case["_intents"]:
            lines.append(f"- {item['intent']}: {item['scope_text']!r}")
        lines.append("```")
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
    agent = RouterAgent()
    _lm = agent._get_lm()
    if _lm is not None:
        _lm.cache = False

    print("=" * 100)
    print(f"RouterAgent — TRES INTENTS eval ({len(CASES)} cases)")
    print(f"Model: {agent.model}, temp={agent.temperature}, max_tokens={agent.max_tokens}")
    print("=" * 100)

    latencies: List[float] = []

    for i, case in enumerate(CASES, 1):
        kw = case["kwargs"]
        t0 = time.perf_counter()
        try:
            result = agent.forward(**kw)
        except Exception as exc:
            result = {
                "intents": [],
                "detected_intents": [],
                "confidence": 0.0,
                "reasoning": f"EXCEPTION: {type(exc).__name__}: {exc}",
            }
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        case["_detected"] = result.get("detected_intents") or []
        case["_confidence"] = result.get("confidence") or 0.0
        case["_reasoning"] = result.get("reasoning") or ""
        case["_intents"] = result.get("intents") or []
        case["_elapsed_ms"] = elapsed
        case["_auto_verdict"] = auto_verdict(case, result)

        print()
        print("-" * 100)
        print(f"[{i:>2}] {case['label']}  ({elapsed:.0f}ms)")
        print("-" * 100)
        print(f"  EXPECT:    {case['expected_intents']}")
        print(f"  INPUT:     {kw['latest_message']!r}")
        print(f"  DETECTED:  {case['_detected']}  conf={case['_confidence']}")
        print(f"  REASONING: {case['_reasoning'][:250]}")
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
        "~/Documents/easyscale/kb/07-MVP/Tech/Tests/Agente Router"
    )
    os.makedirs(folder, exist_ok=True)
    date_tag = time.strftime("%Y-%m-%d")
    round_label = os.environ.get("EVAL_ROUND_LABEL", f"run {time.strftime('%H%M%S')}")
    default_path = os.path.join(folder, f"tres intents - {round_label} ({date_tag}).md")
    out_path = os.environ.get("EVAL_REPORT_PATH", default_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(CASES, latencies))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    run_eval()
