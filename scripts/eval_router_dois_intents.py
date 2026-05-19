"""
Evaluation harness — RouterAgent DOIS INTENTS dimension.

Tests classification of messages that carry TWO simultaneous intents.
Distribution in production: ~30-40% of messages have 2 intents.

The 8 cases below cover realistic combinations seen in clinic conversations,
focusing on:
  - INFO + CTA pairings (most common in production)
  - GREETING + something else
  - Edge case: INTAKE + HUMAN_ESCALATION (frustrated lead)

Verdict criterion (STRICT, per case):
  - intents list MUST equal expected_intents list (same values, same order).
    Order matters because the spec requires INFORMATIONAL -> CTA -> TERMINAL.
  - confidence MUST be >= 0.70.
  - scope_text MUST exist and be non-empty for each intent.

Usage:
    cd easyscale-sofia
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_router_dois_intents.py
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
# Test matrix — 8 cases (realistic 2-intent combinations)
# ============================================================================
# Ordering rule from spec: INFORMATIONAL -> CTA -> TERMINAL
# Informational: BUSINESS_INFO, TOPIC_KNOWLEDGE, REENGAGE, GREETING, UNCLASSIFIED
# CTA: INTAKE, SCHEDULE
# Terminal: HUMAN_ESCALATION

CASES: List[Dict[str, Any]] = [
    # Q1 — INFO + CTA (most common in production)
    {
        "id": "D.1",
        "label": "D.1 — BUSINESS_INFO + SCHEDULE (preco + agendamento)",
        "expected_intents": ["BUSINESS_INFO", "SCHEDULE"],
        "kwargs": {
            "latest_message": "quanto custa o botox? quero marcar uma sessao",
            "history": [],
            "conversation_stage": "new",
        },
    },
    {
        "id": "D.2",
        "label": "D.2 — TOPIC_KNOWLEDGE + INTAKE (pergunta neutra + interesse proprio)",
        "expected_intents": ["TOPIC_KNOWLEDGE", "INTAKE"],
        "kwargs": {
            "latest_message": "como funciona o preenchimento? tenho marca de expressao e queria entender se serve pra mim",
            "history": [],
            "conversation_stage": "new",
        },
    },
    # Q2 — GREETING + algo (paciente cumprimenta e ja traz intencao)
    {
        "id": "D.3",
        "label": "D.3 — GREETING + SCHEDULE",
        "expected_intents": ["GREETING", "SCHEDULE"],
        "kwargs": {
            "latest_message": "oi, queria marcar uma limpeza de pele pra amanha",
            "history": [],
            "conversation_stage": "new",
        },
    },
    {
        "id": "D.4",
        "label": "D.4 — GREETING + BUSINESS_INFO",
        "expected_intents": ["GREETING", "BUSINESS_INFO"],
        "kwargs": {
            "latest_message": "bom dia, voces aceitam Unimed?",
            "history": [],
            "conversation_stage": "new",
        },
    },
    # Q3 — Multi-info ou multi-CTA (variacoes do dia a dia)
    {
        "id": "D.5",
        "label": "D.5 — BUSINESS_INFO + INTAKE (preco + sintoma proprio)",
        "expected_intents": ["BUSINESS_INFO", "INTAKE"],
        "kwargs": {
            "latest_message": "quanto custa botox? tenho marcas de expressao",
            "history": [],
            "conversation_stage": "new",
        },
    },
    {
        "id": "D.6",
        "label": "D.6 — TOPIC_KNOWLEDGE + SCHEDULE (pergunta neutra + agendar)",
        "expected_intents": ["TOPIC_KNOWLEDGE", "SCHEDULE"],
        "kwargs": {
            "latest_message": "como funciona o peeling? tem horario amanha?",
            "history": [],
            "conversation_stage": "new",
        },
    },
    # Q4 — Edge cases envolvendo TERMINAL
    {
        "id": "D.7",
        "label": "D.7 — BUSINESS_INFO + HUMAN_ESCALATION (pergunta + frustrado)",
        "expected_intents": ["BUSINESS_INFO", "HUMAN_ESCALATION"],
        "kwargs": {
            "latest_message": "quanto custa? prefiro falar com um atendente humano",
            "history": [],
            "conversation_stage": "new",
        },
    },
    {
        "id": "D.8",
        "label": "D.8 — INTAKE + HUMAN_ESCALATION (lead que pede humano)",
        "expected_intents": ["INTAKE", "HUMAN_ESCALATION"],
        "kwargs": {
            "latest_message": "quero fazer botox, mas preciso falar com um atendente",
            "history": [],
            "conversation_stage": "new",
        },
    },
]


# ============================================================================
# Verdict (strict)
# ============================================================================

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


# ============================================================================
# Report
# ============================================================================

def _render_markdown(cases: List[Dict[str, Any]], latencies: List[float]) -> str:
    lines: List[str] = []
    lines.append("# Avaliação DOIS INTENTS — RouterAgent")
    lines.append("")
    lines.append(f"- **Modelo:** `{ROUTER_MODEL}`")
    lines.append(f"- **Temperature:** `{ROUTER_TEMPERATURE}`")
    lines.append(f"- **max_tokens:** `{ROUTER_MAX_TOKENS}`")
    lines.append(f"- **Threshold:** `{DEFAULT_CONFIDENCE_THRESHOLD}`")
    lines.append(f"- **Casos:** {len(cases)} (combinacoes realistas de 2 intents)")
    lines.append(
        f"- **Latência:** min={min(latencies):.0f}ms  "
        f"p50={sorted(latencies)[len(latencies)//2]:.0f}ms  "
        f"max={max(latencies):.0f}ms"
    )
    lines.append("")
    lines.append("## Critério (estrito)")
    lines.append("")
    lines.append("- `detected_intents` deve igualar `expected_intents` EXATAMENTE (mesma ordem).")
    lines.append("- Ordem esperada segue spec: INFORMATIONAL -> CTA -> TERMINAL.")
    lines.append("- `confidence >= 0.70`.")
    lines.append("- `scope_text` não-vazio em cada intent.")
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
    print(f"RouterAgent — DOIS INTENTS eval ({len(CASES)} cases)")
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
        "~/Documents/easyscale/kb/07-MVP/Tech/Tests/Agente Router"
    )
    os.makedirs(folder, exist_ok=True)
    date_tag = time.strftime("%Y-%m-%d")
    round_label = os.environ.get("EVAL_ROUND_LABEL", f"run {time.strftime('%H%M%S')}")
    default_path = os.path.join(folder, f"dois intents - {round_label} ({date_tag}).md")
    out_path = os.environ.get("EVAL_REPORT_PATH", default_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(CASES, latencies))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    run_eval()
