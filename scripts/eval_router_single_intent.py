"""
Evaluation harness — RouterAgent SINGLE INTENT dimension.

Tests classification of messages that should produce exactly ONE intent.
Covers each of the 8 IntentType values with 2 cases each (16 total),
varying tone, formality and specificity.

Verdict criterion (STRICT, per case):
  - intents list MUST equal expected_intents list (same values, same order)
  - confidence MUST be:
      - >= 0.70 when expected_intent != UNCLASSIFIED
      - < 0.70 OR result in UNCLASSIFIED when expected_intent == UNCLASSIFIED
        (because <0.70 triggers the UNCLASSIFIED fallback in forward())

scope_text is NOT enforced here — heuristic can't judge substring quality.
We only assert it exists and is non-empty for each intent (sanity check).

Usage:
    cd easyscale-sofia
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_router_single_intent.py
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict, List, Optional

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
# Test matrix — 8 intents × 2 cases = 16 cases
# ============================================================================

CASES: List[Dict[str, Any]] = [
    # ---------- BUSINESS_INFO (preço, horário, endereço, convênio, serviços) ----------
    {
        "id": "BI.1",
        "label": "BI.1 — preço explícito",
        "expected_intents": ["BUSINESS_INFO"],
        "expect_unclassified": False,
        "kwargs": {
            "latest_message": "quanto custa o botox?",
            "history": [],
            "conversation_stage": "new",
        },
    },
    {
        "id": "BI.2",
        "label": "BI.2 — convênio (Unimed)",
        "expected_intents": ["BUSINESS_INFO"],
        "expect_unclassified": False,
        "kwargs": {
            "latest_message": "voces aceitam Unimed?",
            "history": [],
            "conversation_stage": "new",
        },
    },
    # ---------- TOPIC_KNOWLEDGE (como funciona, contraindicações) ----------
    {
        "id": "TK.1",
        "label": "TK.1 — pergunta de mecanismo",
        "expected_intents": ["TOPIC_KNOWLEDGE"],
        "expect_unclassified": False,
        "kwargs": {
            "latest_message": "como funciona o preenchimento labial?",
            "history": [],
            "conversation_stage": "new",
        },
    },
    {
        "id": "TK.2",
        "label": "TK.2 — gatilho clínico (amamentação)",
        "expected_intents": ["TOPIC_KNOWLEDGE"],
        "expect_unclassified": False,
        "kwargs": {
            "latest_message": "posso fazer botox amamentando?",
            "history": [],
            "conversation_stage": "new",
        },
    },
    # ---------- REENGAGE (retomada de conversa inativa) ----------
    {
        "id": "RE.1",
        "label": "RE.1 — retomada explícita após pausa",
        "expected_intents": ["REENGAGE"],
        "expect_unclassified": False,
        "kwargs": {
            "latest_message": "oi, voltei pra continuar nosso papo",
            "history": [
                {"role": "human", "content": "queria saber sobre limpeza"},
                {"role": "ai", "content": "Claro! A limpeza parte de R$ 180."},
                {"role": "human", "content": "deixa eu pensar e te falo depois"},
            ],
            "conversation_stage": "inactive",
        },
    },
    {
        "id": "RE.2",
        "label": "RE.2 — paciente retoma após dias",
        "expected_intents": ["REENGAGE"],
        "expect_unclassified": False,
        "kwargs": {
            "latest_message": "olá, lembra de mim? continuamos nossa conversa?",
            "history": [
                {"role": "human", "content": "qual o valor do peeling?"},
                {"role": "ai", "content": "R$ 250 a sessão."},
            ],
            "conversation_stage": "inactive",
        },
    },
    # ---------- GREETING (saudação sem intenção adicional) ----------
    {
        "id": "GR.1",
        "label": "GR.1 — 'oi' simples primeira mensagem",
        "expected_intents": ["GREETING"],
        "expect_unclassified": False,
        "kwargs": {
            "latest_message": "oi",
            "history": [],
            "conversation_stage": "new",
        },
    },
    {
        "id": "GR.2",
        "label": "GR.2 — 'bom dia' simples primeira mensagem",
        "expected_intents": ["GREETING"],
        "expect_unclassified": False,
        "kwargs": {
            "latest_message": "bom dia",
            "history": [],
            "conversation_stage": "new",
        },
    },
    # ---------- UNCLASSIFIED (fora de contexto, baixa confiança) ----------
    {
        "id": "UN.1",
        "label": "UN.1 — sticker simulado / texto sem sentido",
        "expected_intents": ["UNCLASSIFIED"],
        "expect_unclassified": True,
        "kwargs": {
            "latest_message": "😂😂😂",
            "history": [],
            "conversation_stage": "new",
        },
    },
    {
        "id": "UN.2",
        "label": "UN.2 — fragmento ambíguo",
        "expected_intents": ["UNCLASSIFIED"],
        "expect_unclassified": True,
        "kwargs": {
            "latest_message": "aham",
            "history": [],
            "conversation_stage": "new",
        },
    },
    # ---------- INTAKE (declara interesse esperando consultoria) ----------
    {
        "id": "IN.1",
        "label": "IN.1 — interesse com sintoma",
        "expected_intents": ["INTAKE"],
        "expect_unclassified": False,
        "kwargs": {
            "latest_message": "tenho marcas de expressão na testa, o que voces fazem?",
            "history": [],
            "conversation_stage": "new",
        },
    },
    {
        "id": "IN.2",
        "label": "IN.2 — declaração direta de interesse",
        "expected_intents": ["INTAKE"],
        "expect_unclassified": False,
        "kwargs": {
            "latest_message": "quero fazer botox",
            "history": [],
            "conversation_stage": "new",
        },
    },
    # ---------- SCHEDULE (agendar/cancelar/remarcar) ----------
    {
        "id": "SC.1",
        "label": "SC.1 — 'quero marcar' direto",
        "expected_intents": ["SCHEDULE"],
        "expect_unclassified": False,
        "kwargs": {
            "latest_message": "quero marcar uma limpeza de pele",
            "history": [],
            "conversation_stage": "new",
        },
    },
    {
        "id": "SC.2",
        "label": "SC.2 — pergunta de disponibilidade",
        "expected_intents": ["SCHEDULE"],
        "expect_unclassified": False,
        "kwargs": {
            "latest_message": "tem horario amanha de manha?",
            "history": [],
            "conversation_stage": "new",
        },
    },
    # ---------- HUMAN_ESCALATION (pedido explícito de humano) ----------
    {
        "id": "HE.1",
        "label": "HE.1 — pedido explícito de atendente",
        "expected_intents": ["HUMAN_ESCALATION"],
        "expect_unclassified": False,
        "kwargs": {
            "latest_message": "quero falar com atendente",
            "history": [],
            "conversation_stage": "new",
        },
    },
    {
        "id": "HE.2",
        "label": "HE.2 — pedido com termo 'humano'",
        "expected_intents": ["HUMAN_ESCALATION"],
        "expect_unclassified": False,
        "kwargs": {
            "latest_message": "pode me transferir pra um humano por favor",
            "history": [],
            "conversation_stage": "new",
        },
    },
]


# ============================================================================
# Verdict
# ============================================================================

def auto_verdict(case: Dict[str, Any], result: Dict[str, Any]) -> str:
    expected = case["expected_intents"]
    detected = result.get("detected_intents") or []
    confidence = float(result.get("confidence") or 0.0)
    expect_unclassified = case["expect_unclassified"]

    # Strict equality of list (order + values).
    if detected != expected:
        return "NO"

    # Confidence rule.
    if expect_unclassified:
        # We expect the system to output UNCLASSIFIED. This can happen either
        # because the LLM returned UNCLASSIFIED directly OR because confidence
        # fell below threshold and forward() downgraded. Both are OK as long
        # as the final intent is UNCLASSIFIED.
        # The strict check above already ensures detected == ["UNCLASSIFIED"].
        return "YES"
    else:
        # Confidence must be >= threshold (otherwise forward() would have
        # downgraded to UNCLASSIFIED, which would already have failed the
        # strict equality above).
        if confidence < DEFAULT_CONFIDENCE_THRESHOLD:
            return "NO"

    # scope_text sanity (must exist + non-empty for each intent).
    for item in result.get("intents") or []:
        if not (item.get("scope_text") or "").strip():
            return "NO"

    return "YES"


# ============================================================================
# Report
# ============================================================================

def _render_markdown(cases: List[Dict[str, Any]], latencies: List[float]) -> str:
    lines: List[str] = []
    lines.append("# Avaliação SINGLE INTENT — RouterAgent")
    lines.append("")
    lines.append(f"- **Modelo:** `{ROUTER_MODEL}`")
    lines.append(f"- **Temperature:** `{ROUTER_TEMPERATURE}`")
    lines.append(f"- **max_tokens:** `{ROUTER_MAX_TOKENS}`")
    lines.append(f"- **Threshold:** `{DEFAULT_CONFIDENCE_THRESHOLD}` (UNCLASSIFIED se abaixo)")
    lines.append(f"- **Casos:** {len(cases)} (8 intents × 2)")
    lines.append(
        f"- **Latência:** min={min(latencies):.0f}ms  "
        f"p50={sorted(latencies)[len(latencies)//2]:.0f}ms  "
        f"p99={sorted(latencies)[int(len(latencies)*0.99)] if len(latencies) > 99 else max(latencies):.0f}ms  "
        f"max={max(latencies):.0f}ms"
    )
    lines.append("")
    lines.append("## Critério (estrito)")
    lines.append("")
    lines.append("- `detected_intents` deve ser exatamente igual à `expected_intents` (mesma ordem, mesmos valores).")
    lines.append("- Se `expect_unclassified=False`: `confidence >= 0.70`.")
    lines.append("- Se `expect_unclassified=True`: aceita confidence <0.70 (forward() faz downgrade).")
    lines.append("- `scope_text` deve existir e ser não-vazio em cada intent.")
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
        lines.append(f"**Esperado:** `{case['expected_intents']}` (unclassified={case['expect_unclassified']})")
        lines.append("")
        lines.append(f"**Input:** `{case['kwargs']['latest_message']!r}`")
        lines.append(f"**Stage:** `{case['kwargs']['conversation_stage']}`")
        if case["kwargs"]["history"]:
            lines.append(f"**History len:** {len(case['kwargs']['history'])}")
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


# ============================================================================
# Runner
# ============================================================================

def run_eval() -> None:
    agent = RouterAgent()
    _lm = agent._get_lm()
    if _lm is not None:
        _lm.cache = False

    print("=" * 100)
    print(f"RouterAgent — SINGLE INTENT eval ({len(CASES)} cases)")
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
        print(f"  EXPECT:    {case['expected_intents']}  (unclassified={case['expect_unclassified']})")
        print(f"  INPUT:     {kw['latest_message']!r}")
        print(f"  DETECTED:  {case['_detected']}  conf={case['_confidence']}")
        print(f"  REASONING: {case['_reasoning'][:200]}")
        print(f"  AUTO_VER:  {case['_auto_verdict']}")

    print()
    print("=" * 100)
    sorted_lats = sorted(latencies)
    p50 = sorted_lats[len(sorted_lats) // 2]
    p99 = sorted_lats[int(len(sorted_lats) * 0.99)] if len(sorted_lats) > 99 else max(sorted_lats)
    print(f"Latency: min={min(latencies):.0f}ms  p50={p50:.0f}ms  p99={p99:.0f}ms  max={max(latencies):.0f}ms")
    auto_yes = sum(1 for c in CASES if c["_auto_verdict"] == "YES")
    print(f"Auto-score: {auto_yes}/{len(CASES)} YES")
    print("=" * 100)

    folder = os.path.expanduser(
        "~/Documents/easyscale/kb/07-MVP/Tech/Tests/Agente Router"
    )
    os.makedirs(folder, exist_ok=True)
    date_tag = time.strftime("%Y-%m-%d")
    round_label = os.environ.get("EVAL_ROUND_LABEL", f"run {time.strftime('%H%M%S')}")
    default_path = os.path.join(folder, f"single intent - {round_label} ({date_tag}).md")
    out_path = os.environ.get("EVAL_REPORT_PATH", default_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(CASES, latencies))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    run_eval()
