"""
Evaluation harness focused exclusively on the PERÍODO DO DIA dimension.

Tests the v26 rule:

    Se o paciente utilizar "bom dia", "boa tarde" ou "boa noite":
    - Espelhe exatamente o cumprimento utilizado.
    - O cumprimento do paciente tem prioridade sobre o cumprimento do few_shot.

    Se não houver cumprimento explícito na mensagem do paciente:
    - siga o padrão demonstrado no few_shot.

The matrix has 4 quadrants × 2 cases = 8 cases:

  Q1 — patient uses "bom dia/tarde/noite" + few_shot uses SAME
       -> response uses patient's greeting (which matches few_shot)
  Q2 — patient uses "bom dia/tarde/noite" + few_shot uses DIFFERENT
       -> response uses PATIENT's greeting (priority over few_shot)
  Q3 — patient uses neutral "oi/olá" + few_shot uses "bom dia/tarde/noite"
       -> response uses FEW_SHOT's greeting (no patient signal)
  Q4 — patient uses "boa noite" (less common) + few_shot uses neutral
       -> response uses PATIENT's "boa noite" (priority)

Verdict criterion (YES per case):
  - Q1/Q2/Q4: response MUST start with the patient's specific greeting token.
  - Q3: response MUST start with the few_shot's greeting token (since no
        patient-specific period was provided).

Usage:
    cd easyscale-sofia
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_periodo.py
"""

from __future__ import annotations

import os
import re
import time
import warnings
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.core.config import init_dspy  # noqa: E402

init_dspy()

import dspy  # noqa: E402

dspy.settings.lm.cache = False

from app.agents.greeting.agent import (  # noqa: E402
    GreetingAgent,
    _build_user_prompt,
    _coerce_few_shot,
    _normalize_contact_name,
    SYSTEM_PROMPT,
    GREETING_MODEL,
    GREETING_TEMPERATURE,
    GREETING_MAX_TOKENS,
    TECHNICAL_FALLBACK,
)


# ============================================================================
# Few-shot fixtures (v26: single string per clinic)
# ============================================================================

FEW_SHOT_BOM_DIA = "Bom dia! Aqui é da Lumina Estética. Como posso te ajudar?"
FEW_SHOT_BOA_TARDE = "Boa tarde! Aqui é da Vita Premium. Em que posso ser útil?"
FEW_SHOT_OLA_NEUTRO = "Olá! Aqui é da Clínica Bella. Como posso te ajudar?"
FEW_SHOT_OI_NEUTRO = "Oi! Aqui é do Studio Bem-Estar."


# ============================================================================
# Test matrix
# ============================================================================

CASES: List[Dict[str, Any]] = [
    # ----- Q1: paciente e few_shot CONCORDAM no periodo -> espelha
    {
        "id": "Q1.1",
        "quadrant": "paciente e few_shot CONCORDAM no periodo -> espelha cumprimento",
        "label": "Q1.1 — 'bom dia' do paciente, few_shot 'Bom dia!'",
        "expected": "bom dia",
        "kwargs": {
            "patient_message": "bom dia",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_BOM_DIA,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q1.2",
        "quadrant": "paciente e few_shot CONCORDAM no periodo -> espelha cumprimento",
        "label": "Q1.2 — 'boa tarde' do paciente, few_shot 'Boa tarde!'",
        "expected": "boa tarde",
        "kwargs": {
            "patient_message": "boa tarde",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "few_shot": FEW_SHOT_BOA_TARDE,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q2: paciente DIVERGE do few_shot -> prioridade do paciente
    {
        "id": "Q2.1",
        "quadrant": "paciente DIVERGE do few_shot -> prioridade do paciente",
        "label": "Q2.1 — 'boa tarde' do paciente, few_shot 'Bom dia!'",
        "expected": "boa tarde",
        "kwargs": {
            "patient_message": "boa tarde",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_BOM_DIA,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q2.2",
        "quadrant": "paciente DIVERGE do few_shot -> prioridade do paciente",
        "label": "Q2.2 — 'bom dia' do paciente, few_shot 'Boa tarde!'",
        "expected": "bom dia",
        "kwargs": {
            "patient_message": "bom dia",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "few_shot": FEW_SHOT_BOA_TARDE,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q3: paciente NEUTRO + few_shot temporal -> usa do few_shot
    {
        "id": "Q3.1",
        "quadrant": "paciente NEUTRO + few_shot temporal -> usa cumprimento do few_shot",
        "label": "Q3.1 — 'oi' do paciente, few_shot 'Bom dia!'",
        "expected": "bom dia",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_BOM_DIA,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q3.2",
        "quadrant": "paciente NEUTRO + few_shot temporal -> usa cumprimento do few_shot",
        "label": "Q3.2 — 'olá' do paciente, few_shot 'Boa tarde!'",
        "expected": "boa tarde",
        "kwargs": {
            "patient_message": "olá",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "few_shot": FEW_SHOT_BOA_TARDE,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q4: paciente 'boa noite' + few_shot neutro -> prioridade do paciente
    {
        "id": "Q4.1",
        "quadrant": "paciente 'boa noite' + few_shot neutro -> prioridade do paciente",
        "label": "Q4.1 — 'boa noite' do paciente, few_shot 'Olá!'",
        "expected": "boa noite",
        "kwargs": {
            "patient_message": "boa noite",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Bella",
            "assistant_name": "Sofia",
            "few_shot": FEW_SHOT_OLA_NEUTRO,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q4.2",
        "quadrant": "paciente 'boa noite' + few_shot neutro -> prioridade do paciente",
        "label": "Q4.2 — 'boa noite' do paciente, few_shot 'Oi!'",
        "expected": "boa noite",
        "kwargs": {
            "patient_message": "boa noite",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Studio Bem-Estar",
            "assistant_name": "Maya",
            "few_shot": FEW_SHOT_OI_NEUTRO,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
]


# ============================================================================
# Heuristic auto-verdict
# ============================================================================

def auto_verdict(case: Dict[str, Any], response: str) -> str:
    """Check if response STARTS with the expected greeting token."""
    if not response:
        return "NO"
    expected = case["expected"].lower()
    # Lowercase the first ~20 chars; check expected appears near the start.
    lowered = response.lower().lstrip()
    if lowered.startswith(expected):
        return "YES"
    # Also accept it if it's in the first 15 chars (e.g. emoji prefix).
    if expected in lowered[:25]:
        return "YES"
    return "NO"


# ============================================================================
# Report renderer
# ============================================================================

def _render_config_block() -> List[str]:
    lines: List[str] = []
    lines.append("## Configuração do agente (no momento da rodada)")
    lines.append("")
    lines.append(f"- **Modelo:** `{GREETING_MODEL}`")
    lines.append(f"- **Temperature:** `{GREETING_TEMPERATURE}`")
    lines.append(f"- **max_tokens:** `{GREETING_MAX_TOKENS}`")
    lines.append(f"- **Fallback técnico:** `{TECHNICAL_FALLBACK!r}`")
    lines.append("- **Cache:** desabilitado (eval)")
    lines.append("")
    lines.append("### Few-shots utilizados nesta bateria (v26: 1 por fixture)")
    lines.append("")
    for name, fs in [
        ("FEW_SHOT_BOM_DIA (Lumina, 'Bom dia!')", FEW_SHOT_BOM_DIA),
        ("FEW_SHOT_BOA_TARDE (Vita Premium, 'Boa tarde!')", FEW_SHOT_BOA_TARDE),
        ("FEW_SHOT_OLA_NEUTRO (Clínica Bella, 'Olá!')", FEW_SHOT_OLA_NEUTRO),
        ("FEW_SHOT_OI_NEUTRO (Studio Bem-Estar, 'Oi!')", FEW_SHOT_OI_NEUTRO),
    ]:
        lines.append(f"**{name}**:")
        lines.append("```")
        lines.append(fs)
        lines.append("```")
        lines.append("")
    lines.append("### System prompt")
    lines.append("")
    lines.append("```")
    lines.append(SYSTEM_PROMPT)
    lines.append("```")
    lines.append("")
    return lines


def _render_markdown(cases: List[Dict[str, Any]], latencies: List[float]) -> str:
    lines: List[str] = []
    lines.append("# Avaliação de PERÍODO DO DIA — GreetingAgent")
    lines.append("")
    lines.append(f"- **Modelo:** `{GREETING_MODEL}`")
    lines.append(f"- **Temperature:** `{GREETING_TEMPERATURE}`")
    lines.append(f"- **max_tokens:** `{GREETING_MAX_TOKENS}`")
    lines.append(f"- **Casos:** {len(cases)} (4 quadrantes × 2)")
    lines.append(
        f"- **Latência:** min={min(latencies):.0f}ms  "
        f"p50={sorted(latencies)[len(latencies)//2]:.0f}ms  "
        f"max={max(latencies):.0f}ms"
    )
    lines.append("")
    lines.append("## Critério")
    lines.append("")
    lines.append(
        "Para cada caso, marque YES se o aspecto **PERÍODO DO DIA** estiver correto."
    )
    lines.append("")
    lines.append("- **expected = bom dia / boa tarde / boa noite** → o `response` DEVE iniciar com esse cumprimento.")
    lines.append("")
    lines.append("Outras dimensões fora de escopo.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.extend(_render_config_block())
    lines.append("---")
    lines.append("")

    quadrants_seen: List[str] = []
    for case in cases:
        if case["quadrant"] not in quadrants_seen:
            quadrants_seen.append(case["quadrant"])
            lines.append(f"## {case['quadrant']}")
            lines.append("")

        lines.append(f"### {case['label']}")
        lines.append("")
        lines.append(f"_Latência: {case['_elapsed_ms']:.0f}ms_ — `{case['_reasoning']}`")
        lines.append("")
        lines.append(f"**Expectativa:** `{case['expected']}`")
        lines.append("")
        lines.append("**Input (user prompt enviado ao LLM):**")
        lines.append("")
        lines.append("```")
        lines.append(case["_user_prompt"])
        lines.append("```")
        lines.append("")
        lines.append("**Output:**")
        lines.append("")
        if not case["_output"]:
            lines.append("> _(vazio — fallback, silêncio intencional ou erro)_")
        else:
            lines.append(f"> {case['_output']}")
        lines.append("")
        llm_reasoning = case.get("_llm_reasoning", "")
        if llm_reasoning:
            lines.append("**Reasoning do modelo (depuração):**")
            lines.append("")
            lines.append("```")
            lines.append(llm_reasoning)
            lines.append("```")
            lines.append("")
        lines.append(f"**Veredito automático heurístico:** `{case['_auto_verdict']}`")
        lines.append("")
        lines.append("**Veredito humano:**")
        lines.append("")
        lines.append("- [ ] YES")
        lines.append("- [ ] NO — motivo: ___")
        lines.append("")
        lines.append("---")
        lines.append("")

    auto_yes = sum(1 for c in cases if c["_auto_verdict"] == "YES")
    lines.append("## Score automático (heurístico)")
    lines.append("")
    lines.append(f"- **Auto-YES:** {auto_yes}/{len(cases)}")
    lines.append("")
    lines.append("**Aprovado pelo humano:** [ ] Sim   [ ] Não (re-rodar)")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Runner
# ============================================================================

def run_eval() -> None:
    temp_override = os.environ.get("EVAL_TEMPERATURE")
    if temp_override is not None:
        agent = GreetingAgent(temperature=float(temp_override))
    else:
        agent = GreetingAgent()
    _agent_lm = agent._get_lm()
    if _agent_lm is not None:
        _agent_lm.cache = False
        if temp_override is not None:
            _agent_lm.kwargs["temperature"] = float(temp_override)

    print("=" * 100)
    print(f"GreetingAgent — PERÍODO DO DIA eval ({len(CASES)} cases)")
    print(f"Model: {agent.model}, temp={agent.temperature}, max_tokens={agent.max_tokens}")
    print("=" * 100)

    latencies: List[float] = []

    for i, case in enumerate(CASES, 1):
        kw = case["kwargs"]
        patient_name = _normalize_contact_name(kw.get("patient_name"))
        few_shot = _coerce_few_shot(kw.get("few_shot"), None, None, None)
        case["_user_prompt"] = _build_user_prompt(
            patient_message=kw["patient_message"],
            patient_intents=kw.get("patient_intents") or [],
            patient_name=patient_name,
            clinic_name=kw["clinic_name"],
            assistant_name=kw["assistant_name"],
            few_shot=few_shot,
            session_summary=kw.get("session_summary", ""),
            recent_relevant_messages=kw.get("recent_relevant_messages") or [],
            time_gap_hours=kw.get("time_gap_hours"),
        )

        t0 = time.perf_counter()
        out = agent.forward(**kw)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        content = out["messages"][0]["content"] if out["messages"] else ""
        reasoning = out["reasoning"]
        llm_reasoning = (out.get("data") or {}).get("llm_reasoning", "")
        case["_output"] = content
        case["_reasoning"] = reasoning
        case["_llm_reasoning"] = llm_reasoning
        case["_elapsed_ms"] = elapsed
        case["_auto_verdict"] = auto_verdict(case, content)

        print()
        print("-" * 100)
        print(f"[{i:>2}] {case['label']}  ({elapsed:.0f}ms)")
        print("-" * 100)
        print(f"  EXPECT_START: {case['expected']!r}")
        print(f"  INPUT:        {kw['patient_message']!r}")
        print(f"  OUTPUT:       {content!r}")
        if llm_reasoning:
            print(f"  REASONING:    {llm_reasoning}")
        print(f"  AUTO_VER:     {case['_auto_verdict']}")

    print()
    print("=" * 100)
    print(
        f"Latency: min={min(latencies):.0f}ms  "
        f"p50={sorted(latencies)[len(latencies)//2]:.0f}ms  "
        f"max={max(latencies):.0f}ms"
    )
    auto_yes = sum(1 for c in CASES if c["_auto_verdict"] == "YES")
    print(f"Auto-score: {auto_yes}/{len(CASES)} YES (heurístico)")
    print("=" * 100)

    folder = os.path.expanduser(
        "~/Documents/easyscale/kb/07-MVP/Tech/Tests/Agente Greeting"
    )
    os.makedirs(folder, exist_ok=True)
    date_tag = time.strftime("%Y-%m-%d")
    round_label = os.environ.get("EVAL_ROUND_LABEL", f"run {time.strftime('%H%M%S')}")
    temp_suffix = ""
    if temp_override is not None:
        temp_suffix = f" temp{float(temp_override):.1f}"
    default_path = os.path.join(folder, f"periodo v26{temp_suffix} - {round_label} ({date_tag}).md")
    out_path = os.environ.get("EVAL_REPORT_PATH", default_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(CASES, latencies))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    run_eval()
