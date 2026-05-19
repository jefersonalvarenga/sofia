"""
Evaluation harness focused exclusively on the APRESENTAÇÃO dimension.

Tests whether the GreetingAgent correctly applies the rule:

    Apresentações só devem ocorrer SE recent_relevant_messages estiver
    vazio E for o padrão predominante dos few-shots.

The matrix has 4 quadrants (2 cases each = 8 cases total):

  Q1 — FIRST_CONTACT_WITH_PRESENTATION_FEWSHOTS  -> deve apresentar
  Q2 — FIRST_CONTACT_WITHOUT_PRESENTATION_FEWSHOTS -> não apresenta
  Q3 — RESUMPTION_WITH_PRESENTATION_FEWSHOTS  -> não reapresenta
  Q4 — RESUMPTION_WITHOUT_PRESENTATION_FEWSHOTS -> não apresenta

Verdict criterion (YES per case):
  - Q1: response MUST contain a presentation token (e.g. "Aqui é da X").
  - Q2/Q3/Q4: response MUST NOT contain a presentation token.

All other dimensions (CTA, name, cordiality, period) are out of scope
for this evaluation — judge only the presentation aspect.

Usage:
    cd easyscale-sofia
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_apresentacao.py
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
# Few-shot fixtures (v26: single string per clinic, NOT a list)
# ============================================================================

# WITH PRESENTATION — explicitly presents the clinic.
FEW_SHOT_PRES_ACOLHEDOR = "Olá! Aqui é da Lumina Estética. Como posso te ajudar hoje?"
FEW_SHOT_PRES_SEMIFORMAL = "Bom dia, aqui é da Vita Premium. Em que posso ser útil?"

# WITHOUT PRESENTATION — no "Aqui é da ...", no "sou X da ...".
FEW_SHOT_SEM_APRESENTACAO_NEUTRO = "Olá! Como posso te ajudar?"
FEW_SHOT_SEM_APRESENTACAO_PERIODO = "Bom dia! Como posso te ajudar?"


# ============================================================================
# Test matrix — 4 quadrants × 2 cases = 8 cases
# ============================================================================

CASES: List[Dict[str, Any]] = [
    # ----- Q1: primeiro contato + few-shots COM apresentação -> DEVE apresentar
    {
        "id": "Q1.1",
        "quadrant": "primeiro_contato + few-shots COM apresentação -> DEVE apresentar",
        "label": "Q1.1 — Lumina acolhedor, 'oi', primeiro contato",
        "expected": "apresenta",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_PRES_ACOLHEDOR,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q1.2",
        "quadrant": "primeiro_contato + few-shots COM apresentação -> DEVE apresentar",
        "label": "Q1.2 — Vita Premium semiformal, 'bom dia', primeiro contato",
        "expected": "apresenta",
        "kwargs": {
            "patient_message": "bom dia",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "few_shot": FEW_SHOT_PRES_SEMIFORMAL,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q2: primeiro contato + few-shots SEM apresentação -> NÃO apresenta
    {
        "id": "Q2.1",
        "quadrant": "primeiro_contato + few-shots SEM apresentação -> NÃO apresenta",
        "label": "Q2.1 — few-shots neutros, 'oi', clinic_name='Clínica Bella'",
        "expected": "nao_apresenta",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Bella",
            "assistant_name": "Sofia",
            "few_shot": FEW_SHOT_SEM_APRESENTACAO_NEUTRO,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q2.2",
        "quadrant": "primeiro_contato + few-shots SEM apresentação -> NÃO apresenta",
        "label": "Q2.2 — few-shots período, 'bom dia', clinic_name='Studio Alfa'",
        "expected": "nao_apresenta",
        "kwargs": {
            "patient_message": "bom dia",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Studio Alfa",
            "assistant_name": "Maya",
            "few_shot": FEW_SHOT_SEM_APRESENTACAO_PERIODO,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q3: retomada + few-shots COM apresentação -> NÃO reapresenta
    {
        "id": "Q3.1",
        "quadrant": "retomada + few-shots COM apresentação -> NÃO reapresenta",
        "label": "Q3.1 — Lumina, retomada 2h após saudação prévia, 'oi' novo",
        "expected": "nao_apresenta",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_PRES_ACOLHEDOR,
            "recent_relevant_messages": [
                {"role": "patient", "content": "oi, qual o valor da limpeza?"},
                {
                    "role": "greeting",
                    "content": "Olá! Aqui é da Lumina Estética. A limpeza parte de R$ 180.",
                },
            ],
            "session_summary": "Paciente perguntou sobre preço de limpeza.",
            "time_gap_hours": 2,
        },
    },
    {
        "id": "Q3.2",
        "quadrant": "retomada + few-shots COM apresentação -> NÃO reapresenta",
        "label": "Q3.2 — Vita Premium, retomada 50h, 'boa tarde'",
        "expected": "nao_apresenta",
        "kwargs": {
            "patient_message": "boa tarde",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "few_shot": FEW_SHOT_PRES_SEMIFORMAL,
            "recent_relevant_messages": [
                {"role": "patient", "content": "oi"},
                {
                    "role": "greeting",
                    "content": "Olá! Aqui é da Vita Premium. Em que posso ser útil?",
                },
                {"role": "patient", "content": "obrigada, vou pensar"},
            ],
            "session_summary": "Paciente perguntou sobre serviços e agradeceu.",
            "time_gap_hours": 50,
        },
    },
    # ----- Q4: retomada + few-shots SEM apresentação -> NÃO apresenta
    {
        "id": "Q4.1",
        "quadrant": "retomada + few-shots SEM apresentação -> NÃO apresenta",
        "label": "Q4.1 — few-shots neutros, retomada 24h, 'oi'",
        "expected": "nao_apresenta",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Bella",
            "assistant_name": "Sofia",
            "few_shot": FEW_SHOT_SEM_APRESENTACAO_NEUTRO,
            "recent_relevant_messages": [
                {"role": "patient", "content": "oi"},
                {"role": "greeting", "content": "Olá! Em que posso ajudar?"},
            ],
            "session_summary": "",
            "time_gap_hours": 24,
        },
    },
    {
        "id": "Q4.2",
        "quadrant": "retomada + few-shots SEM apresentação -> NÃO apresenta",
        "label": "Q4.2 — few-shots período, retomada 120h, 'bom dia'",
        "expected": "nao_apresenta",
        "kwargs": {
            "patient_message": "bom dia",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Studio Alfa",
            "assistant_name": "Maya",
            "few_shot": FEW_SHOT_SEM_APRESENTACAO_PERIODO,
            "recent_relevant_messages": [
                {"role": "patient", "content": "oi"},
                {"role": "greeting", "content": "Olá! Como posso te ajudar?"},
                {"role": "patient", "content": "depois eu volto"},
            ],
            "session_summary": "Paciente disse que voltaria depois.",
            "time_gap_hours": 120,
        },
    },
]


# ============================================================================
# Heuristic auto-verdict (best-effort hint; humano decide o final)
# ============================================================================

_PRESENTATION_PATTERNS = [
    r"\baqui é d[aoe]\b",
    r"\baqui é\s+\w+\s+d[aoe]\b",  # "aqui é Iris da..."
    r"\bsou\s+\w+\s+d[aoe]\b",     # "sou Helena da..."
    r"\bseja bem-vind[oa]\b",
]


def detect_presentation(text: str) -> bool:
    """Best-effort detection of a presentation token in the response."""
    if not text:
        return False
    lowered = text.lower()
    for pat in _PRESENTATION_PATTERNS:
        if re.search(pat, lowered):
            return True
    return False


def auto_verdict(case: Dict[str, Any], response: str) -> str:
    """Return 'YES', 'NO' or 'CHECK' as a heuristic hint."""
    has_presentation = detect_presentation(response)
    if case["expected"] == "apresenta":
        return "YES" if has_presentation else "NO"
    # expected == "nao_apresenta"
    return "NO" if has_presentation else "YES"


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
        ("FEW_SHOT_PRES_ACOLHEDOR (Lumina, com apresentação)", FEW_SHOT_PRES_ACOLHEDOR),
        ("FEW_SHOT_PRES_SEMIFORMAL (Vita Premium, com apresentação)", FEW_SHOT_PRES_SEMIFORMAL),
        ("FEW_SHOT_SEM_APRESENTACAO_NEUTRO", FEW_SHOT_SEM_APRESENTACAO_NEUTRO),
        ("FEW_SHOT_SEM_APRESENTACAO_PERIODO", FEW_SHOT_SEM_APRESENTACAO_PERIODO),
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
    lines.append("# Avaliação de APRESENTAÇÃO — GreetingAgent")
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
        "Para cada caso, marque YES se o aspecto **APRESENTAÇÃO** estiver correto:"
    )
    lines.append("")
    lines.append("- **expected = apresenta** → o `response` DEVE conter token de apresentação (ex: \"Aqui é da Clínica X\", \"sou Helena da...\").")
    lines.append("- **expected = nao_apresenta** → o `response` NÃO DEVE conter token de apresentação.")
    lines.append("")
    lines.append(
        "Outros aspectos (CTA, nome do paciente, cordialidade, período do dia) **estão fora do escopo deste round**."
    )
    lines.append("")
    lines.append("Há um **veredito automático heurístico** sugerido por caso. Confirme ou ajuste manualmente.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.extend(_render_config_block())
    lines.append("---")
    lines.append("")

    # Group cases by quadrant for readability.
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

    # Auto-score summary.
    auto_yes = sum(1 for c in cases if c["_auto_verdict"] == "YES")
    lines.append("## Score automático (heurístico)")
    lines.append("")
    lines.append(f"- **Auto-YES:** {auto_yes}/{len(cases)}")
    lines.append("")
    lines.append("(Sujeito a confirmação humana — heurística pode classificar mal saídas ambíguas.)")
    lines.append("")
    lines.append("**Aprovado pelo humano:** [ ] Sim   [ ] Não (re-rodar)")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Runner
# ============================================================================

def run_eval() -> None:
    # Allow temperature override via env var for experiments.
    temp_override = os.environ.get("EVAL_TEMPERATURE")
    if temp_override is not None:
        agent = GreetingAgent(temperature=float(temp_override))
    else:
        agent = GreetingAgent()
    # Force-disable cache on the agent's own LM to measure real latency.
    _agent_lm = agent._get_lm()
    if _agent_lm is not None:
        _agent_lm.cache = False
        # Propagate temperature override to the LM itself (DSPy stores it there).
        if temp_override is not None:
            _agent_lm.kwargs["temperature"] = float(temp_override)

    print("=" * 100)
    print(f"GreetingAgent — APRESENTAÇÃO eval ({len(CASES)} cases)")
    print(f"Model: {agent.model}, temp={agent.temperature}, max_tokens={agent.max_tokens}")
    print("=" * 100)

    latencies: List[float] = []

    for i, case in enumerate(CASES, 1):
        kw = case["kwargs"]

        # Reconstruct the user prompt the LLM will see (mirrors forward()).
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
        print(f"  EXPECT:    {case['expected']}")
        print(f"  INPUT:     {kw['patient_message']!r}")
        print(f"  OUTPUT:    {content!r}")
        if llm_reasoning:
            print(f"  REASONING: {llm_reasoning}")
        print(f"  AUTO_VER:  {case['_auto_verdict']}")

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
    default_path = os.path.join(folder, f"apresentacao v26{temp_suffix} - {round_label} ({date_tag}).md")
    out_path = os.environ.get("EVAL_REPORT_PATH", default_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(CASES, latencies))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    run_eval()
