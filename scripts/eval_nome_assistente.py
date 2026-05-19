"""
Evaluation harness focused exclusively on the NOME ASSISTENTE dimension.

Tests the v26 rule:

    Utilize o nome da assistente apenas se isso estiver presente no few_shot.
    Caso contrário: utilize apenas o nome da clínica na apresentação.

    A presença dos campos clinic_name e assistant_name NÃO autoriza apresentação
    por si só.

The matrix has 4 quadrants × 2 cases = 8 cases:

  Q1 — assistant_name PRESENT + few_shot USES assistant name
       -> response MUST include the assistant name
  Q2 — assistant_name PRESENT + few_shot does NOT mention assistant
       -> response MUST NOT include assistant name (use clinic only or none)
  Q3 — assistant_name NULL/empty + few_shot has assistant placeholder
       -> response MUST NOT include any assistant name
  Q4 — assistant_name PRESENT + few_shot has NO presentation at all
       -> response MUST NOT include assistant name (no presentation to put it in)

Verdict criterion (YES per case):
  - Q1: response MUST contain assistant_name.
  - Q2/Q3/Q4: response MUST NOT contain assistant_name or any literal assistant
        name from the few_shot (e.g. "Helena", "Iris").

Usage:
    cd easyscale-sofia
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_nome_assistente.py
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

# WITH assistant name — clinic style introduces the assistant.
FEW_SHOT_WITH_ASSISTANT_ACOLHEDOR = "Olá! Sou a Helena da Lumina Estética. Como posso te ajudar?"
FEW_SHOT_WITH_ASSISTANT_INFORMAL = "Oi! Aqui é a Maya do Studio Bem-Estar."

# WITHOUT assistant name (clinic name only).
FEW_SHOT_NO_ASSISTANT_FORMAL = "Olá, seja bem-vindo à Clínica Vita Premium."
FEW_SHOT_NO_ASSISTANT_NEUTRO = "Oi! Aqui é da Clínica Bella. Como posso te ajudar?"


# ============================================================================
# Test matrix
# ============================================================================

CASES: List[Dict[str, Any]] = [
    # ----- Q1: assistant_name presente + few_shot USA -> response usa nome
    {
        "id": "Q1.1",
        "quadrant": "assistant_name PRESENTE + few_shot USA -> response usa nome",
        "label": "Q1.1 — assistant='Helena', Lumina few_shot com Helena",
        "expected": "usa_nome_assistente",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Helena",
            "few_shot": FEW_SHOT_WITH_ASSISTANT_ACOLHEDOR,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q1.2",
        "quadrant": "assistant_name PRESENTE + few_shot USA -> response usa nome",
        "label": "Q1.2 — assistant='Maya', Studio few_shot com Maya",
        "expected": "usa_nome_assistente",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Studio Bem-Estar",
            "assistant_name": "Maya",
            "few_shot": FEW_SHOT_WITH_ASSISTANT_INFORMAL,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q2: assistant_name presente + few_shot SEM -> NAO usa nome
    {
        "id": "Q2.1",
        "quadrant": "assistant_name PRESENTE + few_shot SEM -> response NAO usa nome assistente",
        "label": "Q2.1 — assistant='Iris', Vita Premium few_shot sem assistente",
        "expected": "nao_usa_nome_assistente",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_NO_ASSISTANT_FORMAL,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q2.2",
        "quadrant": "assistant_name PRESENTE + few_shot SEM -> response NAO usa nome assistente",
        "label": "Q2.2 — assistant='Sofia', Clinica Bella few_shot sem assistente",
        "expected": "nao_usa_nome_assistente",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Bella",
            "assistant_name": "Sofia",
            "few_shot": FEW_SHOT_NO_ASSISTANT_NEUTRO,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q3: assistant_name=None + few_shot tem nome de assistente -> NAO inventa
    {
        "id": "Q3.1",
        "quadrant": "assistant_name NULL + few_shot tem placeholder -> NAO usa nome do few_shot",
        "label": "Q3.1 — assistant=None, few_shot 'Sou a Helena da Lumina'",
        "expected": "nao_usa_nome_do_few_shot",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "",  # empty/null
            "few_shot": FEW_SHOT_WITH_ASSISTANT_ACOLHEDOR,  # has "Helena"
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q3.2",
        "quadrant": "assistant_name NULL + few_shot tem placeholder -> NAO usa nome do few_shot",
        "label": "Q3.2 — assistant=None, few_shot 'Aqui é a Maya do Studio'",
        "expected": "nao_usa_nome_do_few_shot",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Studio Bem-Estar",
            "assistant_name": "",
            "few_shot": FEW_SHOT_WITH_ASSISTANT_INFORMAL,  # has "Maya"
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q4: assistant_name presente + few_shot SEM apresentacao -> NAO inventa apresentacao com nome
    {
        "id": "Q4.1",
        "quadrant": "assistant_name PRESENTE + few_shot SEM apresentacao -> NAO inventa",
        "label": "Q4.1 — assistant='Iris', few_shot 'Olá! Como posso te ajudar?'",
        "expected": "nao_inventa_apresentacao_com_nome",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Studio Alfa",
            "assistant_name": "Iris",
            "few_shot": "Olá! Como posso te ajudar?",  # no presentation at all
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q4.2",
        "quadrant": "assistant_name PRESENTE + few_shot SEM apresentacao -> NAO inventa",
        "label": "Q4.2 — assistant='Sofia', few_shot 'Bom dia! Como posso te ajudar?'",
        "expected": "nao_inventa_apresentacao_com_nome",
        "kwargs": {
            "patient_message": "bom dia",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Bella",
            "assistant_name": "Sofia",
            "few_shot": "Bom dia! Como posso te ajudar?",
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
]


# ============================================================================
# Heuristic auto-verdict
# ============================================================================

# Names that appear in our few_shots (used as placeholders).
_FEW_SHOT_ASSISTANT_NAMES = ["helena", "maya"]


def auto_verdict(case: Dict[str, Any], response: str) -> str:
    expected = case["expected"]
    lowered = (response or "").lower()
    kw = case["kwargs"]
    assistant_name = (kw.get("assistant_name") or "").lower().strip()

    if expected == "usa_nome_assistente":
        if not assistant_name:
            return "CHECK"
        return "YES" if assistant_name in lowered else "NO"

    if expected == "nao_usa_nome_assistente":
        if assistant_name and assistant_name in lowered:
            return "NO"
        return "YES"

    if expected == "nao_usa_nome_do_few_shot":
        # Check no placeholder name from few_shot appears.
        for name in _FEW_SHOT_ASSISTANT_NAMES:
            if name in lowered:
                return "NO"
        return "YES"

    if expected == "nao_inventa_apresentacao_com_nome":
        # Should not include assistant name AT ALL since few_shot has no
        # presentation pattern.
        if assistant_name and assistant_name in lowered:
            return "NO"
        # Also check no presentation tokens appeared (which would be a bigger
        # failure but covered by APRESENTAÇÃO dimension; here we just check name).
        return "YES"

    return "CHECK"


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
        ("FEW_SHOT_WITH_ASSISTANT_ACOLHEDOR (Lumina, com 'Helena')", FEW_SHOT_WITH_ASSISTANT_ACOLHEDOR),
        ("FEW_SHOT_WITH_ASSISTANT_INFORMAL (Studio, com 'Maya')", FEW_SHOT_WITH_ASSISTANT_INFORMAL),
        ("FEW_SHOT_NO_ASSISTANT_FORMAL (Vita Premium)", FEW_SHOT_NO_ASSISTANT_FORMAL),
        ("FEW_SHOT_NO_ASSISTANT_NEUTRO (Clínica Bella)", FEW_SHOT_NO_ASSISTANT_NEUTRO),
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
    lines.append("# Avaliação de NOME ASSISTENTE — GreetingAgent")
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
        "Para cada caso, marque YES se o aspecto **NOME ASSISTENTE** estiver correto:"
    )
    lines.append("")
    lines.append("- **expected = usa_nome_assistente** → o `response` DEVE conter `assistant_name`.")
    lines.append("- **expected = nao_usa_nome_assistente** → o `response` NÃO DEVE conter `assistant_name`.")
    lines.append("- **expected = nao_usa_nome_do_few_shot** → o `response` NÃO DEVE conter nome literal do few_shot.")
    lines.append("- **expected = nao_inventa_apresentacao_com_nome** → o `response` NÃO DEVE inventar apresentação só porque assistant_name existe.")
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
    print(f"GreetingAgent — NOME ASSISTENTE eval ({len(CASES)} cases)")
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
        print(f"  EXPECT:    {case['expected']}")
        print(f"  ASSIST:    {kw.get('assistant_name')!r}")
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
    default_path = os.path.join(folder, f"nome assistente v26{temp_suffix} - {round_label} ({date_tag}).md")
    out_path = os.environ.get("EVAL_REPORT_PATH", default_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(CASES, latencies))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    run_eval()
