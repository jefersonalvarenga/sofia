"""
Evaluation harness focused exclusively on the NOME PACIENTE dimension.

Tests the v26 rule:

    Utilize o nome do paciente apenas se isso estiver presente no few_shot.
    A presença de patient_name NÃO implica uso obrigatório do nome na resposta.
    Nunca utilize o nome do paciente apenas porque ele apareceu anteriormente
    no histórico.

The matrix has 4 quadrants × 2 cases = 8 cases:

  Q1 — patient_name PRESENT + few_shot USES name token
       -> response MUST include the patient name
  Q2 — patient_name PRESENT + few_shot does NOT use name
       -> response MUST NOT include the patient name
  Q3 — patient_name NULL + few_shot USES name token (e.g. "Olá, {nome}!")
       -> response MUST NOT include any name (no fabrication)
  Q4 — patient_name in HISTORY only (not in input) + few_shot does NOT use name
       -> response MUST NOT include the name from history

Verdict criterion (YES per case):
  - Q1: response MUST contain `patient_name` (case-sensitive substring match
        OK; we'll lowercase compare).
  - Q2/Q3/Q4: response MUST NOT contain any patient name (we check the
        case's patient_name + any name appearing in history).

Few_shot "uses name token" means it contains "{nome}" or a literal name
like "Camila" as a placeholder.

Usage:
    cd easyscale-sofia
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_nome_paciente.py
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

# WITH patient name token — uses a literal name as placeholder.
FEW_SHOT_WITH_NAME_ACOLHEDOR = "Olá, Camila! Aqui é da Lumina Estética. Como posso te ajudar?"
FEW_SHOT_WITH_NAME_INFORMAL = "E aí, Lucas! Aqui é do Studio Bem-Estar."

# WITHOUT patient name — clinic style does NOT personalize.
FEW_SHOT_NO_NAME_FORMAL = "Olá, seja bem-vindo à Clínica Vita Premium."
FEW_SHOT_NO_NAME_NEUTRO = "Oi! Aqui é da Clínica Bella. Como posso te ajudar?"


# ============================================================================
# Test matrix — 4 quadrants × 2 cases = 8 cases
# ============================================================================

CASES: List[Dict[str, Any]] = [
    # ----- Q1: patient_name presente + few_shot usa nome -> response usa nome
    {
        "id": "Q1.1",
        "quadrant": "patient_name PRESENTE + few_shot USA nome -> response usa nome",
        "label": "Q1.1 — patient_name='Mariana', Lumina com nome no padrao",
        "expected": "usa_nome",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": "Mariana",
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_WITH_NAME_ACOLHEDOR,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q1.2",
        "quadrant": "patient_name PRESENTE + few_shot USA nome -> response usa nome",
        "label": "Q1.2 — patient_name='Pedro', Studio Bem-Estar com nome",
        "expected": "usa_nome",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": "Pedro",
            "clinic_name": "Studio Bem-Estar",
            "assistant_name": "Maya",
            "few_shot": FEW_SHOT_WITH_NAME_INFORMAL,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q2: patient_name presente + few_shot SEM nome -> response NAO usa nome
    {
        "id": "Q2.1",
        "quadrant": "patient_name PRESENTE + few_shot SEM nome -> response NAO usa nome",
        "label": "Q2.1 — patient_name='Mariana', Vita Premium sem nome no padrao",
        "expected": "nao_usa_nome",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": "Mariana",
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "few_shot": FEW_SHOT_NO_NAME_FORMAL,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q2.2",
        "quadrant": "patient_name PRESENTE + few_shot SEM nome -> response NAO usa nome",
        "label": "Q2.2 — patient_name='Pedro', Clinica Bella sem nome no padrao",
        "expected": "nao_usa_nome",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": "Pedro",
            "clinic_name": "Clínica Bella",
            "assistant_name": "Sofia",
            "few_shot": FEW_SHOT_NO_NAME_NEUTRO,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q3: patient_name NULL + few_shot USA nome -> response NAO inventa nome
    {
        "id": "Q3.1",
        "quadrant": "patient_name NULL + few_shot USA nome -> response NAO inventa",
        "label": "Q3.1 — patient_name=None, Lumina com nome 'Camila' no few_shot",
        "expected": "nao_inventa_nome",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_WITH_NAME_ACOLHEDOR,  # has "Camila"
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q3.2",
        "quadrant": "patient_name NULL + few_shot USA nome -> response NAO inventa",
        "label": "Q3.2 — patient_name=None, Studio com nome 'Lucas' no few_shot",
        "expected": "nao_inventa_nome",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Studio Bem-Estar",
            "assistant_name": "Maya",
            "few_shot": FEW_SHOT_WITH_NAME_INFORMAL,  # has "Lucas"
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q4: nome aparece SO no historico + few_shot SEM nome -> NAO usa
    {
        "id": "Q4.1",
        "quadrant": "nome SO no historico + few_shot SEM nome -> NAO usa nome do historico",
        "label": "Q4.1 — historico menciona 'Carolina', few_shot sem nome, retomada",
        "expected": "nao_usa_nome",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "few_shot": FEW_SHOT_NO_NAME_FORMAL,
            "recent_relevant_messages": [
                {"role": "patient", "content": "oi, sou Carolina"},
                {"role": "greeting", "content": "Olá, seja bem-vindo à Clínica Vita Premium."},
            ],
            "session_summary": "Paciente disse o nome dela é Carolina.",
            "time_gap_hours": 1,
        },
    },
    {
        "id": "Q4.2",
        "quadrant": "nome SO no historico + few_shot SEM nome -> NAO usa nome do historico",
        "label": "Q4.2 — historico menciona 'João', few_shot sem nome, retomada 24h",
        "expected": "nao_usa_nome",
        "kwargs": {
            "patient_message": "boa tarde",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Bella",
            "assistant_name": "Sofia",
            "few_shot": FEW_SHOT_NO_NAME_NEUTRO,
            "recent_relevant_messages": [
                {"role": "patient", "content": "oi, aqui é o João"},
                {"role": "greeting", "content": "Oi! Aqui é da Clínica Bella. Como posso te ajudar?"},
            ],
            "session_summary": "Paciente se identificou como João.",
            "time_gap_hours": 24,
        },
    },
]


# ============================================================================
# Heuristic auto-verdict
# ============================================================================

# Names that might appear in few_shot (literals used as placeholders).
_FEW_SHOT_NAMES = ["camila", "lucas"]


def _names_to_check(case: Dict[str, Any]) -> List[str]:
    """All names that should NOT appear in response when expected=nao_usa_nome
    or nao_inventa_nome. Includes patient_name + names from history + few_shot
    placeholders."""
    names: List[str] = []
    kw = case["kwargs"]
    if kw.get("patient_name"):
        names.append(kw["patient_name"].lower())
    for msg in kw.get("recent_relevant_messages") or []:
        # Best-effort: scan for capitalized words in patient turns.
        if msg.get("role") == "patient":
            for word in re.findall(r"\b([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)\b", msg.get("content", "")):
                if len(word) >= 3 and word.lower() not in {"oi", "olá", "boa", "bom"}:
                    names.append(word.lower())
    # Add few_shot placeholder names.
    fs = (kw.get("few_shot") or "").lower()
    for name in _FEW_SHOT_NAMES:
        if name in fs:
            names.append(name)
    return list(set(names))


def auto_verdict(case: Dict[str, Any], response: str) -> str:
    expected = case["expected"]
    lowered = (response or "").lower()
    kw = case["kwargs"]

    if expected == "usa_nome":
        target = (kw.get("patient_name") or "").lower()
        if not target:
            return "CHECK"
        return "YES" if target in lowered else "NO"

    if expected in ("nao_usa_nome", "nao_inventa_nome"):
        for name in _names_to_check(case):
            if name in lowered:
                return "NO"
        return "YES"

    return "CHECK"


# ============================================================================
# Report renderer (same structure)
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
        ("FEW_SHOT_WITH_NAME_ACOLHEDOR (Lumina, com 'Camila')", FEW_SHOT_WITH_NAME_ACOLHEDOR),
        ("FEW_SHOT_WITH_NAME_INFORMAL (Studio, com 'Lucas')", FEW_SHOT_WITH_NAME_INFORMAL),
        ("FEW_SHOT_NO_NAME_FORMAL (Vita Premium)", FEW_SHOT_NO_NAME_FORMAL),
        ("FEW_SHOT_NO_NAME_NEUTRO (Clínica Bella)", FEW_SHOT_NO_NAME_NEUTRO),
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
    lines.append("# Avaliação de NOME PACIENTE — GreetingAgent")
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
        "Para cada caso, marque YES se o aspecto **NOME PACIENTE** estiver correto:"
    )
    lines.append("")
    lines.append("- **expected = usa_nome** → o `response` DEVE conter `patient_name`.")
    lines.append("- **expected = nao_usa_nome** → o `response` NÃO DEVE conter `patient_name` (vindo de input ou histórico).")
    lines.append("- **expected = nao_inventa_nome** → o `response` NÃO DEVE conter nome literal do few_shot (Camila, Lucas).")
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
    print(f"GreetingAgent — NOME PACIENTE eval ({len(CASES)} cases)")
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
        print(f"  PT_NAME:   {kw.get('patient_name')!r}")
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
    default_path = os.path.join(folder, f"nome paciente v26{temp_suffix} - {round_label} ({date_tag}).md")
    out_path = os.environ.get("EVAL_REPORT_PATH", default_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(CASES, latencies))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    run_eval()
