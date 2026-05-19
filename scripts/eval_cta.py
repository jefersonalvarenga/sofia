"""
Evaluation harness focused exclusively on the CTA dimension.

Tests whether the GreetingAgent correctly applies the rule:

    CTAs presentes no few_shot só devem ser utilizados quando
    patient_intents estiver vazio.

    Quando houver intenção em patient_intents:
    - nunca inclua CTA
    - preserve a estrutura, o tom e o padrão de apresentação demonstrados no few_shot

The matrix has 4 quadrants (2 cases each = 8 cases total):

  Q1 — NO_INTENT + few_shot WITH CTA       -> DEVE incluir CTA
  Q2 — NO_INTENT + few_shot WITHOUT CTA    -> NÃO inclui CTA
  Q3 — WITH_INTENT + few_shot WITH CTA     -> REMOVE CTA mas preserva apresentação
  Q4 — WITH_INTENT + few_shot WITHOUT CTA  -> NÃO inclui CTA

Verdict criterion (YES per case):
  - Q1: response MUST contain a CTA token (e.g. "como posso ajudar").
  - Q2/Q4: response MUST NOT contain a CTA token.
  - Q3: response MUST NOT contain CTA AND MUST preserve the presentation pattern
        demonstrated in few_shot (when few_shot has one).

All other dimensions (presentation, name, cordiality, period) are out of scope
for this evaluation — judge only the CTA aspect, with the single exception
that Q3 also checks presentation preservation (since that is part of the CTA rule).

Usage:
    cd easyscale-sofia
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_cta.py
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

# WITH CTA — explicit commercial/operational question at the end.
FEW_SHOT_WITH_CTA_ACOLHEDOR = "Olá! Aqui é da Lumina Estética. Como posso te ajudar hoje?"
FEW_SHOT_WITH_CTA_INFORMAL = "E aí! Aqui é do Studio Bem-Estar, em que posso te ajudar?"

# WITHOUT CTA — pure greeting + presentation, no operational question.
FEW_SHOT_NO_CTA_FORMAL = "Olá, seja bem-vindo à Clínica Vita Premium."
FEW_SHOT_NO_CTA_NEUTRO = "Oi! Aqui é da Clínica Bella."


# ============================================================================
# Test matrix — 4 quadrants × 2 cases = 8 cases
# ============================================================================

CASES: List[Dict[str, Any]] = [
    # ----- Q1: intents=[] + few_shot COM CTA -> DEVE incluir CTA
    {
        "id": "Q1.1",
        "quadrant": "intents=[] + few_shot COM CTA -> DEVE incluir CTA",
        "label": "Q1.1 — Lumina acolhedor com CTA, 'oi', sem intent",
        "expected": "inclui_cta",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_WITH_CTA_ACOLHEDOR,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q1.2",
        "quadrant": "intents=[] + few_shot COM CTA -> DEVE incluir CTA",
        "label": "Q1.2 — Studio Bem-Estar informal com CTA, 'oi', sem intent",
        "expected": "inclui_cta",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Studio Bem-Estar",
            "assistant_name": "Maya",
            "few_shot": FEW_SHOT_WITH_CTA_INFORMAL,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q2: intents=[] + few_shot SEM CTA -> NÃO inclui CTA
    {
        "id": "Q2.1",
        "quadrant": "intents=[] + few_shot SEM CTA -> NÃO inclui CTA",
        "label": "Q2.1 — Vita Premium formal sem CTA, 'oi', sem intent",
        "expected": "nao_inclui_cta",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "few_shot": FEW_SHOT_NO_CTA_FORMAL,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q2.2",
        "quadrant": "intents=[] + few_shot SEM CTA -> NÃO inclui CTA",
        "label": "Q2.2 — Clínica Bella neutro sem CTA, 'oi', sem intent",
        "expected": "nao_inclui_cta",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Bella",
            "assistant_name": "Sofia",
            "few_shot": FEW_SHOT_NO_CTA_NEUTRO,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q3: intents=[X] + few_shot COM CTA -> REMOVE CTA mas preserva apresentação
    {
        "id": "Q3.1",
        "quadrant": "intents=[X] + few_shot COM CTA -> REMOVE CTA mas preserva apresentação",
        "label": "Q3.1 — Lumina com CTA, paciente pergunta sobre peeling (TOPIC_KNOWLEDGE)",
        "expected": "remove_cta_preserva_apresentacao",
        "kwargs": {
            "patient_message": "vcs fazem peeling?",
            "patient_intents": ["TOPIC_KNOWLEDGE"],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_WITH_CTA_ACOLHEDOR,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q3.2",
        "quadrant": "intents=[X] + few_shot COM CTA -> REMOVE CTA mas preserva apresentação",
        "label": "Q3.2 — Studio Bem-Estar com CTA, paciente quer agendar (SCHEDULE)",
        "expected": "remove_cta_preserva_apresentacao",
        "kwargs": {
            "patient_message": "queria marcar uma limpeza de pele",
            "patient_intents": ["SCHEDULE"],
            "patient_name": None,
            "clinic_name": "Studio Bem-Estar",
            "assistant_name": "Maya",
            "few_shot": FEW_SHOT_WITH_CTA_INFORMAL,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q4: intents=[X] + few_shot SEM CTA -> NÃO inclui CTA
    {
        "id": "Q4.1",
        "quadrant": "intents=[X] + few_shot SEM CTA -> NÃO inclui CTA",
        "label": "Q4.1 — Vita Premium sem CTA, paciente pergunta preço (PRICE)",
        "expected": "nao_inclui_cta",
        "kwargs": {
            "patient_message": "quanto custa o botox?",
            "patient_intents": ["PRICE"],
            "patient_name": None,
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "few_shot": FEW_SHOT_NO_CTA_FORMAL,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q4.2",
        "quadrant": "intents=[X] + few_shot SEM CTA -> NÃO inclui CTA",
        "label": "Q4.2 — Clínica Bella sem CTA, paciente quer informações (TOPIC_KNOWLEDGE)",
        "expected": "nao_inclui_cta",
        "kwargs": {
            "patient_message": "queria saber mais sobre os tratamentos",
            "patient_intents": ["TOPIC_KNOWLEDGE"],
            "patient_name": None,
            "clinic_name": "Clínica Bella",
            "assistant_name": "Sofia",
            "few_shot": FEW_SHOT_NO_CTA_NEUTRO,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
]


# ============================================================================
# Heuristic auto-verdict (best-effort hint; humano decide o final)
# ============================================================================

# CTA patterns — commercial/operational questions at the end of a message.
# Spec: "Como posso ajudar?", "Em que posso ajudar?", "No que posso ajudar?",
# qualquer pergunta comercial/operacional.
# Reciprocity ("e você?", "como vai?") is NOT a CTA.
_CTA_PATTERNS = [
    r"\bcomo posso (?:te )?ajudar\b",
    r"\bem que posso (?:te )?ajudar\b",
    r"\bem que posso ser útil\b",
    r"\bno que posso (?:te )?ajudar\b",
    r"\bposso (?:te )?ajudar\b",
    r"\bcomo posso ser útil\b",
]

# Presentation patterns — same as eval_apresentacao.py.
_PRESENTATION_PATTERNS = [
    r"\baqui é d[aoe]\b",
    r"\baqui é\s+\w+\s+d[aoe]\b",
    r"\bsou\s+\w+\s+d[aoe]\b",
    r"\bseja bem-vind[oa]\b",
]


def detect_cta(text: str) -> bool:
    """Best-effort detection of a CTA token in the response."""
    if not text:
        return False
    lowered = text.lower()
    for pat in _CTA_PATTERNS:
        if re.search(pat, lowered):
            return True
    return False


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
    has_cta = detect_cta(response)
    has_presentation = detect_presentation(response)
    expected = case["expected"]

    if expected == "inclui_cta":
        return "YES" if has_cta else "NO"

    if expected == "nao_inclui_cta":
        return "NO" if has_cta else "YES"

    if expected == "remove_cta_preserva_apresentacao":
        # Few_shot has presentation (Q3 uses few-shots WITH CTA which are
        # also WITH presentation). Response must NOT have CTA but MUST have
        # presentation.
        if has_cta:
            return "NO"
        few_shot_has_presentation = detect_presentation(
            case["kwargs"].get("few_shot", "")
        )
        if few_shot_has_presentation and not has_presentation:
            return "NO"
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
        ("FEW_SHOT_WITH_CTA_ACOLHEDOR (Lumina, com CTA)", FEW_SHOT_WITH_CTA_ACOLHEDOR),
        ("FEW_SHOT_WITH_CTA_INFORMAL (Studio Bem-Estar, com CTA)", FEW_SHOT_WITH_CTA_INFORMAL),
        ("FEW_SHOT_NO_CTA_FORMAL (Vita Premium, sem CTA)", FEW_SHOT_NO_CTA_FORMAL),
        ("FEW_SHOT_NO_CTA_NEUTRO (Clínica Bella, sem CTA)", FEW_SHOT_NO_CTA_NEUTRO),
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
    lines.append("# Avaliação de CTA — GreetingAgent")
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
        "Para cada caso, marque YES se o aspecto **CTA** estiver correto:"
    )
    lines.append("")
    lines.append("- **expected = inclui_cta** → o `response` DEVE conter token de CTA (ex: \"como posso ajudar\", \"em que posso ser útil\").")
    lines.append("- **expected = nao_inclui_cta** → o `response` NÃO DEVE conter token de CTA.")
    lines.append("- **expected = remove_cta_preserva_apresentacao** → o `response` NÃO DEVE conter CTA, MAS DEVE manter o padrão de apresentação demonstrado no few_shot.")
    lines.append("")
    lines.append(
        "Outros aspectos (nome do paciente, cordialidade, período do dia) **estão fora do escopo deste round**."
    )
    lines.append("")
    lines.append("Há um **veredito automático heurístico** sugerido por caso. Confirme ou ajuste manualmente.")
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
    lines.append("(Sujeito a confirmação humana — heurística pode classificar mal saídas ambíguas.)")
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
    print(f"GreetingAgent — CTA eval ({len(CASES)} cases)")
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
        print(f"  INPUT:     {kw['patient_message']!r}")
        print(f"  INTENTS:   {kw.get('patient_intents') or []}")
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
    default_path = os.path.join(folder, f"cta v26{temp_suffix} - {round_label} ({date_tag}).md")
    out_path = os.environ.get("EVAL_REPORT_PATH", default_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(CASES, latencies))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    run_eval()
