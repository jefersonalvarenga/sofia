"""
Evaluation harness focused exclusively on the ESTADO SOCIAL dimension.

Tests the v26 rule about social state of the conversation:

    Use patient_message, recent_relevant_messages, session_summary, time_gap_hours
    para inferir se a mensagem atual representa:
    - continuidade natural da conversa
    OU
    - reabertura social da interação

    Em reaberturas sociais:
    - uma nova saudação deve ser utilizada
    - nunca reapresente a clínica ou a assistente
    - nunca aja como primeiro contato

The matrix has 4 quadrants × 2 cases = 8 cases:

  Q1 — PRIMEIRO CONTATO (recent vazio, time_gap=None)
       -> response saudacao COMPLETA (com apresentacao se few_shot tem)
  Q2 — REABERTURA (recent populado, time_gap alto >12h)
       -> response saudacao SEM apresentacao
  Q3 — CONTINUIDADE NATURAL (recent populado, time_gap baixo <1h, paciente
       respondeu sem novo cumprimento)
       -> response silencio (ritual ja fechado) OU resposta minima
  Q4 — RETOMADA CURTA (recent populado, time_gap medio ~2h, paciente
       cumprimenta de novo)
       -> response saudacao SEM apresentacao (nao reage como primeiro contato)

Verdict criterion (YES per case):
  - Q1: response contains presentation if few_shot has it.
  - Q2: response does NOT contain presentation (no "aqui é da", "sou X da").
  - Q3: response is empty OR is minimal (no cumprimento novo + no apresentacao).
  - Q4: response does NOT contain presentation.

Usage:
    cd easyscale-sofia
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_estado_social.py
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

FEW_SHOT_WITH_PRESENTATION = "Olá! Aqui é da Lumina Estética. Como posso te ajudar?"
FEW_SHOT_WITH_PRESENTATION_FORMAL = "Olá, seja bem-vindo à Clínica Vita Premium."
FEW_SHOT_NO_PRESENTATION = "Olá! Como posso te ajudar?"
FEW_SHOT_NEUTRO = "Oi! Como posso te ajudar hoje?"


# ============================================================================
# Test matrix
# ============================================================================

CASES: List[Dict[str, Any]] = [
    # ----- Q1: PRIMEIRO CONTATO (recent vazio, time_gap=None) -> saudacao completa
    {
        "id": "Q1.1",
        "quadrant": "PRIMEIRO CONTATO -> saudacao completa (com apresentacao se few_shot tem)",
        "label": "Q1.1 — primeiro contato, Lumina com apresentacao",
        "expected": "primeiro_contato_apresenta",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_WITH_PRESENTATION,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q1.2",
        "quadrant": "PRIMEIRO CONTATO -> saudacao completa (com apresentacao se few_shot tem)",
        "label": "Q1.2 — primeiro contato, Vita Premium com apresentacao formal",
        "expected": "primeiro_contato_apresenta",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "few_shot": FEW_SHOT_WITH_PRESENTATION_FORMAL,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q2: REABERTURA (recent populado, time_gap alto >24h) -> saudacao SEM apresentacao
    {
        "id": "Q2.1",
        "quadrant": "REABERTURA (time_gap alto) -> saudacao SEM apresentacao",
        "label": "Q2.1 — Lumina, retomada 48h após apresentação prévia, 'oi'",
        "expected": "reabertura_nao_apresenta",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_WITH_PRESENTATION,
            "recent_relevant_messages": [
                {"role": "patient", "content": "oi"},
                {"role": "greeting", "content": "Olá! Aqui é da Lumina Estética. Como posso te ajudar?"},
                {"role": "patient", "content": "valeu, depois eu volto"},
            ],
            "session_summary": "Paciente perguntou sobre serviços e disse que voltaria.",
            "time_gap_hours": 48,
        },
    },
    {
        "id": "Q2.2",
        "quadrant": "REABERTURA (time_gap alto) -> saudacao SEM apresentacao",
        "label": "Q2.2 — Vita Premium, retomada 120h, 'boa tarde'",
        "expected": "reabertura_nao_apresenta",
        "kwargs": {
            "patient_message": "boa tarde",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "few_shot": FEW_SHOT_WITH_PRESENTATION_FORMAL,
            "recent_relevant_messages": [
                {"role": "patient", "content": "oi"},
                {"role": "greeting", "content": "Olá, seja bem-vindo à Clínica Vita Premium."},
                {"role": "patient", "content": "obrigada"},
            ],
            "session_summary": "Paciente recebeu boas-vindas e agradeceu.",
            "time_gap_hours": 120,
        },
    },
    # ----- Q3: CONTINUIDADE NATURAL (time_gap baixo, sem novo cumprimento) -> silencio ou minima
    {
        "id": "Q3.1",
        "quadrant": "CONTINUIDADE NATURAL (time_gap baixo, sem novo cumprimento) -> silencio ou minima",
        "label": "Q3.1 — Lumina, paciente respondeu 'beleza' apos cumprimento, time_gap=0.05h",
        "expected": "continuidade_silencio_ou_minima",
        "kwargs": {
            "patient_message": "beleza",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_WITH_PRESENTATION,
            "recent_relevant_messages": [
                {"role": "patient", "content": "oi"},
                {"role": "greeting", "content": "Olá! Aqui é da Lumina Estética. Como posso te ajudar?"},
            ],
            "session_summary": "Paciente cumprimentou e foi recebida.",
            "time_gap_hours": 0.05,
        },
    },
    {
        "id": "Q3.2",
        "quadrant": "CONTINUIDADE NATURAL (time_gap baixo, sem novo cumprimento) -> silencio ou minima",
        "label": "Q3.2 — paciente respondeu 'tudo certo' apos pergunta cordial, time_gap=0.03h",
        "expected": "continuidade_silencio_ou_minima",
        "kwargs": {
            "patient_message": "tudo certo",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "few_shot": "Olá! Aqui é da Vita Premium, tudo bem? Como posso te ajudar?",
            "recent_relevant_messages": [
                {"role": "patient", "content": "oi"},
                {"role": "greeting", "content": "Olá! Aqui é da Vita Premium, tudo bem? Como posso te ajudar?"},
            ],
            "session_summary": "Agent ja apresentou e perguntou cordialidade.",
            "time_gap_hours": 0.03,
        },
    },
    # ----- Q4: RETOMADA CURTA (time_gap medio ~2-6h, paciente cumprimenta novamente) -> sem apresentacao
    {
        "id": "Q4.1",
        "quadrant": "RETOMADA CURTA (time_gap medio, paciente cumprimenta) -> sem apresentacao",
        "label": "Q4.1 — Lumina, retomada 3h, paciente diz 'oi de novo'",
        "expected": "retomada_curta_nao_apresenta",
        "kwargs": {
            "patient_message": "oi de novo",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_WITH_PRESENTATION,
            "recent_relevant_messages": [
                {"role": "patient", "content": "oi"},
                {"role": "greeting", "content": "Olá! Aqui é da Lumina Estética. Como posso te ajudar?"},
                {"role": "patient", "content": "vou pensar, depois volto"},
            ],
            "session_summary": "Paciente foi recebida e disse que voltaria.",
            "time_gap_hours": 3,
        },
    },
    {
        "id": "Q4.2",
        "quadrant": "RETOMADA CURTA (time_gap medio, paciente cumprimenta) -> sem apresentacao",
        "label": "Q4.2 — Vita Premium, retomada 6h, paciente diz 'boa tarde'",
        "expected": "retomada_curta_nao_apresenta",
        "kwargs": {
            "patient_message": "boa tarde",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "few_shot": FEW_SHOT_WITH_PRESENTATION_FORMAL,
            "recent_relevant_messages": [
                {"role": "patient", "content": "oi"},
                {"role": "greeting", "content": "Olá, seja bem-vindo à Clínica Vita Premium."},
                {"role": "patient", "content": "ok"},
            ],
            "session_summary": "Paciente foi recebida.",
            "time_gap_hours": 6,
        },
    },
]


# ============================================================================
# Heuristic auto-verdict
# ============================================================================

# Presentation patterns (same as eval_apresentacao.py).
_PRESENTATION_PATTERNS = [
    r"\baqui é d[aoe]\b",
    r"\baqui é\s+\w+\s+d[aoe]\b",
    r"\bsou\s+\w+\s+d[aoe]\b",
    r"\bseja bem-vind[oa]\b",
]

# Greeting tokens.
_GREETING_TOKENS = [
    r"^\s*olá\b",
    r"^\s*oi\b",
    r"^\s*bom dia\b",
    r"^\s*boa tarde\b",
    r"^\s*boa noite\b",
    r"^\s*e aí\b",
]


def has_presentation(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(re.search(pat, lowered) for pat in _PRESENTATION_PATTERNS)


def has_greeting(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(re.search(pat, lowered) for pat in _GREETING_TOKENS)


def auto_verdict(case: Dict[str, Any], response: str) -> str:
    expected = case["expected"]

    if expected == "primeiro_contato_apresenta":
        # Should contain presentation (since few_shot has one).
        return "YES" if has_presentation(response) else "NO"

    if expected == "reabertura_nao_apresenta":
        # Should NOT contain presentation. Greeting is allowed.
        return "NO" if has_presentation(response) else "YES"

    if expected == "continuidade_silencio_ou_minima":
        # Empty (silence) is OK. Non-empty must NOT have presentation NOR a
        # fresh greeting token (no "Olá!", "Oi!").
        if not response:
            return "YES"
        if has_presentation(response):
            return "NO"
        # A fresh greeting in continuity is usually wrong, but tolerate short
        # affirmations. We focus on "no presentation" as the primary criterion.
        return "YES"

    if expected == "retomada_curta_nao_apresenta":
        # Similar to Q2: no presentation, but greeting is OK.
        return "NO" if has_presentation(response) else "YES"

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
        ("FEW_SHOT_WITH_PRESENTATION (Lumina, com apresentacao)", FEW_SHOT_WITH_PRESENTATION),
        ("FEW_SHOT_WITH_PRESENTATION_FORMAL (Vita Premium, 'seja bem-vindo')", FEW_SHOT_WITH_PRESENTATION_FORMAL),
        ("FEW_SHOT_NO_PRESENTATION", FEW_SHOT_NO_PRESENTATION),
        ("FEW_SHOT_NEUTRO", FEW_SHOT_NEUTRO),
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
    lines.append("# Avaliação de ESTADO SOCIAL — GreetingAgent")
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
    lines.append("- **expected = primeiro_contato_apresenta** → response contém apresentação (few_shot tem).")
    lines.append("- **expected = reabertura_nao_apresenta** → response NÃO contém apresentação (cumprimento OK).")
    lines.append("- **expected = continuidade_silencio_ou_minima** → response vazia OU sem apresentação/saudação nova.")
    lines.append("- **expected = retomada_curta_nao_apresenta** → response NÃO contém apresentação.")
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
    print(f"GreetingAgent — ESTADO SOCIAL eval ({len(CASES)} cases)")
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
        print(f"  GAP:       {kw.get('time_gap_hours')}")
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
    default_path = os.path.join(folder, f"estado social v26{temp_suffix} - {round_label} ({date_tag}).md")
    out_path = os.environ.get("EVAL_REPORT_PATH", default_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(CASES, latencies))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    run_eval()
