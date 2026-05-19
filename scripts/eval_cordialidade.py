"""
Evaluation harness focused exclusively on the CORDIALIDADE dimension.

Tests three CORDIALIDADE sub-rules from the v26 prompt:

  1. CORDIALIDADE INICIADA PELO PACIENTE
     Quando paciente pergunta "tudo bem?", agent must:
     - responder afirmativamente
     - devolver a cordialidade espelhando o tom
     Example: "tudo bem?" -> "Tudo bem, e você?"

  2. CORDIALIDADE COMO PADRÃO DA CLÍNICA
     Se few_shot tem cordialidade ("Tudo bem?", "Como vai?"), agent CAN
     initiate cordiality even when patient didn't (in first contact).

  3. NÃO REPETIR (G3.2)
     Quando paciente responde a uma pergunta cordial prévia, agent NÃO deve
     fazer NOVA pergunta cordial — encerra o ritual.
     Example: history has "Tudo bem?" + patient says "td joia kkk e ai?"
     -> response should affirm without re-asking, or "{"response": ""}".

The matrix has 4 quadrants × 2 cases = 8 cases:

  Q1 — patient asks cordiality + few_shot WITH cordiality
       -> reciprocates with affirmation + devolution
  Q2 — patient asks cordiality + few_shot WITHOUT cordiality
       -> still reciprocates (patient-initiated has priority)
  Q3 — patient does NOT ask + few_shot WITH cordiality
       -> agent initiates cordiality (mirroring few_shot)
  Q4 — ritual already happened (G3.2) + patient closing it
       -> agent does NOT re-ask; affirms or stays silent

Verdict criterion (YES per case):
  - Q1/Q2: response contains both affirmation ("tudo bem", "td bem", "vou bem")
           AND devolution ("e você", "e contigo", "e aí").
  - Q3: response contains cordiality question (e.g. "tudo bem?", "como vai?").
  - Q4: response does NOT contain cordiality question (no "tudo bem?", etc).

Usage:
    cd easyscale-sofia
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_cordialidade.py
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

# WITH cordiality (clinic's default pattern includes "tudo bem?" or similar).
FEW_SHOT_WITH_CORDIALITY = "Olá! Aqui é da Lumina Estética, tudo bem? Como posso te ajudar?"
FEW_SHOT_WITH_CORDIALITY_INFORMAL = "E aí, tudo certo? Aqui é do Studio Bem-Estar."

# WITHOUT cordiality (no "tudo bem?", "como vai?").
FEW_SHOT_NO_CORDIALITY_FORMAL = "Olá, seja bem-vindo à Clínica Vita Premium."
FEW_SHOT_NO_CORDIALITY_NEUTRO = "Oi! Aqui é da Clínica Bella. Como posso te ajudar?"


# ============================================================================
# Test matrix — 4 quadrants × 2 cases = 8 cases
# ============================================================================

CASES: List[Dict[str, Any]] = [
    # ----- Q1: paciente pergunta cordialidade + few_shot COM cordialidade
    # -> agent reciprocates (afirma + devolve)
    {
        "id": "Q1.1",
        "quadrant": "paciente pergunta + few_shot COM cordialidade -> reciproca (afirma + devolve)",
        "label": "Q1.1 — 'oi tudo bem?', Lumina com cordialidade",
        "expected": "reciproca_cordialidade",
        "kwargs": {
            "patient_message": "oi tudo bem?",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_WITH_CORDIALITY,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q1.2",
        "quadrant": "paciente pergunta + few_shot COM cordialidade -> reciproca (afirma + devolve)",
        "label": "Q1.2 — 'tudo bem?', Studio Bem-Estar com cordialidade",
        "expected": "reciproca_cordialidade",
        "kwargs": {
            "patient_message": "tudo bem?",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Studio Bem-Estar",
            "assistant_name": "Maya",
            "few_shot": FEW_SHOT_WITH_CORDIALITY_INFORMAL,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q2: paciente pergunta cordialidade + few_shot SEM cordialidade
    # -> still reciprocates (patient-initiated has priority)
    {
        "id": "Q2.1",
        "quadrant": "paciente pergunta + few_shot SEM cordialidade -> reciproca (prioridade do paciente)",
        "label": "Q2.1 — 'oi td bem', Vita Premium sem cordialidade",
        "expected": "reciproca_cordialidade",
        "kwargs": {
            "patient_message": "oi td bem",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "few_shot": FEW_SHOT_NO_CORDIALITY_FORMAL,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q2.2",
        "quadrant": "paciente pergunta + few_shot SEM cordialidade -> reciproca (prioridade do paciente)",
        "label": "Q2.2 — 'oi, como vai?', Clínica Bella sem cordialidade",
        "expected": "reciproca_cordialidade",
        "kwargs": {
            "patient_message": "oi, como vai?",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Clínica Bella",
            "assistant_name": "Sofia",
            "few_shot": FEW_SHOT_NO_CORDIALITY_NEUTRO,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q3: paciente NÃO pergunta + few_shot COM cordialidade
    # -> agent inicia cordialidade (espelhando few_shot)
    {
        "id": "Q3.1",
        "quadrant": "paciente NÃO pergunta + few_shot COM cordialidade -> agent inicia",
        "label": "Q3.1 — 'oi', Lumina com cordialidade no padrão",
        "expected": "inicia_cordialidade",
        "kwargs": {
            "patient_message": "oi",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_WITH_CORDIALITY,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    {
        "id": "Q3.2",
        "quadrant": "paciente NÃO pergunta + few_shot COM cordialidade -> agent inicia",
        "label": "Q3.2 — 'bom dia', Studio Bem-Estar com cordialidade no padrão",
        "expected": "inicia_cordialidade",
        "kwargs": {
            "patient_message": "bom dia",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Studio Bem-Estar",
            "assistant_name": "Maya",
            "few_shot": FEW_SHOT_WITH_CORDIALITY_INFORMAL,
            "recent_relevant_messages": [],
            "session_summary": "",
            "time_gap_hours": None,
        },
    },
    # ----- Q4: ritual já aconteceu (G3.2) + paciente encerrando
    # -> agent NÃO re-pergunta; afirma ou silencia
    {
        "id": "Q4.1",
        "quadrant": "ritual ja aconteceu (G3.2) + paciente encerrando -> nao re-pergunta",
        "label": "Q4.1 — agent ja perguntou 'tudo bem?', paciente diz 'td joia kkk e ai?'",
        "expected": "encerra_ritual",
        "kwargs": {
            "patient_message": "td joia kkk e ai?",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "few_shot": FEW_SHOT_WITH_CORDIALITY,
            "recent_relevant_messages": [
                {"role": "patient", "content": "oi"},
                {
                    "role": "greeting",
                    "content": "Olá! Aqui é da Lumina Estética, tudo bem? Como posso te ajudar?",
                },
            ],
            "session_summary": "Paciente iniciou contato, agent já perguntou tudo bem.",
            "time_gap_hours": 0.05,
        },
    },
    {
        "id": "Q4.2",
        "quadrant": "ritual ja aconteceu (G3.2) + paciente encerrando -> nao re-pergunta",
        "label": "Q4.2 — paciente respondeu cordialidade previa, agora apenas 'tudo bem e você?'",
        "expected": "encerra_ritual",
        "kwargs": {
            "patient_message": "tudo bem e você?",
            "patient_intents": [],
            "patient_name": None,
            "clinic_name": "Studio Bem-Estar",
            "assistant_name": "Maya",
            "few_shot": FEW_SHOT_WITH_CORDIALITY_INFORMAL,
            "recent_relevant_messages": [
                {"role": "patient", "content": "oi"},
                {
                    "role": "greeting",
                    "content": "E aí, tudo certo? Aqui é do Studio Bem-Estar.",
                },
            ],
            "session_summary": "Agent já cumprimentou e perguntou tudo certo.",
            "time_gap_hours": 0.02,
        },
    },
]


# ============================================================================
# Heuristic auto-verdict (best-effort hint; humano decide o final)
# ============================================================================

# Affirmation patterns ("tudo bem", "td bem", "vou bem", "tudo certo").
_AFFIRMATION_PATTERNS = [
    r"\btudo bem\b",
    r"\btd bem\b",
    r"\bvou bem\b",
    r"\bestou bem\b",
    r"\btô bem\b",
    r"\btudo certo\b",
    r"\btudo ót[ií]m[oa]\b",
    r"\btudo joia\b",
    r"\btudo tranquilo\b",
]

# Devolution patterns ("e você?", "e contigo?", "e aí?").
_DEVOLUTION_PATTERNS = [
    r"\be (?:com\s+)?você\??",
    r"\be contigo\??",
    r"\be a[íi]\??",
    r"\be o senhor\??",
    r"\be a senhora\??",
]

# Cordiality question patterns ("tudo bem?", "como vai?", "tudo certo?").
_CORDIALITY_QUESTION_PATTERNS = [
    r"\btudo bem\??",
    r"\btd bem\??",
    r"\bcomo vai\??",
    r"\btudo certo\??",
    r"\bcomo você está\??",
    r"\btudo ót[ií]m[oa]\??",
]


def _matches_any(text: str, patterns: List[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    for pat in patterns:
        if re.search(pat, lowered):
            return True
    return False


def has_affirmation(text: str) -> bool:
    return _matches_any(text, _AFFIRMATION_PATTERNS)


def has_devolution(text: str) -> bool:
    return _matches_any(text, _DEVOLUTION_PATTERNS)


def has_cordiality_question(text: str) -> bool:
    """Detect a fresh cordiality question (assumes text is the agent response)."""
    if not text:
        return False
    # Strict: must end with cordiality token AND a '?' nearby OR be a question
    # form. Simple regex enough for our short outputs.
    lowered = text.lower()
    for pat in _CORDIALITY_QUESTION_PATTERNS:
        if re.search(pat, lowered):
            # Ensure it's NOT just affirmation form like "tudo bem" without '?'.
            # If the response is "Tudo bem, e você?", that ALSO matches "tudo bem"
            # but the cordiality question is the "e você?" part (devolution).
            # So we want to detect: "tudo bem?", "como vai?", "tudo certo?" — i.e.
            # forms where the cordiality token itself ends with ? in the text.
            # Approximate: check if "tudo bem?" or similar appears.
            return True
    return False


def auto_verdict(case: Dict[str, Any], response: str) -> str:
    """Return 'YES' or 'NO' as a heuristic hint."""
    expected = case["expected"]

    if expected == "reciproca_cordialidade":
        # Must have affirmation AND devolution.
        return "YES" if (has_affirmation(response) and has_devolution(response)) else "NO"

    if expected == "inicia_cordialidade":
        # Must have a cordiality question.
        return "YES" if has_cordiality_question(response) else "NO"

    if expected == "encerra_ritual":
        # Must NOT have a fresh cordiality question (no re-asking).
        # Affirmation is OK; silence (empty response) is OK.
        if not response:
            return "YES"
        # Look for cordiality QUESTIONS (with ?). Affirmation without ? is fine.
        lowered = response.lower()
        # Match "tudo bem?" but not "tudo bem!" or "tudo bem,".
        for pat in [
            r"\btudo bem\?",
            r"\btd bem\?",
            r"\bcomo vai\?",
            r"\btudo certo\?",
            r"\bcomo você está\?",
        ]:
            if re.search(pat, lowered):
                return "NO"
        return "YES"

    return "CHECK"


# ============================================================================
# Report renderer (identical structure to eval_cta.py)
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
        ("FEW_SHOT_WITH_CORDIALITY (Lumina, com 'tudo bem?')", FEW_SHOT_WITH_CORDIALITY),
        ("FEW_SHOT_WITH_CORDIALITY_INFORMAL (Studio, com 'tudo certo?')", FEW_SHOT_WITH_CORDIALITY_INFORMAL),
        ("FEW_SHOT_NO_CORDIALITY_FORMAL (Vita Premium)", FEW_SHOT_NO_CORDIALITY_FORMAL),
        ("FEW_SHOT_NO_CORDIALITY_NEUTRO (Clínica Bella)", FEW_SHOT_NO_CORDIALITY_NEUTRO),
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
    lines.append("# Avaliação de CORDIALIDADE — GreetingAgent")
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
        "Para cada caso, marque YES se o aspecto **CORDIALIDADE** estiver correto:"
    )
    lines.append("")
    lines.append("- **expected = reciproca_cordialidade** → o `response` DEVE conter afirmação (\"tudo bem\") + devolução (\"e você?\").")
    lines.append("- **expected = inicia_cordialidade** → o `response` DEVE conter pergunta cordial (\"tudo bem?\", \"como vai?\").")
    lines.append("- **expected = encerra_ritual** → o `response` NÃO DEVE conter pergunta cordial fresca (ritual G3.2).")
    lines.append("")
    lines.append("Outras dimensões fora de escopo. Há veredito automático heurístico — confirme manualmente.")
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
    print(f"GreetingAgent — CORDIALIDADE eval ({len(CASES)} cases)")
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
    default_path = os.path.join(folder, f"cordialidade v26{temp_suffix} - {round_label} ({date_tag}).md")
    out_path = os.environ.get("EVAL_REPORT_PATH", default_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(CASES, latencies))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    run_eval()
