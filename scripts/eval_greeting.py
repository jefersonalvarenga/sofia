"""
Manual evaluation harness for the GreetingAgent v15.

Spec: kb/07-MVP/Tech/03-Discussoes/03 - Greeting Agent Spec v0.11.md

Runs realistic scenarios against gpt-4o-mini and writes a versioned
markdown report into the Obsidian vault ("Agente Greeting" folder).
Target: >= 85% YES (natural, faithful to few-shots, respecting R1-R4).

v11 differences vs v10:
  - No silence path. Agent always responds when invoked.
  - Cases reorganized around 4 collapsed rules (R1-R4).
  - Same patient message tested with gap=-1 and gap>0 to verify R4
    (apresentação só em primeiro contato).

Usage:
    cd easyscale-sofia
    PYTHONPATH=. python scripts/eval_greeting.py
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_greeting.py
"""

from __future__ import annotations

import os
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

from app.agents.greeting.agent import GreetingAgent, _build_user_prompt, _coerce_few_shots, _normalize_contact_name  # noqa: E402


FEW_SHOTS_ACOLHEDOR = [
    "Olá! Aqui é da Lumina Estética. Como posso te ajudar hoje?",
    "Oi! 😊 Aqui é da Lumina. Em que posso ajudar?",
    "Olá, tudo bem? Aqui é da Lumina, como posso te ajudar?",
]

FEW_SHOTS_SEMIFORMAL = [
    "Olá, seja bem-vindo à Clínica Vita Premium. Como podemos te ajudar?",
    "Bom dia, aqui é da Vita Premium. Em que posso ser útil?",
    "Boa tarde, sou Helena da Vita Premium. Como posso atender você?",
]

FEW_SHOTS_INFORMAL = [
    "E aí! Aqui é do Studio Bem-Estar, em que posso te ajudar?",
    "Oiê! Tudo bem? Aqui é do Studio, no que posso ajudar?",
    "Opa! Aqui é da Iris do Studio, manda aí o que precisa",
]


# ===== Casos v12: history-based em vez de gap =====
# Helper para tornar histórias mais legíveis
def _h_empty() -> List[Dict[str, str]]:
    return []


def _h_bot_greeted(bot_msg: str, patient_reply: Optional[str] = None) -> List[Dict[str, str]]:
    """Histórico: paciente abriu, greeting já saudou. Opcionalmente paciente respondeu."""
    h = [
        {"role": "patient", "content": "oi"},
        {"role": "greeting", "content": bot_msg},
    ]
    if patient_reply:
        h.append({"role": "patient", "content": patient_reply})
    return h


# G1: SEM INTENÇÃO — espera saudação + CTA leve (R2)
SEM_INTENCAO: List[Dict[str, Any]] = [
    {
        "label": "G1.1 - 'oi' + primeiro contato (acolhedor)",
        "kwargs": {
            "latest_incoming": "oi",
            "contact_name": "Camila",
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "initial_greetings": FEW_SHOTS_ACOLHEDOR,
            "history": _h_empty(),
        },
    },
    {
        "label": "G1.2 - 'oi' + retomada após pausa (acolhedor, SEM apresentação)",
        "kwargs": {
            "latest_incoming": "oi",
            "contact_name": "Camila",
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "initial_greetings": FEW_SHOTS_ACOLHEDOR,
            "period_day": "afternoon",
            # Cenário plausível: paciente fez consulta breve dias atrás
            # (preço de tratamento), agora retoma com novo "oi"
            "history": [
                {"role": "patient", "content": "oi, qual o valor do botox?"},
                {
                    "role": "greeting",
                    "content": "Olá Camila! Aqui é da Lumina Estética. O botox parte de R$ 1.200 a região.",
                },
                {"role": "patient", "content": "obrigada, vou pensar"},
            ],
            "time_gap_hours": 48,
        },
    },
    {
        "label": "G1.3 - 'bom dia' + primeiro contato (semiformal)",
        "kwargs": {
            "latest_incoming": "bom dia",
            "contact_name": "Pedro",
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "initial_greetings": FEW_SHOTS_SEMIFORMAL,
            "period_day": "morning",
            "history": _h_empty(),
        },
    },
    {
        "label": "G1.4 - 'tudo bem?' + primeiro contato (informal, Conv 1)",
        "kwargs": {
            "latest_incoming": "tudo bem?",
            "contact_name": "Lucas",
            "clinic_name": "Studio Bem-Estar",
            "assistant_name": "Iris",
            "initial_greetings": FEW_SHOTS_INFORMAL,
            "history": _h_empty(),
        },
    },
    {
        "label": "G1.5 - 'boa tarde tudo bem?' sem nome + primeiro contato",
        "kwargs": {
            "latest_incoming": "boa tarde, tudo bem?",
            "contact_name": None,
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "initial_greetings": FEW_SHOTS_ACOLHEDOR,
            "period_day": "afternoon",
            "history": _h_empty(),
        },
    },
    {
        "label": "G1.6 - 'tudo bem' SEM '?' + primeiro contato (paciente seco)",
        "kwargs": {
            "latest_incoming": "tudo bem",
            "contact_name": "Marcela",
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "initial_greetings": FEW_SHOTS_SEMIFORMAL,
            "history": _h_empty(),
        },
    },
]

# G2: COM INTENÇÃO — espera saudação SEM CTA, sem mencionar tópico (R3)
COM_INTENCAO: List[Dict[str, Any]] = [
    {
        "label": "G2.1 - 'oi, queria saber preço de botox' + primeiro contato",
        "kwargs": {
            "latest_incoming": "oi, queria saber preço de botox",
            "contact_name": "Camila",
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "initial_greetings": FEW_SHOTS_ACOLHEDOR,
            "history": _h_empty(),
            "patient_intents": ["BUSINESS_INFO"],
        },
    },
    {
        "label": "G2.2 - 'bom dia, vcs atendem sábado?' + primeiro contato",
        "kwargs": {
            "latest_incoming": "bom dia, vcs atendem sábado?",
            "contact_name": "Pedro",
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "initial_greetings": FEW_SHOTS_SEMIFORMAL,
            "period_day": "morning",
            "history": _h_empty(),
            "patient_intents": ["BUSINESS_INFO"],
        },
    },
    {
        "label": "G2.3 - 'eae bora marcar massagem?' + primeiro contato (informal)",
        "kwargs": {
            "latest_incoming": "eae bora marcar uma massagem?",
            "contact_name": "Lucas",
            "clinic_name": "Studio Bem-Estar",
            "assistant_name": "Iris",
            "initial_greetings": FEW_SHOTS_INFORMAL,
            "history": _h_empty(),
            "patient_intents": ["SCHEDULE"],
        },
    },
    {
        "label": "G2.4 - 'queria cancelar' + primeiro contato",
        "kwargs": {
            "latest_incoming": "queria cancelar minha consulta",
            "contact_name": "Marcela",
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "initial_greetings": FEW_SHOTS_ACOLHEDOR,
            "history": _h_empty(),
            "patient_intents": ["SCHEDULE"],
        },
    },
    {
        "label": "G2.5 - 'queria cancelar' + retomada (SEM apresentação)",
        "kwargs": {
            "latest_incoming": "queria cancelar minha consulta",
            "contact_name": "Marcela",
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "initial_greetings": FEW_SHOTS_ACOLHEDOR,
            "period_day": "evening",
            "history": _h_bot_greeted(
                "Olá Marcela! Aqui é da Lumina Estética. Como posso te ajudar?",
            ),
            "patient_intents": ["SCHEDULE"],
            "time_gap_hours": 50,
        },
    },
    {
        "label": "G2.6 - 'vcs fazem peeling químico?' + primeiro contato",
        "kwargs": {
            "latest_incoming": "vcs fazem peeling químico?",
            "contact_name": "Patrícia",
            "clinic_name": "Studio Bem-Estar",
            "assistant_name": "Iris",
            "initial_greetings": FEW_SHOTS_INFORMAL,
            "history": _h_empty(),
            "patient_intents": ["TOPIC_KNOWLEDGE"],
        },
    },
    {
        "label": "G2.7 - 'preciso de consulta urgente' + primeiro contato",
        "kwargs": {
            "latest_incoming": "preciso de uma consulta urgente amanhã",
            "contact_name": "Roberto",
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "initial_greetings": FEW_SHOTS_SEMIFORMAL,
            "history": _h_empty(),
            "patient_intents": ["SCHEDULE"],
        },
    },
]

# G3: RECIPROCIDADE — paciente devolve cordialidade (Conv 1)
RECIPROCIDADE: List[Dict[str, Any]] = [
    {
        "label": "G3.1 - 'tudo bem e você?' depois de bot ter perguntado",
        "kwargs": {
            "latest_incoming": "tudo bem e você?",
            "contact_name": "Camila",
            "clinic_name": "Lumina Estética",
            "assistant_name": "Iris",
            "initial_greetings": FEW_SHOTS_ACOLHEDOR,
            # Bot abriu com saudação + pergunta de cordialidade (sem
            # auto-responder); paciente devolve com pergunta espelhada.
            "history": [
                {"role": "patient", "content": "oi"},
                {
                    "role": "greeting",
                    "content": "Olá Camila! Aqui é da Lumina Estética, tudo bem?",
                },
            ],
        },
    },
    {
        "label": "G3.2 - 'td joia kkk e ai?' depois de bot ter perguntado (informal)",
        "kwargs": {
            "latest_incoming": "td joia kkk e ai?",
            "contact_name": "Lucas",
            "clinic_name": "Studio Bem-Estar",
            "assistant_name": "Iris",
            "initial_greetings": FEW_SHOTS_INFORMAL,
            "history": [
                {"role": "patient", "content": "eae"},
                {
                    "role": "greeting",
                    "content": "Oiê, Lucas! Tudo bem? Aqui é da Iris do Studio Bem-Estar.",
                },
            ],
        },
    },
    {
        "label": "G3.3 - 'estou bem e a senhora?' (semiformal, ritual fecha)",
        "kwargs": {
            "latest_incoming": "estou bem, e a senhora?",
            "contact_name": "Beatriz",
            "clinic_name": "Clínica Vita Premium",
            "assistant_name": "Helena",
            "initial_greetings": FEW_SHOTS_SEMIFORMAL,
            "history": [
                {"role": "patient", "content": "boa tarde"},
                {
                    "role": "greeting",
                    "content": "Boa tarde, Beatriz. Tudo bem por aqui, e a senhora? Aqui é a Helena da Clínica Vita Premium.",
                },
            ],
        },
    },
]


def _render_config_block() -> List[str]:
    """Render full agent configuration into the report header for traceability."""
    # Late import to avoid circulars at module load
    from app.agents.greeting.agent import (
        GREETING_MAX_TOKENS,
        GREETING_MODEL,
        GREETING_TEMPERATURE,
        SYSTEM_PROMPT,
        TECHNICAL_FALLBACK,
        _build_user_prompt,
    )

    lines: List[str] = []
    lines.append("## Configuração do agente (no momento da rodada)")
    lines.append("")
    lines.append(f"- **Modelo:** `{GREETING_MODEL}`")
    lines.append(f"- **Temperature:** `{GREETING_TEMPERATURE}`")
    lines.append(f"- **max_tokens:** `{GREETING_MAX_TOKENS}`")
    lines.append(f"- **Fallback técnico:** `{TECHNICAL_FALLBACK!r}`")
    lines.append(f"- **Cache:** desabilitado")
    lines.append("")
    lines.append("### Few-shots usados nesta rodada")
    lines.append("")
    lines.append("**FEW_SHOTS_ACOLHEDOR** (Lumina Estética):")
    lines.append("```")
    for ex in FEW_SHOTS_ACOLHEDOR:
        lines.append(f"- {ex}")
    lines.append("```")
    lines.append("")
    lines.append("**FEW_SHOTS_SEMIFORMAL** (Clínica Vita Premium):")
    lines.append("```")
    for ex in FEW_SHOTS_SEMIFORMAL:
        lines.append(f"- {ex}")
    lines.append("```")
    lines.append("")
    lines.append("**FEW_SHOTS_INFORMAL** (Studio Bem-Estar):")
    lines.append("```")
    for ex in FEW_SHOTS_INFORMAL:
        lines.append(f"- {ex}")
    lines.append("```")
    lines.append("")
    lines.append("### System prompt")
    lines.append("")
    lines.append("```")
    lines.append(SYSTEM_PROMPT)
    lines.append("```")
    lines.append("")
    lines.append("### User prompt — template (placeholders preenchidos por caso)")
    lines.append("")
    example_user = _build_user_prompt(
        patient_message="<PATIENT_MESSAGE>",
        patient_intents=["<INTENT>"],
        patient_name="<PATIENT_NAME>",
        clinic_name="<CLINIC_NAME>",
        assistant_name="<ASSISTANT_NAME>",
        few_shots=["<EXEMPLO 1>", "<EXEMPLO 2>", "<EXEMPLO 3>"],
        session_summary="<SUMMARY>",
        recent_relevant_messages=[],
        time_gap_hours=None,
    )
    lines.append("```")
    lines.append(example_user)
    lines.append("```")
    lines.append("")
    return lines


def _render_markdown(cases: List[Dict[str, Any]], latencies: List[float]) -> str:
    lines: List[str] = []
    lines.append(f"# GreetingAgent v15 — Avaliação manual ({len(cases)} entradas)")
    lines.append("")
    from app.agents.greeting.agent import (
        GREETING_MAX_TOKENS as _MX,
        GREETING_MODEL as _MD,
        GREETING_TEMPERATURE as _TP,
    )
    lines.append(f"- **Modelo:** `{_MD}`")
    lines.append(f"- **Temperature:** `{_TP}`")
    lines.append(f"- **max_tokens:** `{_MX}`")
    lines.append(f"- **Cache:** desabilitado")
    lines.append(
        f"- **Latência:** min={min(latencies):.0f}ms  "
        f"p50={sorted(latencies)[len(latencies)//2]:.0f}ms  "
        f"max={max(latencies):.0f}ms"
    )
    lines.append("")
    lines.append("## Critério (v15)")
    lines.append("")
    lines.append("Para cada caso, marque YES se TODOS:")
    lines.append("")
    lines.append("1. **Fidelidade ao few-shot** — saída parece um dos exemplos da clínica (estrutura, pontuação, tom).")
    lines.append("2. **Apresentação por gap (R4)** — só aparece em `gap == -1`. Em retomadas, omite.")
    lines.append("3. **R1 reciprocidade** — se paciente trouxe cortesia/pergunta, espelha de volta.")
    lines.append("4. **R2/R3 CTA** — só com CTA quando paciente NÃO trouxe intenção (R2). Com intenção, SEM CTA (R3).")
    lines.append("5. **Não menciona o tópico** que o paciente trouxe.")
    lines.append("6. **CTA-safe** — sem `agendar`, `marcar`, `R$`, `horário`, oferta.")
    lines.append("")
    lines.append("**Meta:** ≥ 85% YES. Agent sempre responde — não há `<SILENCE>` em v11.")
    lines.append("")
    lines.append("---")
    lines.append("")
    # Inject full configuration block (model, prompt, few-shots, user template)
    lines.extend(_render_config_block())
    lines.append("---")
    lines.append("")

    sections = [
        ("G1 — sem intenção (R2: padrão + cortesia + CTA leve)", "G1"),
        ("G2 — com intenção (R3: padrão SEM CTA, sem mencionar tópico)", "G2"),
        ("G3 — reciprocidade (R1: espelha cortesia)", "G3"),
    ]

    case_idx = 0
    for section_title, prefix in sections:
        section_cases = [c for c in cases if c["label"].startswith(prefix)]
        lines.append(f"## {section_title}")
        lines.append("")
        for case in section_cases:
            case_idx += 1
            lines.append(f"### [{case_idx:>2}] {case['label']}")
            lines.append("")
            lines.append(f"_Latência: {case['_elapsed_ms']:.0f}ms_ — `{case['_reasoning']}`")
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
                lines.append("> _(vazio — fallback ou erro)_")
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
            lines.append("**Veredito:**")
            lines.append("")
            lines.append("- [ ] YES")
            lines.append("- [ ] NO — motivo: ___")
            lines.append("")
            lines.append("---")
            lines.append("")

    lines.append("## Score final")
    lines.append("")
    lines.append("- G1 sem intenção: ___/6 YES")
    lines.append("- G2 com intenção: ___/7 YES")
    lines.append("- G3 reciprocidade: ___/3 YES")
    lines.append(f"- **Total: ___/{len(cases)} YES**")
    lines.append("")
    lines.append("**Aprovado:** [ ] Sim   [ ] Não (re-rodar)")
    lines.append("")
    return "\n".join(lines)


def run_eval() -> None:
    agent = GreetingAgent()
    # Force-disable cache on the agent's own LM so the eval measures real
    # latency every run (otherwise repeat runs return cached responses in <10ms).
    _agent_lm = agent._get_lm()
    if _agent_lm is not None:
        _agent_lm.cache = False
    all_cases = SEM_INTENCAO + COM_INTENCAO + RECIPROCIDADE

    print("=" * 100)
    print(f"GreetingAgent v21 — manual evaluation ({len(all_cases)} cases)")
    print(f"Model: {agent.model}, temp={agent.temperature}, max_tokens={agent.max_tokens}")
    print("=" * 100)

    latencies: List[float] = []

    for i, case in enumerate(all_cases, 1):
        kw = case["kwargs"]

        # Reconstruct the exact user prompt the LLM will see, mirroring the
        # legacy-alias reconciliation that GreetingAgent.forward() does.
        patient_message = kw.get("patient_message") or kw.get("latest_incoming") or kw.get("scope_text") or ""
        patient_intents = kw.get("patient_intents") or []
        patient_name = _normalize_contact_name(kw.get("patient_name") or kw.get("contact_name"))
        clinic_name = kw.get("clinic_name", "Clínica")
        assistant_name = kw.get("assistant_name", "Iris")
        few_shots = _coerce_few_shots(
            kw.get("few_shots"), kw.get("initial_greetings"), kw.get("greeting_example")
        )
        session_summary = kw.get("session_summary", "")
        recent_relevant_messages = kw.get("recent_relevant_messages")
        if recent_relevant_messages is None:
            recent_relevant_messages = kw.get("history") or []
        time_gap_hours = kw.get("time_gap_hours")

        case["_user_prompt"] = _build_user_prompt(
            patient_message=patient_message,
            patient_intents=patient_intents,
            patient_name=patient_name,
            clinic_name=clinic_name,
            assistant_name=assistant_name,
            few_shots=few_shots,
            session_summary=session_summary,
            recent_relevant_messages=recent_relevant_messages,
            time_gap_hours=time_gap_hours,
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

        msg_key = (
            case["kwargs"].get("patient_message")
            or case["kwargs"].get("latest_incoming")
            or ""
        )
        print()
        print("-" * 100)
        print(f"[{i:>2}] {case['label']}  ({elapsed:.0f}ms)")
        print("-" * 100)
        print(f"  INPUT:     {msg_key!r}")
        print(f"  OUTPUT:    {content!r}")
        if llm_reasoning:
            print(f"  REASONING: {llm_reasoning}")

    print()
    print("=" * 100)
    print(
        f"Latency: min={min(latencies):.0f}ms  "
        f"p50={sorted(latencies)[len(latencies)//2]:.0f}ms  "
        f"max={max(latencies):.0f}ms"
    )
    print("=" * 100)

    folder = os.path.expanduser(
        "~/Documents/easyscale/kb/07-MVP/Tech/Tests/Agente Greeting"
    )
    date_tag = time.strftime("%Y-%m-%d")
    round_label = os.environ.get("EVAL_ROUND_LABEL", f"run {time.strftime('%H%M%S')}")
    default_path = os.path.join(folder, f"v21 - {round_label} ({date_tag}).md")
    out_path = os.environ.get("EVAL_REPORT_PATH", default_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(all_cases, latencies))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    run_eval()
