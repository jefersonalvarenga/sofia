"""
Evaluation harness — ScheduleRouter SEQUENCE dimension (happy path).

Tests classification of messages that follow the expected sequence handed
off by the upstream Manager agent. Covers each of the 10 sub-types
(8 SEQUENCE + 2 deviation NOT included here) with 3 cases each => 30 cases.

Note: SCHEDULE_FALLBACK and SCHEDULE_CANCEL/CHANGE belong to the deviation
eval and are NOT included here. SCHEDULE_COMPLETION is included because it
is the terminal state of every happy-path sequence.

Mocked sequences used:
  - Avaliacao: [INTAKE, CASHIER, EVALUATION, COMPLETION]
  - Confirmation (cron): [CONFIRMATION, COMPLETION]
  - Reminder (cron): [REMINDER, COMPLETION]
  - Procedimento: [CASHIER, SERVICE, SERVICE_PROTOCOL, COMPLETION]

Verdict (STRICT):
  - next_intent MUST equal expected_next_intent
  - is_deviation MUST be False (it's the happy path)
  - confidence >= 0.70
  - session_data is propagated (non-empty when input was non-empty)

Usage:
    cd easyscale-sofia
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_schedule_sequence.py
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.agents.router.schedule_router import (  # noqa: E402
    ScheduleRouter,
    SYSTEM_PROMPT,
    SCHEDULE_ROUTER_MODEL,
    SCHEDULE_ROUTER_TEMPERATURE,
    SCHEDULE_ROUTER_MAX_TOKENS,
    DEFAULT_SCHEDULE_CONFIDENCE_THRESHOLD,
)


# Reusable sequence mocks (passed in by the upstream Manager in production).
SEQ_EVALUATION = ["SCHEDULE_INTAKE", "SCHEDULE_CASHIER", "SCHEDULE_EVALUATION", "SCHEDULE_COMPLETION"]
SEQ_SERVICE = ["SCHEDULE_CASHIER", "SCHEDULE_SERVICE", "SCHEDULE_SERVICE_PROTOCOL", "SCHEDULE_COMPLETION"]
SEQ_CONFIRMATION = ["SCHEDULE_CONFIRMATION", "SCHEDULE_COMPLETION"]
SEQ_REMINDER = ["SCHEDULE_REMINDER", "SCHEDULE_COMPLETION"]


# ============================================================================
# Test matrix — 10 sub-types × 3 cases = 30 cases (happy path only)
# ============================================================================

CASES: List[Dict[str, Any]] = [
    # ---------- SCHEDULE_INTAKE (3 cases) ----------
    {
        "id": "INTAKE.1",
        "label": "INTAKE.1 — primeiro contato fluxo de avaliacao",
        "expected_next_intent": "SCHEDULE_INTAKE",
        "kwargs": {
            "latest_message": "oi, quero saber sobre avaliacao de fotona",
            "history": [],
            "sequence": SEQ_EVALUATION,
            "current_stage": "new",
            "session_data": [{"name": "evaluation", "data": {"service": "fotona"}}],
        },
    },
    {
        "id": "INTAKE.2",
        "label": "INTAKE.2 — paciente responde pergunta clinica, intake em andamento",
        "expected_next_intent": "SCHEDULE_INTAKE",
        "kwargs": {
            # Sub-agente intake ainda tem mais perguntas a fazer — continua em INTAKE
            "latest_message": "nao tomo nenhum medicamento",
            "history": [
                {"role": "ai", "content": "Antes de prosseguir vou te fazer algumas perguntas clinicas. Voce toma algum medicamento?"},
            ],
            "sequence": SEQ_EVALUATION,
            "current_stage": "new",  # ainda nao entrou em intake; primeira mensagem do paciente respondendo
            "session_data": [{"name": "evaluation", "data": {"service": "fotona"}}],
        },
    },
    {
        "id": "INTAKE.3",
        "label": "INTAKE.3 — primeiro contato no fluxo de evaluation com servico botox",
        "expected_next_intent": "SCHEDULE_INTAKE",
        "kwargs": {
            "latest_message": "queria fazer botox",
            "history": [],
            "sequence": SEQ_EVALUATION,
            "current_stage": "new",
            "session_data": [{"name": "evaluation", "data": {"service": "botox"}}],
        },
    },
    # ---------- SCHEDULE_CASHIER (3 cases) ----------
    {
        "id": "CASHIER.1",
        "label": "CASHIER.1 — paciente acabou intake, segue para cashier",
        "expected_next_intent": "SCHEDULE_CASHIER",
        "kwargs": {
            "latest_message": "ja respondi tudo, sem alergias, sem medicamentos",
            "history": [
                {"role": "ai", "content": "Voce toma algum medicamento? Tem alergia conhecida?"},
            ],
            "sequence": SEQ_EVALUATION,
            "current_stage": "SCHEDULE_INTAKE",
            "session_data": [{"name": "evaluation", "data": {"service": "fotona"}}],
        },
    },
    {
        "id": "CASHIER.2",
        "label": "CASHIER.2 — paciente pergunta sobre valor diretamente (fluxo service)",
        "expected_next_intent": "SCHEDULE_CASHIER",
        "kwargs": {
            "latest_message": "quanto fica o procedimento de fotona?",
            "history": [],
            "sequence": SEQ_SERVICE,
            "current_stage": "new",
            "session_data": [{"name": "service", "data": {"service": "fotona"}}],
        },
    },
    {
        "id": "CASHIER.3",
        "label": "CASHIER.3 — paciente confirma valor, segue para cashier do fluxo procedimento",
        "expected_next_intent": "SCHEDULE_CASHIER",
        "kwargs": {
            "latest_message": "concordo com o valor, pode prosseguir",
            "history": [],
            "sequence": ["SCHEDULE_CASHIER", "SCHEDULE_SERVICE", "SCHEDULE_SERVICE_PROTOCOL", "SCHEDULE_COMPLETION"],
            "current_stage": "new",
            "session_data": [{"name": "service", "data": {"service": "preenchimento"}}],
        },
    },
    # ---------- SCHEDULE_EVALUATION (3 cases) ----------
    {
        "id": "EVAL.1",
        "label": "EVAL.1 — paciente pagou sinal e vai agendar avaliacao",
        "expected_next_intent": "SCHEDULE_EVALUATION",
        "kwargs": {
            "latest_message": "ja paguei o sinal, qual o horario?",
            "history": [],
            "sequence": SEQ_EVALUATION,
            "current_stage": "SCHEDULE_CASHIER",
            "session_data": [{"name": "evaluation", "data": {"service": "fotona"}}],
        },
    },
    {
        "id": "EVAL.2",
        "label": "EVAL.2 — paciente confirma valor da avaliacao",
        "expected_next_intent": "SCHEDULE_EVALUATION",
        "kwargs": {
            "latest_message": "pode ser, fechou. quando consigo?",
            "history": [
                {"role": "ai", "content": "A avaliacao fica em R$ 150."},
            ],
            "sequence": SEQ_EVALUATION,
            "current_stage": "SCHEDULE_CASHIER",
            "session_data": [{"name": "evaluation", "data": {"service": "preenchimento"}}],
        },
    },
    {
        "id": "EVAL.3",
        "label": "EVAL.3 — paciente direto pede horario de avaliacao",
        "expected_next_intent": "SCHEDULE_EVALUATION",
        "kwargs": {
            "latest_message": "quando tem horario pra avaliacao?",
            "history": [],
            "sequence": ["SCHEDULE_EVALUATION", "SCHEDULE_COMPLETION"],
            "current_stage": "new",
            "session_data": [{"name": "evaluation", "data": {"service": "botox"}}],
        },
    },
    # ---------- SCHEDULE_SERVICE (3 cases) ----------
    {
        "id": "SERVICE.1",
        "label": "SERVICE.1 — apos cashier do fluxo procedimento, agenda procedimento",
        "expected_next_intent": "SCHEDULE_SERVICE",
        "kwargs": {
            "latest_message": "pode confirmar o valor, quando temos horario disponivel?",
            "history": [],
            "sequence": SEQ_SERVICE,
            "current_stage": "SCHEDULE_CASHIER",
            "session_data": [{"name": "service", "data": {"service": "fotona"}}],
        },
    },
    {
        "id": "SERVICE.2",
        "label": "SERVICE.2 — paciente diz que ja decidiu fazer o procedimento",
        "expected_next_intent": "SCHEDULE_SERVICE",
        "kwargs": {
            "latest_message": "fechei, quando consigo marcar a sessao?",
            "history": [],
            "sequence": SEQ_SERVICE,
            "current_stage": "SCHEDULE_CASHIER",
            "session_data": [{"name": "service", "data": {"service": "preenchimento labial"}}],
        },
    },
    {
        "id": "SERVICE.3",
        "label": "SERVICE.3 — sequencia mais simples direto em SERVICE",
        "expected_next_intent": "SCHEDULE_SERVICE",
        "kwargs": {
            "latest_message": "tem agenda pra essa semana?",
            "history": [],
            "sequence": ["SCHEDULE_SERVICE", "SCHEDULE_SERVICE_PROTOCOL", "SCHEDULE_COMPLETION"],
            "current_stage": "new",
            "session_data": [{"name": "service", "data": {"service": "peeling"}}],
        },
    },
    # ---------- SCHEDULE_SERVICE_PROTOCOL (3 cases) ----------
    {
        "id": "PROTOCOL.1",
        "label": "PROTOCOL.1 — paciente confirma horario do procedimento, recebe protocolo",
        "expected_next_intent": "SCHEDULE_SERVICE_PROTOCOL",
        "kwargs": {
            "latest_message": "perfeito, confirmado terca as 14h",
            "history": [
                {"role": "ai", "content": "Tem horario terca as 14h ou quinta as 10h."},
            ],
            "sequence": SEQ_SERVICE,
            "current_stage": "SCHEDULE_SERVICE",
            "session_data": [{"name": "service", "data": {"service": "fotona", "slot": "terca 14h"}}],
        },
    },
    {
        "id": "PROTOCOL.2",
        "label": "PROTOCOL.2 — paciente fechou agendamento e pede informacoes de preparo",
        "expected_next_intent": "SCHEDULE_SERVICE_PROTOCOL",
        "kwargs": {
            "latest_message": "agendei, preciso saber se tem algum preparo antes do procedimento",
            "history": [],
            "sequence": SEQ_SERVICE,
            "current_stage": "SCHEDULE_SERVICE",
            "session_data": [{"name": "service", "data": {"service": "preenchimento"}}],
        },
    },
    {
        "id": "PROTOCOL.3",
        "label": "PROTOCOL.3 — confirmacao curta apos agendamento",
        "expected_next_intent": "SCHEDULE_SERVICE_PROTOCOL",
        "kwargs": {
            "latest_message": "ok, agendado",
            "history": [
                {"role": "ai", "content": "Confirmado quinta as 10h. Vou enviar as orientacoes de preparo agora."},
            ],
            "sequence": SEQ_SERVICE,
            "current_stage": "SCHEDULE_SERVICE",
            "session_data": [{"name": "service", "data": {"service": "fotona"}}],
        },
    },
    # ---------- SCHEDULE_CONFIRMATION (3 cases) ----------
    {
        "id": "CONFIRM.1",
        "label": "CONFIRM.1 — paciente confirma presenca explicitamente",
        "expected_next_intent": "SCHEDULE_CONFIRMATION",
        "kwargs": {
            "latest_message": "sim, confirmo presenca",
            "history": [
                {"role": "ai", "content": "Voce confirma sua presenca na consulta de amanha as 15h?"},
            ],
            "sequence": SEQ_CONFIRMATION,
            "current_stage": "new",
            "session_data": [{"name": "confirmation", "data": {"appointment_id": "apt_123"}}],
        },
    },
    {
        "id": "CONFIRM.2",
        "label": "CONFIRM.2 — paciente confirma com 'pode contar comigo'",
        "expected_next_intent": "SCHEDULE_CONFIRMATION",
        "kwargs": {
            "latest_message": "pode contar comigo, vou sim",
            "history": [],
            "sequence": SEQ_CONFIRMATION,
            "current_stage": "new",
            "session_data": [{"name": "confirmation", "data": {"appointment_id": "apt_456"}}],
        },
    },
    {
        "id": "CONFIRM.3",
        "label": "CONFIRM.3 — paciente confirma com 'tudo certo'",
        "expected_next_intent": "SCHEDULE_CONFIRMATION",
        "kwargs": {
            "latest_message": "tudo certo, estarei la",
            "history": [],
            "sequence": SEQ_CONFIRMATION,
            "current_stage": "new",
            "session_data": [{"name": "confirmation", "data": {"appointment_id": "apt_789"}}],
        },
    },
    # ---------- SCHEDULE_REMINDER (3 cases) ----------
    {
        "id": "REMIND.1",
        "label": "REMIND.1 — cron disparou reminder e paciente acabou de receber",
        "expected_next_intent": "SCHEDULE_REMINDER",
        "kwargs": {
            "latest_message": "ok, recebi",
            "history": [
                {"role": "ai", "content": "Oi! Lembrando que sua consulta e em 2h. Confirma sua presenca?"},
            ],
            "sequence": SEQ_REMINDER,
            "current_stage": "new",
            "session_data": [{"name": "reminder", "data": {"appointment_id": "apt_999"}}],
        },
    },
    {
        "id": "REMIND.2",
        "label": "REMIND.2 — paciente pergunta endereco apos reminder",
        "expected_next_intent": "SCHEDULE_REMINDER",
        "kwargs": {
            "latest_message": "qual o endereco da clinica mesmo?",
            "history": [
                {"role": "ai", "content": "Lembrando do seu horario hoje as 16h. Algo que voce precise?"},
            ],
            "sequence": SEQ_REMINDER,
            "current_stage": "new",
            "session_data": [{"name": "reminder", "data": {"appointment_id": "apt_111"}}],
        },
    },
    {
        "id": "REMIND.3",
        "label": "REMIND.3 — paciente pergunta tempo de antecedencia para chegar",
        "expected_next_intent": "SCHEDULE_REMINDER",
        "kwargs": {
            "latest_message": "preciso chegar com quanto tempo de antecedencia?",
            "history": [],
            "sequence": SEQ_REMINDER,
            "current_stage": "new",
            "session_data": [{"name": "reminder", "data": {"appointment_id": "apt_222"}}],
        },
    },
    # ---------- SCHEDULE_COMPLETION (3 cases) ----------
    {
        "id": "COMPLETE.1",
        "label": "COMPLETE.1 — paciente fechou confirmacao, sub-fluxo concluido",
        "expected_next_intent": "SCHEDULE_COMPLETION",
        "kwargs": {
            "latest_message": "obrigado",
            "history": [
                {"role": "human", "content": "sim, confirmo"},
                {"role": "ai", "content": "Otimo! Presenca confirmada."},
            ],
            "sequence": SEQ_CONFIRMATION,
            "current_stage": "SCHEDULE_CONFIRMATION",
            "session_data": [{"name": "confirmation", "data": {"appointment_id": "apt_333"}}],
        },
    },
    {
        "id": "COMPLETE.2",
        "label": "COMPLETE.2 — paciente recebeu protocolo, ack rapido",
        "expected_next_intent": "SCHEDULE_COMPLETION",
        "kwargs": {
            "latest_message": "ok, entendi as orientacoes",
            "history": [
                {"role": "ai", "content": "Aqui estao as orientacoes de preparo para o procedimento..."},
            ],
            "sequence": SEQ_SERVICE,
            "current_stage": "SCHEDULE_SERVICE_PROTOCOL",
            "session_data": [{"name": "service", "data": {"service": "fotona"}}],
        },
    },
    {
        "id": "COMPLETE.3",
        "label": "COMPLETE.3 — paciente fechou avaliacao",
        "expected_next_intent": "SCHEDULE_COMPLETION",
        "kwargs": {
            "latest_message": "perfeito, agendado entao",
            "history": [
                {"role": "ai", "content": "Agendado terca as 16h para sua avaliacao de fotona."},
            ],
            "sequence": SEQ_EVALUATION,
            "current_stage": "SCHEDULE_EVALUATION",
            "session_data": [{"name": "evaluation", "data": {"service": "fotona", "slot": "terca 16h"}}],
        },
    },
]


# ============================================================================
# Verdict
# ============================================================================

def auto_verdict(case: Dict[str, Any], result: Dict[str, Any]) -> str:
    expected = case["expected_next_intent"]
    next_intent = result.get("next_intent")
    is_deviation = bool(result.get("is_deviation", False))
    confidence = float(result.get("confidence") or 0.0)

    if next_intent != expected:
        return "NO"
    if is_deviation:
        # Happy path => must NOT be a deviation.
        return "NO"
    if confidence < DEFAULT_SCHEDULE_CONFIDENCE_THRESHOLD:
        return "NO"

    # session_data sanity: if input had entries, output must too.
    if case["kwargs"].get("session_data") and not result.get("session_data"):
        return "NO"

    return "YES"


# ============================================================================
# Report
# ============================================================================

def _render_markdown(cases: List[Dict[str, Any]], latencies: List[float]) -> str:
    lines: List[str] = []
    lines.append("# Avaliação SEQUENCE — ScheduleRouter (caminho feliz)")
    lines.append("")
    lines.append(f"- **Modelo:** `{SCHEDULE_ROUTER_MODEL}`")
    lines.append(f"- **Temperature:** `{SCHEDULE_ROUTER_TEMPERATURE}`")
    lines.append(f"- **max_tokens:** `{SCHEDULE_ROUTER_MAX_TOKENS}`")
    lines.append(f"- **Threshold:** `{DEFAULT_SCHEDULE_CONFIDENCE_THRESHOLD}`")
    lines.append(f"- **Casos:** {len(cases)} (10 sub-tipos × 3)")
    sorted_lats = sorted(latencies)
    p50 = sorted_lats[len(sorted_lats) // 2]
    p99 = sorted_lats[int(len(sorted_lats) * 0.99)] if len(sorted_lats) > 99 else max(sorted_lats)
    lines.append(
        f"- **Latência:** min={min(latencies):.0f}ms  "
        f"p50={p50:.0f}ms  p99={p99:.0f}ms  max={max(latencies):.0f}ms"
    )
    lines.append("")
    lines.append("## Critério (estrito)")
    lines.append("")
    lines.append("- `next_intent` igual ao esperado.")
    lines.append("- `is_deviation` deve ser `False` (caminho feliz).")
    lines.append("- `confidence >= 0.70`.")
    lines.append("- `session_data` propagado quando havia entrada.")
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
        lines.append(f"**Esperado:** `{case['expected_next_intent']}`")
        lines.append("")
        lines.append(f"**Sequence:** `{case['kwargs']['sequence']}`")
        lines.append(f"**Current stage:** `{case['kwargs']['current_stage']}`")
        lines.append(f"**Input:** `{case['kwargs']['latest_message']!r}`")
        lines.append("")
        lines.append(f"**Detected:** `{case['_next_intent']}`  is_deviation=`{case['_is_deviation']}`  conf=`{case['_confidence']}`")
        lines.append(f"**Reasoning:** {case['_reasoning']!r}")
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
    router = ScheduleRouter()
    _lm = router._get_lm()
    if _lm is not None:
        _lm.cache = False

    print("=" * 100)
    print(f"ScheduleRouter — SEQUENCE eval ({len(CASES)} cases)")
    print(f"Model: {router.model}, temp={router.temperature}, max_tokens={router.max_tokens}")
    print("=" * 100)

    latencies: List[float] = []

    for i, case in enumerate(CASES, 1):
        kw = case["kwargs"]
        t0 = time.perf_counter()
        try:
            result = router.forward(**kw)
        except Exception as exc:
            result = {
                "next_intent": "SCHEDULE_FALLBACK",
                "is_deviation": True,
                "session_data": [],
                "confidence": 0.0,
                "reasoning": f"EXCEPTION: {type(exc).__name__}: {exc}",
            }
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        case["_next_intent"] = result.get("next_intent")
        case["_is_deviation"] = result.get("is_deviation")
        case["_confidence"] = result.get("confidence") or 0.0
        case["_reasoning"] = result.get("reasoning") or ""
        case["_session_data"] = result.get("session_data") or []
        case["_elapsed_ms"] = elapsed
        case["_auto_verdict"] = auto_verdict(case, result)

        print()
        print("-" * 100)
        print(f"[{i:>2}] {case['label']}  ({elapsed:.0f}ms)")
        print("-" * 100)
        print(f"  EXPECT:    {case['expected_next_intent']}")
        print(f"  SEQUENCE:  {kw['sequence']}")
        print(f"  STAGE:     {kw['current_stage']}")
        print(f"  INPUT:     {kw['latest_message']!r}")
        print(f"  DETECTED:  {case['_next_intent']}  dev={case['_is_deviation']}  conf={case['_confidence']}")
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
        "~/Documents/easyscale/kb/07-MVP/Tech/Tests/Schedule Router"
    )
    os.makedirs(folder, exist_ok=True)
    date_tag = time.strftime("%Y-%m-%d")
    round_label = os.environ.get("EVAL_ROUND_LABEL", f"run {time.strftime('%H%M%S')}")
    default_path = os.path.join(folder, f"sequence - {round_label} ({date_tag}).md")
    out_path = os.environ.get("EVAL_REPORT_PATH", default_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(CASES, latencies))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    run_eval()
