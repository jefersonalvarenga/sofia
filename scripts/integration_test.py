#!/usr/bin/env python3
"""
Sofia Integration Test — chama POST /v1/sofia via HTTP (como o n8n faz)
e verifica que os agentes corretos são ativados para cada cenário.

Uso:
  python scripts/integration_test.py [--base-url http://localhost:8000]

Variáveis de ambiente:
  API_KEY          — chave de autenticação da Sofia (obrigatório)
  SUPABASE_URL     — URL do Supabase (necessário para o servidor subir)
  SUPABASE_KEY     — chave do Supabase
  OPENAI_API_KEY   — chave OpenAI para o DSPy
  TEST_CLINIC_ID   — UUID da clínica de testes (default: Sorriso Da Gente)
  MIN_AVG_SCORE    — score mínimo para passar (default: 0.8)
"""

import argparse
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

# ─── Configuração ──────────────────────────────────────────────────────────────

BASE_URL = os.getenv("SOFIA_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
CLINIC_ID = os.getenv("TEST_CLINIC_ID", "a4a04b17-0158-48b2-b4e3-1175825c84c4")
MIN_AVG_SCORE = float(os.getenv("MIN_AVG_SCORE", "0.8"))

# JID único por run para isolar sessões paralelas no banco
RUN_ID = str(uuid.uuid4())[:8]
TEST_JID = f"5500000{RUN_ID}@s.whatsapp.net"

# Slots fixos para testes de agendamento (datas futuras hardcoded)
TEST_SLOTS = [
    "2027-01-05 09:00",
    "2027-01-05 10:00",
    "2027-01-05 11:00",
]


# ─── Modelos de resultado ──────────────────────────────────────────────────────

@dataclass
class TurnResult:
    turn: int
    message_sent: str
    status_code: int
    response_message: str
    agent_name: str
    conversation_stage: str
    requires_human: bool
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    name: str
    description: str
    turns: List[TurnResult] = field(default_factory=list)
    passed: bool = False
    failure_reason: str = ""
    score: float = 0.0


# ─── HTTP helper ───────────────────────────────────────────────────────────────

def call_sofia(
    client: httpx.Client,
    message: str,
    available_slots: List[str],
    turn: int,
) -> TurnResult:
    try:
        resp = client.post(
            f"{BASE_URL}/v1/sofia",
            json={
                "instance_id": "test-instance",
                "clinic_id": CLINIC_ID,
                "remote_jid": TEST_JID,
                "push_name": "Paciente Teste",
                "message": message,
                "message_type": "text",
                "wamid": f"test-{RUN_ID}-t{turn}",
                "available_slots": available_slots,
            },
            timeout=45.0,
        )
        if resp.status_code == 200:
            body = resp.json()
            return TurnResult(
                turn=turn,
                message_sent=message,
                status_code=200,
                response_message=body.get("response_message", ""),
                agent_name=body.get("agent_name", ""),
                conversation_stage=body.get("conversation_stage", ""),
                requires_human=body.get("requires_human", False),
            )
        else:
            return TurnResult(
                turn=turn,
                message_sent=message,
                status_code=resp.status_code,
                response_message="",
                agent_name="",
                conversation_stage="",
                requires_human=False,
                error=resp.text[:300],
            )
    except Exception as e:
        return TurnResult(
            turn=turn,
            message_sent=message,
            status_code=0,
            response_message="",
            agent_name="",
            conversation_stage="",
            requires_human=False,
            error=str(e),
        )


# ─── Cenários ─────────────────────────────────────────────────────────────────
#
# Cada cenário tem:
#   turns       — lista de mensagens do paciente (multi-turno para flows longos)
#   assertions  — lista de (tipo, valor_esperado) aplicadas ao ÚLTIMO turno
#
# Tipos de assertion:
#   "agent_name"       — verifica agent_name exato
#   "requires_human"   — verifica bool requires_human
#   "no_error"         — verifica HTTP 200 sem erro
#   "response_not_empty" — verifica que response_message não está vazio

SCENARIOS: List[Dict[str, Any]] = [
    {
        "name": "greeting",
        "description": "Saudação simples → FAQResponder ativa, sem escalada",
        "turns": [
            {"message": "oi, boa tarde!", "available_slots": []},
        ],
        "assertions": [
            ("agent_name", "FAQResponder"),
            ("requires_human", False),
            ("no_error", None),
            ("response_not_empty", None),
        ],
    },
    {
        "name": "faq_pricing",
        "description": "Pergunta sobre preço → FAQResponder responde com valor",
        "turns": [
            {"message": "quanto custa clareamento dental?", "available_slots": []},
        ],
        "assertions": [
            ("agent_name", "FAQResponder"),
            ("requires_human", False),
            ("no_error", None),
            ("response_not_empty", None),
        ],
    },
    {
        "name": "schedule_intent",
        "description": "Intenção clara de agendar + serviço → Scheduler coleta dados",
        "turns": [
            {"message": "quero marcar uma consulta de clareamento", "available_slots": TEST_SLOTS},
        ],
        "assertions": [
            ("agent_name", "Scheduler"),
            ("requires_human", False),
            ("no_error", None),
            ("response_not_empty", None),
        ],
    },
    {
        "name": "human_escalation",
        "description": "Pedido explícito de atendente → HumanEscalation + requires_human=True",
        "turns": [
            {"message": "quero falar com um atendente humano por favor", "available_slots": []},
        ],
        "assertions": [
            ("agent_name", "HumanEscalation"),
            ("requires_human", True),
            ("no_error", None),
            ("response_not_empty", None),
        ],
    },
    {
        "name": "no_500_on_ambiguous",
        "description": "Mensagem ambígua → sem HTTP 500, resposta gerada",
        "turns": [
            {"message": "hmm", "available_slots": []},
        ],
        "assertions": [
            ("no_error", None),
            ("response_not_empty", None),
        ],
    },
    {
        "name": "faq_insurance",
        "description": "Pergunta sobre convênio → FAQResponder responde",
        "turns": [
            {"message": "vocês aceitam plano Unimed?", "available_slots": []},
        ],
        "assertions": [
            ("agent_name", "FAQResponder"),
            ("no_error", None),
            ("response_not_empty", None),
        ],
    },
]


# ─── Runner ────────────────────────────────────────────────────────────────────

def run_scenario(client: httpx.Client, scenario: Dict[str, Any]) -> ScenarioResult:
    result = ScenarioResult(name=scenario["name"], description=scenario["description"])
    last_turn: Optional[TurnResult] = None

    for i, turn_def in enumerate(scenario["turns"]):
        tr = call_sofia(client, turn_def["message"], turn_def["available_slots"], i + 1)
        result.turns.append(tr)
        last_turn = tr

        # Abort multi-turn early on HTTP error
        if tr.error or tr.status_code != 200:
            result.failure_reason = (
                f"Turn {i+1} falhou: status={tr.status_code} err={tr.error}"
            )
            result.score = 0.0
            return result

        # Brief pause between turns (same session)
        if i < len(scenario["turns"]) - 1:
            time.sleep(1)

    # Evaluate assertions against the last turn
    assert_count = len(scenario["assertions"])
    passed_count = 0
    failures = []

    for assertion_type, expected in scenario["assertions"]:
        ok = False
        if assertion_type == "agent_name":
            ok = last_turn.agent_name == expected
            if not ok:
                failures.append(
                    f"agent_name={last_turn.agent_name!r} (esperado={expected!r})"
                )
        elif assertion_type == "requires_human":
            ok = last_turn.requires_human == expected
            if not ok:
                failures.append(
                    f"requires_human={last_turn.requires_human} (esperado={expected})"
                )
        elif assertion_type == "no_error":
            ok = last_turn.error is None and last_turn.status_code == 200
            if not ok:
                failures.append(f"erro: status={last_turn.status_code} {last_turn.error}")
        elif assertion_type == "response_not_empty":
            ok = bool(last_turn.response_message.strip())
            if not ok:
                failures.append("response_message está vazio")

        if ok:
            passed_count += 1

    result.score = passed_count / assert_count
    result.passed = result.score >= 1.0
    if failures:
        result.failure_reason = "; ".join(failures)

    return result


# ─── Health check ──────────────────────────────────────────────────────────────

def health_check() -> bool:
    for attempt in range(12):
        try:
            resp = httpx.get(f"{BASE_URL}/v1/health", timeout=5.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        print(f"  Aguardando Sofia subir... tentativa {attempt + 1}/12")
        time.sleep(5)
    return False


# ─── Output ───────────────────────────────────────────────────────────────────

def _build_summary_lines(results: List[ScenarioResult], avg_score: float) -> List[str]:
    """Constrói as linhas do resumo (compartilhado entre stdout e arquivo)."""
    lines = []
    lines.append("\n" + "=" * 65 + "\n")
    lines.append("SOFIA INTEGRATION TEST — RESULTADO FINAL\n")
    lines.append("=" * 65 + "\n")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"\n[{status}] {r.name}  score={r.score:.2f}\n")
        lines.append(f"       {r.description}\n")
        for tr in r.turns:
            lines.append(
                f"       turn {tr.turn}: agent={tr.agent_name!r}"
                f" stage={tr.conversation_stage!r}"
                f" requires_human={tr.requires_human}\n"
            )
            if tr.response_message:
                snippet = tr.response_message[:120].replace("\n", " ")
                lines.append(f"       resposta: {snippet!r}\n")
        if r.failure_reason:
            lines.append(f"       >> {r.failure_reason}\n")

    passed = sum(1 for r in results if r.passed)
    lines.append("\n" + "=" * 65 + "\n")
    lines.append(f"AVG SCORE : {avg_score:.2f}  (minimo={MIN_AVG_SCORE})\n")
    lines.append(f"PASSOU    : {passed}/{len(results)}\n")
    lines.append("=" * 65 + "\n")
    return lines


def print_summary(results: List[ScenarioResult], avg_score: float) -> None:
    for line in _build_summary_lines(results, avg_score):
        print(line, end="")


def write_report_file(results: List[ScenarioResult], avg_score: float) -> None:
    """Grava relatório em ci-test-report.txt para leitura pelo diagnose_failure.py."""
    report_path = os.getenv("CI_REPORT_FILE", "ci-test-report.txt")
    with open(report_path, "w") as f:
        f.writelines(_build_summary_lines(results, avg_score))


def write_github_output(results: List[ScenarioResult], avg_score: float) -> None:
    gh_output = os.getenv("GITHUB_OUTPUT")
    if not gh_output:
        return
    passed = sum(1 for r in results if r.passed)
    with open(gh_output, "a") as f:
        f.write(f"avg_score={avg_score:.4f}\n")
        f.write(f"scenarios_passed={passed}\n")
        f.write(f"scenarios_total={len(results)}\n")


def write_github_step_summary(results: List[ScenarioResult], avg_score: float) -> None:
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return
    passed = sum(1 for r in results if r.passed)
    lines = [
        "## Sofia — Integration Test\n\n",
        f"**AVG Score:** `{avg_score:.2f}` &nbsp; _(mínimo `{MIN_AVG_SCORE}`)_\n\n",
        f"**Cenários:** {passed}/{len(results)} passaram\n\n",
        "| Cenário | Status | Score | Agente | Falha |\n",
        "|---------|:------:|------:|--------|-------|\n",
    ]
    for r in results:
        icon = "✅" if r.passed else "❌"
        last = r.turns[-1] if r.turns else None
        agent = last.agent_name if last else "-"
        lines.append(
            f"| `{r.name}` | {icon} | {r.score:.2f} | {agent} | {r.failure_reason or '-'} |\n"
        )
    with open(summary_file, "w") as f:
        f.writelines(lines)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    global BASE_URL

    parser = argparse.ArgumentParser(description="Sofia Integration Test")
    parser.add_argument("--base-url", default=BASE_URL, help="Base URL da API Sofia")
    args = parser.parse_args()

    BASE_URL = args.base_url

    print("Sofia Integration Test")
    print(f"  base_url  : {BASE_URL}")
    print(f"  clinic_id : {CLINIC_ID}")
    print(f"  test_jid  : {TEST_JID}")
    print(f"  min_score : {MIN_AVG_SCORE}")
    print(f"  cenarios  : {len(SCENARIOS)}")

    # 1. Health check
    print("\n[1/3] Health check...")
    if not health_check():
        print("ERRO: Sofia nao respondeu ao health check. Abortando.")
        return 1
    print("Sofia esta online.")

    # 2. Run scenarios
    print(f"\n[2/3] Rodando {len(SCENARIOS)} cenarios...")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    results: List[ScenarioResult] = []

    with httpx.Client(headers=headers) as client:
        for scenario in SCENARIOS:
            print(f"  -> {scenario['name']}...")
            r = run_scenario(client, scenario)
            results.append(r)
            status = "OK" if r.passed else "FAIL"
            line = f"     [{status}] score={r.score:.2f}"
            if r.failure_reason:
                line += f" -- {r.failure_reason}"
            print(line)

    # 3. Score e output
    avg_score = sum(r.score for r in results) / len(results)

    print("\n[3/3] Emitindo resultados...")
    print_summary(results, avg_score)
    write_report_file(results, avg_score)
    write_github_output(results, avg_score)
    write_github_step_summary(results, avg_score)

    if avg_score < MIN_AVG_SCORE:
        print(f"\nFALHA: score medio {avg_score:.2f} abaixo do minimo {MIN_AVG_SCORE}")
        return 1

    print(f"\nOK: todos os criterios atendidos (score={avg_score:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
