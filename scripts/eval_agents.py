#!/usr/bin/env python3
"""
Sofia Eval Agents — chama agentes DSPy diretamente (sem servidor HTTP, sem Supabase)
e avalia a qualidade das respostas usando tests/eval_cases.json.

Uso:
  python scripts/eval_agents.py

Variáveis de ambiente:
  OPENAI_API_KEY    — chave OpenAI para o DSPy (obrigatório)
  DSPY_PROVIDER     — provider DSPy (default: openai)
  DSPY_MODEL        — modelo DSPy (default: gpt-4o-mini)
  MIN_EVAL_SCORE    — score mínimo para passar (default: 0.75)
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── Configuração ──────────────────────────────────────────────────────────────

MIN_EVAL_SCORE = float(os.getenv("MIN_EVAL_SCORE", "0.75"))
EVAL_CASES_FILE = os.getenv("EVAL_CASES_FILE", "tests/eval_cases.json")
EVAL_REPORT_FILE = os.getenv("CI_EVAL_REPORT_FILE", "ci-eval-report.txt")

DEFAULT_CLINIC_NAME = "Sorriso Da Gente"
DEFAULT_PATIENT_NAME = "Paciente Teste"
DEFAULT_BUSINESS_RULES = "[]"
DEFAULT_SERVICES_CONTEXT = '{"services": [], "offers": []}'

VALID_INTENTS = {"GREETING", "FAQ", "SCHEDULE", "HUMAN_ESCALATION", "REENGAGE", "UNCLASSIFIED"}
VALID_STAGES = {"collecting_service", "presenting_slots", "confirming", "booked"}


# ─── Modelos de resultado ──────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    description: str
    score: float
    passed: bool
    result: Dict[str, Any]
    failure_reason: str = ""
    elapsed_ms: float = 0.0


@dataclass
class AgentEvalResult:
    agent: str
    cases: List[CaseResult] = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.score for c in self.cases) / len(self.cases)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.cases if c.passed)


# ─── Scoring ───────────────────────────────────────────────────────────────────

def score_router_case(result: Dict[str, Any], case: Dict[str, Any]) -> Tuple[float, str]:
    expected_intent = case.get("expected_intent")
    got_intent = result.get("intent", "")

    if expected_intent is None:
        # Edge case: just check no crash (any valid intent returned)
        if got_intent in VALID_INTENTS:
            return 1.0, ""
        return 0.0, f"intent inválido: {got_intent!r}"

    if got_intent == expected_intent:
        return 1.0, ""
    return 0.0, f"intent={got_intent!r} (esperado={expected_intent!r})"


def score_scheduler_case(result: Dict[str, Any], case: Dict[str, Any]) -> Tuple[float, str]:
    expected_stage = case.get("expected_stage")
    got_stage = result.get("conversation_stage", "")

    if expected_stage is None:
        # Edge case: just check no crash (any valid stage returned)
        if got_stage in VALID_STAGES:
            return 1.0, ""
        return 0.0, f"stage inválido: {got_stage!r}"

    if got_stage == expected_stage:
        return 1.0, ""
    return 0.0, f"stage={got_stage!r} (esperado={expected_stage!r})"


def score_faq_case(result: Dict[str, Any], case: Dict[str, Any]) -> Tuple[float, str]:
    response = result.get("response_message", "").lower()
    expected_keywords = case.get("expected_keywords", [])
    forbidden_keywords = case.get("forbidden_keywords", [])
    failures = []

    if not response.strip():
        return 0.0, "response_message vazio"

    if not expected_keywords:
        # Edge case: full score for any non-empty response
        score = 1.0
    else:
        matched = [kw for kw in expected_keywords if kw.lower() in response]
        missing = [kw for kw in expected_keywords if kw.lower() not in response]
        score = len(matched) / len(expected_keywords)
        if missing:
            failures.append(f"keywords ausentes: {missing}")

    for kw in forbidden_keywords:
        if kw.lower() in response:
            score = max(0.0, score - 0.5)
            failures.append(f"keyword proibida encontrada: {kw!r}")

    return score, "; ".join(failures)


# ─── Runners ──────────────────────────────────────────────────────────────────

def run_router_eval(agent: Any, cases: List[Dict]) -> AgentEvalResult:
    eval_result = AgentEvalResult(agent="Router")

    for case in cases:
        t0 = time.time()
        try:
            result = agent.forward(
                latest_message=case["message"],
                history=case.get("history", []),
                conversation_stage=case.get("stage", "new"),
                language="pt-BR",
            )
            score, failure_reason = score_router_case(result, case)
        except Exception as e:
            result = {"intent": "", "confidence": 0.0, "reasoning": str(e)}
            score = 0.0
            failure_reason = f"Excecao: {e}"

        elapsed = (time.time() - t0) * 1000
        eval_result.cases.append(CaseResult(
            case_id=case["id"],
            description=case["description"],
            score=score,
            passed=score >= 1.0,
            result=result,
            failure_reason=failure_reason,
            elapsed_ms=elapsed,
        ))

    return eval_result


def run_scheduler_eval(agent: Any, cases: List[Dict]) -> AgentEvalResult:
    eval_result = AgentEvalResult(agent="Scheduler")

    for case in cases:
        t0 = time.time()
        try:
            result = agent.forward(
                patient_message=case["message"],
                history=case.get("history", []),
                available_slots=case.get("slots", []),
                clinic_name=DEFAULT_CLINIC_NAME,
                patient_name=DEFAULT_PATIENT_NAME,
                stage=case.get("stage", "new"),
                services_list=case.get("services", []),
            )
            score, failure_reason = score_scheduler_case(result, case)
        except Exception as e:
            result = {"conversation_stage": "", "reasoning": str(e)}
            score = 0.0
            failure_reason = f"Excecao: {e}"

        elapsed = (time.time() - t0) * 1000
        eval_result.cases.append(CaseResult(
            case_id=case["id"],
            description=case["description"],
            score=score,
            passed=score >= 1.0,
            result=result,
            failure_reason=failure_reason,
            elapsed_ms=elapsed,
        ))

    return eval_result


def run_faq_eval(agent: Any, cases: List[Dict]) -> AgentEvalResult:
    eval_result = AgentEvalResult(agent="FAQResponder")

    for case in cases:
        t0 = time.time()
        try:
            result = agent.forward(
                patient_message=case["message"],
                history=case.get("history", []),
                clinic_name=DEFAULT_CLINIC_NAME,
                patient_name=DEFAULT_PATIENT_NAME,
                services_context=case.get("services_context", DEFAULT_SERVICES_CONTEXT),
                business_rules=case.get("business_rules", DEFAULT_BUSINESS_RULES),
            )
            score, failure_reason = score_faq_case(result, case)
        except Exception as e:
            result = {"response_message": "", "reasoning": str(e)}
            score = 0.0
            failure_reason = f"Excecao: {e}"

        elapsed = (time.time() - t0) * 1000
        eval_result.cases.append(CaseResult(
            case_id=case["id"],
            description=case["description"],
            score=score,
            passed=score >= 1.0,
            result=result,
            failure_reason=failure_reason,
            elapsed_ms=elapsed,
        ))

    return eval_result


# ─── Output ───────────────────────────────────────────────────────────────────

def _build_summary_lines(
    eval_results: List[AgentEvalResult],
    avg_score: float,
    total_cases: int,
    total_passed: int,
) -> List[str]:
    lines = []
    lines.append("\n" + "=" * 65 + "\n")
    lines.append("SOFIA EVAL AGENTS — RESULTADO FINAL\n")
    lines.append("=" * 65 + "\n")

    for er in eval_results:
        lines.append(f"\n--- {er.agent} ({er.passed_count}/{len(er.cases)})  avg={er.avg_score:.2f} ---\n")
        for c in er.cases:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{status}] {c.case_id}  score={c.score:.2f}  {c.description}\n")
            if c.result.get("intent"):
                conf = c.result.get("confidence", 0)
                lines.append(f"         intent={c.result['intent']!r}  conf={conf:.2f}\n")
            if c.result.get("conversation_stage"):
                lines.append(f"         stage={c.result['conversation_stage']!r}\n")
            if c.result.get("response_message"):
                snippet = c.result["response_message"][:100].replace("\n", " ")
                lines.append(f"         resp: {snippet!r}\n")
            if c.failure_reason:
                lines.append(f"         >> {c.failure_reason}\n")

    lines.append("\n" + "=" * 65 + "\n")
    lines.append(f"AVG SCORE : {avg_score:.2f}  (minimo={MIN_EVAL_SCORE})\n")
    lines.append(f"PASSOU    : {total_passed}/{total_cases}\n")
    lines.append("=" * 65 + "\n")
    return lines


def print_summary(
    eval_results: List[AgentEvalResult],
    avg_score: float,
    total_cases: int,
    total_passed: int,
) -> None:
    for line in _build_summary_lines(eval_results, avg_score, total_cases, total_passed):
        print(line, end="")


def write_report_file(
    eval_results: List[AgentEvalResult],
    avg_score: float,
    total_cases: int,
    total_passed: int,
    failure_reason: str = "",
) -> None:
    """Grava relatório em ci-eval-report.txt para leitura pelo expand_eval_cases.py."""
    lines = _build_summary_lines(eval_results, avg_score, total_cases, total_passed)
    if failure_reason:
        lines.append(f"\nFALHA: {failure_reason}\n")
    with open(EVAL_REPORT_FILE, "w") as f:
        f.writelines(lines)


def write_github_output(eval_results: List[AgentEvalResult], avg_score: float) -> None:
    gh_output = os.getenv("GITHUB_OUTPUT")
    if not gh_output:
        return
    total = sum(len(er.cases) for er in eval_results)
    passed = sum(er.passed_count for er in eval_results)
    with open(gh_output, "a") as f:
        f.write(f"eval_avg_score={avg_score:.4f}\n")
        f.write(f"eval_cases_passed={passed}\n")
        f.write(f"eval_cases_total={total}\n")


def write_github_step_summary(eval_results: List[AgentEvalResult], avg_score: float) -> None:
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return
    total = sum(len(er.cases) for er in eval_results)
    passed = sum(er.passed_count for er in eval_results)
    lines = [
        "## Sofia — Eval DSPy Agents\n\n",
        f"**AVG Score:** `{avg_score:.2f}` &nbsp; _(mínimo `{MIN_EVAL_SCORE}`)_\n\n",
        f"**Casos:** {passed}/{total} passaram\n\n",
    ]
    for er in eval_results:
        lines.append(f"### {er.agent} ({er.passed_count}/{len(er.cases)})\n\n")
        lines.append("| ID | Status | Score | Detalhes |\n")
        lines.append("|-----|:------:|------:|----------|\n")
        for c in er.cases:
            icon = "✅" if c.passed else "❌"
            details = c.failure_reason or "-"
            lines.append(f"| `{c.case_id}` | {icon} | {c.score:.2f} | {details} |\n")
        lines.append("\n")
    with open(summary_file, "w") as f:
        f.writelines(lines)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    # Add project root to path so we can import app modules
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from app.core.config import init_dspy
    from app.agents.router.agent import SofiaRouterAgent
    from app.agents.scheduler.agent import SchedulerAgent
    from app.agents.faq_responder.agent import FAQResponderAgent

    print("Sofia Eval Agents")
    print(f"  eval_cases : {EVAL_CASES_FILE}")
    print(f"  min_score  : {MIN_EVAL_SCORE}")

    # 1. Load eval cases
    with open(EVAL_CASES_FILE) as f:
        eval_cases = json.load(f)

    router_cases = eval_cases.get("router", [])
    scheduler_cases = eval_cases.get("scheduler", [])
    faq_cases = eval_cases.get("faq_responder", [])
    total_cases = len(router_cases) + len(scheduler_cases) + len(faq_cases)

    print(
        f"  casos: {len(router_cases)} router + {len(scheduler_cases)} scheduler"
        f" + {len(faq_cases)} faq = {total_cases} total"
    )

    # 2. Init DSPy
    print("\n[1/4] Inicializando DSPy...")
    init_dspy()

    # 3. Instantiate agents
    print("[2/4] Instanciando agentes...")
    router_agent = SofiaRouterAgent()
    scheduler_agent = SchedulerAgent()
    faq_agent = FAQResponderAgent()

    # 4. Run evals
    print(f"\n[3/4] Rodando {total_cases} casos de eval...")
    eval_results: List[AgentEvalResult] = []

    print(f"  -> Router ({len(router_cases)} casos)...")
    router_result = run_router_eval(router_agent, router_cases)
    eval_results.append(router_result)
    print(
        f"     score={router_result.avg_score:.2f}"
        f" ({router_result.passed_count}/{len(router_result.cases)} passaram)"
    )

    print(f"  -> Scheduler ({len(scheduler_cases)} casos)...")
    scheduler_result = run_scheduler_eval(scheduler_agent, scheduler_cases)
    eval_results.append(scheduler_result)
    print(
        f"     score={scheduler_result.avg_score:.2f}"
        f" ({scheduler_result.passed_count}/{len(scheduler_result.cases)} passaram)"
    )

    print(f"  -> FAQResponder ({len(faq_cases)} casos)...")
    faq_result = run_faq_eval(faq_agent, faq_cases)
    eval_results.append(faq_result)
    print(
        f"     score={faq_result.avg_score:.2f}"
        f" ({faq_result.passed_count}/{len(faq_result.cases)} passaram)"
    )

    # 5. Compute scores
    all_scores = [c.score for er in eval_results for c in er.cases]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    total_passed = sum(er.passed_count for er in eval_results)

    # 6. Output
    print("\n[4/4] Emitindo resultados...")
    print_summary(eval_results, avg_score, total_cases, total_passed)

    failure_reason = ""
    if avg_score < MIN_EVAL_SCORE:
        failure_reason = f"score medio {avg_score:.2f} abaixo do minimo {MIN_EVAL_SCORE}"

    write_report_file(eval_results, avg_score, total_cases, total_passed, failure_reason)
    write_github_output(eval_results, avg_score)
    write_github_step_summary(eval_results, avg_score)

    if failure_reason:
        print(f"\nFALHA: {failure_reason}")
        return 1

    print(f"\nOK: todos os criterios atendidos (score={avg_score:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
