#!/usr/bin/env python3
"""
Expand Sofia eval cases via GLM-4-flash (gratuito).

Lê ci-eval-report.txt, chama GLM-4-flash, gera novos casos de teste e
opcionalmente auto-commita em tests/eval_cases.json.

Uso:
  python scripts/expand_eval_cases.py [--auto-commit]

Variáveis de ambiente:
  GLM_API_KEY       — API key ZhipuAI (obrigatório)
  GLM_MODEL         — modelo GLM (default: glm-4-flash)
  GH_TOKEN          — token GitHub para postar commit comment
  GH_REPO           — repositório GitHub (ex: org/repo)
  GH_SHA            — SHA do commit que disparou o workflow
  GITHUB_RUN_ID     — ID da run do GitHub Actions
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

EVAL_REPORT_FILE = os.getenv("CI_EVAL_REPORT_FILE", "ci-eval-report.txt")
EVAL_CASES_FILE = os.getenv("EVAL_CASES_FILE", "tests/eval_cases.json")
GH_REPO = os.getenv("GH_REPO", "")
GH_SHA = os.getenv("GH_SHA", "")
GITHUB_RUN_ID = os.getenv("GITHUB_RUN_ID", "")

SYSTEM_PROMPT = """\
Você é um engenheiro sênior especialista em testes de LLMs e agentes conversacionais.

A Sofia é um agente de atendimento de pacientes (WhatsApp) para clínicas odontológicas \
com os seguintes componentes:

- **Router**: classifica intenção do paciente.
  Intents válidos: GREETING, FAQ, SCHEDULE, HUMAN_ESCALATION, REENGAGE, UNCLASSIFIED
- **Scheduler**: agendamento multi-etapa.
  Stages: collecting_service → presenting_slots → confirming → booked
  Guards: nunca vai para "booked" sem chosen_slot; nunca vai para "presenting_slots" sem slots disponíveis
- **FAQResponder**: responde perguntas sobre serviços, preços, convênios, horários, endereço

Você receberá um relatório de eval mostrando quais casos passaram/falharam, \
além de exemplos dos casos existentes e a lista de IDs já usados.

Gere NOVOS casos de teste em JSON que:
1. Cobrem lacunas identificadas no relatório (focar nos agentes com scores baixos)
2. Testam variações linguísticas (gírias, erros de digitação, português informal)
3. Incluem casos multi-turno com histórico rico
4. NÃO duplicam IDs ou mensagens já existentes

Schemas dos casos:

Router:
{"id": "rXX", "message": "...", "history": [], "stage": "new",
 "expected_intent": "INTENT_OU_NULL", "description": "..."}

Scheduler:
{"id": "sXX", "message": "...", "history": [], "stage": "collecting_service",
 "slots": ["2027-01-05 09:00"], "services": ["Serviço A"],
 "expected_stage": "STAGE_OU_NULL", "description": "..."}

FAQResponder:
{"id": "fXX", "message": "...", "history": [],
 "services_context": "{\"services\": [...], \"offers\": [...]}",
 "business_rules": "[...]",
 "expected_keywords": [], "forbidden_keywords": [], "description": "..."}

Responda SOMENTE com JSON válido neste formato exato:
{
  "recommendations": "texto markdown com análise das falhas e sugestões de melhoria no código",
  "new_cases": {
    "router": [...],
    "scheduler": [...],
    "faq_responder": [...]
  }
}\
"""


def read_eval_report() -> str:
    if not os.path.exists(EVAL_REPORT_FILE):
        return ""
    with open(EVAL_REPORT_FILE) as f:
        return f.read().strip()


def read_eval_cases() -> Dict[str, Any]:
    with open(EVAL_CASES_FILE) as f:
        return json.load(f)


def collect_existing_ids(cases: Dict[str, Any]) -> List[str]:
    ids = []
    for agent_cases in cases.values():
        if isinstance(agent_cases, list):
            ids.extend(c.get("id", "") for c in agent_cases if isinstance(c, dict))
    return ids


def collect_existing_messages(cases: Dict[str, Any]) -> List[str]:
    messages = []
    for agent_cases in cases.values():
        if isinstance(agent_cases, list):
            messages.extend(
                c.get("message", "").lower()
                for c in agent_cases
                if isinstance(c, dict)
            )
    return messages


def _build_examples_snippet(eval_cases: Dict) -> str:
    """Return compact JSON of 2 examples per agent to guide GLM."""
    examples: Dict[str, Any] = {}
    for agent in ("router", "scheduler", "faq_responder"):
        cases = eval_cases.get(agent, [])
        examples[agent] = cases[:2] if cases else []
    return json.dumps(examples, ensure_ascii=False, indent=2)


def call_glm(report: str, existing_ids: List[str], examples: str) -> str:
    from zhipuai import ZhipuAI

    client = ZhipuAI(api_key=os.getenv("GLM_API_KEY"))
    model = os.getenv("GLM_MODEL", "glm-4-flash")

    user_content = (
        f"IDs já existentes (não duplicar): {existing_ids}\n\n"
        f"Exemplos dos casos existentes:\n{examples}\n\n"
        f"Relatório de eval:\n\n```\n{report}\n```\n\n"
        "Gere novos casos focando nas áreas com falhas e lacunas identificadas. "
        "Mínimo 3 novos casos por agente, máximo 10 por agente."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_tokens=2000,
        temperature=0.7,
    )
    return response.choices[0].message.content


def validate_and_extract(raw_json: str) -> Tuple[str, Dict[str, List]]:
    """Parse GLM response. Returns (recommendations, new_cases_by_agent)."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"Aviso: JSON invalido do GLM: {e}")
        return "", {"router": [], "scheduler": [], "faq_responder": []}

    recommendations = data.get("recommendations", "")
    raw_new = data.get("new_cases", {})

    validated: Dict[str, List] = {"router": [], "scheduler": [], "faq_responder": []}
    for agent in validated:
        for c in raw_new.get(agent, []):
            if isinstance(c, dict) and c.get("id") and c.get("message"):
                validated[agent].append(c)

    return recommendations, validated


def deduplicate(
    new_cases: Dict[str, List],
    existing_ids: List[str],
    existing_messages: List[str],
) -> Dict[str, List]:
    """Remove cases with duplicate IDs or identical messages."""
    seen_ids = set(existing_ids)
    seen_msgs = set(m.lower() for m in existing_messages)
    result: Dict[str, List] = {}

    for agent, cases in new_cases.items():
        unique = []
        for c in cases:
            cid = c.get("id", "")
            cmsg = c.get("message", "").lower()
            if cid in seen_ids:
                print(f"  Skip duplicado (ID): {cid}")
                continue
            if cmsg in seen_msgs:
                print(f"  Skip duplicado (msg): {cmsg!r}")
                continue
            seen_ids.add(cid)
            seen_msgs.add(cmsg)
            unique.append(c)
        result[agent] = unique

    return result


def merge_and_save(eval_cases: Dict, new_cases: Dict[str, List]) -> int:
    """Merge new cases into eval_cases file. Returns count of cases added."""
    total_added = 0
    for agent in ("router", "scheduler", "faq_responder"):
        cases = new_cases.get(agent, [])
        if cases:
            eval_cases[agent] = eval_cases.get(agent, []) + cases
            total_added += len(cases)
            print(f"  +{len(cases)} casos adicionados ao {agent}")

    with open(EVAL_CASES_FILE, "w") as f:
        json.dump(eval_cases, f, ensure_ascii=False, indent=2)

    return total_added


def git_commit_and_push(total_added: int) -> bool:
    commit_msg = (
        f"eval: expand {total_added} casos via GLM-4-flash [skip ci]\n\n"
        "Co-Authored-By: GLM-4-flash <noreply@zhipuai.com>"
    )
    cmds = [
        ["git", "add", EVAL_CASES_FILE],
        ["git", "commit", "-m", commit_msg],
        ["git", "push"],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"Aviso: {' '.join(cmd)} falhou: {r.stderr.strip()}")
            return False
    print(f"Commit: {total_added} novos casos adicionados.")
    return True


def post_commit_comment(recommendations: str, total_added: int) -> bool:
    run_url = f"https://github.com/{GH_REPO}/actions/runs/{GITHUB_RUN_ID}"
    body = (
        "## Sofia Eval — Expansão de Casos via GLM-4-flash 🤖\n\n"
        f"**{total_added} novos casos adicionados** em `tests/eval_cases.json`.\n\n"
        f"### Recomendações\n\n{recommendations}\n\n"
        "---\n"
        f"_Gerado por `glm-4-flash` · commit `{GH_SHA[:8]}` · "
        f"[ver log completo]({run_url})_"
    )
    r = subprocess.run(
        [
            "gh", "api",
            f"repos/{GH_REPO}/commits/{GH_SHA}/comments",
            "-f", f"body={body}",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "GH_TOKEN": os.getenv("GH_TOKEN", "")},
    )
    if r.returncode == 0:
        print(f"Comentario postado no commit {GH_SHA[:8]}.")
        return True
    print(f"Falha ao postar comentario: {r.stderr.strip()}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand Sofia eval cases via GLM-4-flash")
    parser.add_argument("--auto-commit", action="store_true", help="Auto-commita novos casos")
    args = parser.parse_args()

    report = read_eval_report()
    if not report:
        print(f"Arquivo {EVAL_REPORT_FILE} nao encontrado ou vazio. Nada a expandir.")
        return 0

    if not os.getenv("GLM_API_KEY"):
        print("Aviso: GLM_API_KEY nao configurada. Pulando expansao de casos.")
        return 0

    print("Lendo eval_cases.json...")
    eval_cases = read_eval_cases()
    existing_ids = collect_existing_ids(eval_cases)
    existing_messages = collect_existing_messages(eval_cases)
    examples_snippet = _build_examples_snippet(eval_cases)
    print(f"  {len(existing_ids)} casos existentes")

    print(f"Chamando GLM ({os.getenv('GLM_MODEL', 'glm-4-flash')}) para expandir casos...")
    try:
        raw_json = call_glm(report, existing_ids, examples_snippet)
    except Exception as e:
        print(f"Erro ao chamar GLM: {e}")
        return 0

    print("Validando e deduplicando casos...")
    recommendations, new_cases = validate_and_extract(raw_json)
    unique_cases = deduplicate(new_cases, existing_ids, existing_messages)
    total_new = sum(len(v) for v in unique_cases.values())
    print(f"  {total_new} novos casos unicos encontrados")

    if total_new == 0:
        print("Nenhum caso novo para adicionar.")
        if recommendations and GH_REPO and GH_SHA:
            post_commit_comment(recommendations, 0)
        elif recommendations:
            print(f"\nRecomendacoes do GLM:\n{recommendations}")
        return 0

    total_added = merge_and_save(eval_cases, unique_cases)

    if args.auto_commit:
        print("Auto-commitando novos casos...")
        git_commit_and_push(total_added)
    else:
        print(f"[dry-run] {total_added} casos prontos (use --auto-commit para persistir)")

    if recommendations:
        if GH_REPO and GH_SHA:
            post_commit_comment(recommendations, total_added)
        else:
            print(f"\nRecomendacoes do GLM:\n{recommendations}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
