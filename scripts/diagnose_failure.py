#!/usr/bin/env python3
"""
Diagnose Sofia CI failure using GPT-4o-mini and post a commit comment.

Executado automaticamente pelo GitHub Actions quando integration_test.py falha.
Lê ci-test-report.txt, chama OpenAI, posta diagnóstico como commit comment.

Variáveis de ambiente necessárias:
  OPENAI_API_KEY  — chave OpenAI
  GH_TOKEN        — token GitHub (GITHUB_TOKEN injetado pelo Actions)
  GH_REPO         — ex: jefersonalvarenga/sofia
  GH_SHA          — SHA do commit que disparou o workflow
"""

import os
import subprocess
import sys

REPORT_FILE = os.getenv("CI_REPORT_FILE", "ci-test-report.txt")
GH_REPO = os.getenv("GH_REPO", "")
GH_SHA = os.getenv("GH_SHA", "")
GITHUB_RUN_ID = os.getenv("GITHUB_RUN_ID", "")

SYSTEM_PROMPT = """\
Você é um engenheiro sênior especialista em FastAPI, DSPy e LangGraph.

A Sofia é um agente conversacional de atendimento de pacientes (WhatsApp) \
com os seguintes agentes: Router (classifica intenção), FAQResponder, \
Scheduler (agendamento multi-etapa), HumanEscalation (determinístico). \
O fluxo é orquestrado via LangGraph. O banco é Supabase (PostgreSQL).

Analise o relatório de falha do CI e forneça em markdown:
1. **Causa raiz** — 1-2 frases diretas
2. **Correção sugerida** — arquivo e trecho de código se possível
3. **Como reproduzir localmente** — comando curto

Seja objetivo. Máximo 350 palavras.\
"""


def read_report() -> str:
    if not os.path.exists(REPORT_FILE):
        return ""
    with open(REPORT_FILE) as f:
        return f.read().strip()


def call_gpt(report: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Relatório de falha:\n\n```\n{report}\n```"},
        ],
        max_tokens=600,
        temperature=0.2,
    )
    return response.choices[0].message.content


def post_commit_comment(body: str) -> bool:
    result = subprocess.run(
        [
            "gh", "api",
            f"repos/{GH_REPO}/commits/{GH_SHA}/comments",
            "-f", f"body={body}",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "GH_TOKEN": os.getenv("GH_TOKEN", "")},
    )
    if result.returncode == 0:
        print(f"Comentario postado no commit {GH_SHA[:8]}.")
        return True
    else:
        print(f"Falha ao postar comentario: {result.stderr}")
        return False


def main() -> int:
    report = read_report()
    if not report:
        print(f"Arquivo {REPORT_FILE} nao encontrado ou vazio. Nada a diagnosticar.")
        return 0

    if not GH_REPO or not GH_SHA:
        print("GH_REPO ou GH_SHA ausente — pulando post de comentario.")
        print("Diagnostico local:\n")
        diagnosis = call_gpt(report)
        print(diagnosis)
        return 0

    print("Chamando GPT-4o-mini para diagnosticar a falha...")
    diagnosis = call_gpt(report)

    run_url = f"https://github.com/{GH_REPO}/actions/runs/{GITHUB_RUN_ID}"
    comment = (
        "## Sofia CI — Diagnóstico de Falha 🔍\n\n"
        f"{diagnosis}\n\n"
        "---\n"
        f"_Gerado por `gpt-4o-mini` · commit `{GH_SHA[:8]}` · "
        f"[ver log completo]({run_url})_"
    )

    post_commit_comment(comment)
    return 0


if __name__ == "__main__":
    sys.exit(main())
