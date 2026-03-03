#!/usr/bin/env python3
"""
Sofia Improve Agents — loop local de auto-melhoria via GLM.

Lê ci-eval-report.txt, chama GLM-4-plus, propõe melhorias nas signatures DSPy
dos 3 agentes e opcionalmente valida o impacto rodando eval_agents.py.

Uso:
  python scripts/improve_agents.py [--dry-run] [--confirm] [--eval-first]
                                   [--loop N] [--target-score F] [--expand]

Variáveis de ambiente:
  GLM_API_KEY    — API key ZhipuAI (obrigatório)
  GLM_MODEL      — modelo GLM (default: glm-5)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# ─── Caminhos ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_REPORT_FILE = REPO_ROOT / os.getenv("CI_EVAL_REPORT_FILE", "ci-eval-report.txt")
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_agents.py"
EXPAND_SCRIPT = REPO_ROOT / "scripts" / "expand_eval_cases.py"

SIGNATURE_FILES: Dict[str, Path] = {
    "router_signature": REPO_ROOT / "app" / "agents" / "router" / "signatures.py",
    "scheduler_signature": REPO_ROOT / "app" / "agents" / "scheduler" / "signatures.py",
    "faq_signature": REPO_ROOT / "app" / "agents" / "faq_responder" / "signatures.py",
}

# ─── Prompt do GLM ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Você é especialista em DSPy 2.5 e agentes conversacionais de saúde.

Em DSPy, o docstring da classe Signature é a instrução principal do LLM.
As descrições dos campos OutputField e InputField guiam o formato e conteúdo do output.
Pequenas mudanças nestas strings têm grande impacto no comportamento.

Você receberá:
1. Os arquivos signatures.py atuais dos 3 agentes (Router, Scheduler, FAQResponder)
2. Um relatório mostrando quais casos falharam e por quê

Sua tarefa:
- Identificar as instruções que causaram as falhas
- Propor versões melhoradas dos docstrings e field descriptions
- Retornar os 3 arquivos completos + explicação detalhada das mudanças

Restrições OBRIGATÓRIAS:
- NÃO altere nomes de classes, campos (dspy.InputField/OutputField) ou imports
- NÃO altere a estrutura do arquivo — apenas strings de instrução (docstrings e desc="...")
- Mantenha o português brasileiro nas instruções voltadas ao LLM
- Cada instrução deve ser específica e acionável para o LLM
- O código Python deve ser válido e compilável

Responda SOMENTE com JSON válido neste formato exato:
{
  "router_signature": "conteúdo completo do arquivo signatures.py do Router",
  "scheduler_signature": "conteúdo completo do arquivo signatures.py do Scheduler",
  "faq_signature": "conteúdo completo do arquivo signatures.py do FAQResponder",
  "explanation": "markdown: o que foi alterado e por quê, por agente"
}
"""


# ─── Leitura e extração ───────────────────────────────────────────────────────

def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_score_from_report(report: str) -> Optional[float]:
    """Extrai AVG SCORE da linha 'AVG SCORE : X.XX' do ci-eval-report.txt."""
    match = re.search(r"AVG SCORE\s*:\s*([\d.]+)", report)
    if match:
        return float(match.group(1))
    return None


def read_report() -> str:
    if not EVAL_REPORT_FILE.exists():
        return ""
    return EVAL_REPORT_FILE.read_text(encoding="utf-8")


# ─── Subprocessos ─────────────────────────────────────────────────────────────

def run_eval() -> Tuple[float, str]:
    """
    Executa eval_agents.py.
    Retorna (score, report_content). score=0.0 em caso de erro.
    """
    print("  Rodando eval_agents.py...", flush=True)
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT)],
        capture_output=False,   # deixa o output fluir para o terminal
        cwd=str(REPO_ROOT),
    )
    if result.returncode not in (0, 1):
        print(f"  AVISO: eval_agents.py retornou código {result.returncode}")

    report = read_report()
    score = extract_score_from_report(report) or 0.0
    return score, report


def run_expand() -> None:
    """Executa expand_eval_cases.py sem --auto-commit."""
    print("  Rodando expand_eval_cases.py...", flush=True)
    result = subprocess.run(
        [sys.executable, str(EXPAND_SCRIPT)],
        capture_output=False,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        print(f"  AVISO: expand_eval_cases.py retornou código {result.returncode}")


# ─── GLM ──────────────────────────────────────────────────────────────────────

def call_glm(report: str) -> dict:
    """Chama GLM com as signatures + relatório de falhas. Retorna dict parseado."""
    from zhipuai import ZhipuAI  # type: ignore

    api_key = os.environ.get("GLM_API_KEY")
    if not api_key:
        raise RuntimeError("GLM_API_KEY não configurada.")

    client = ZhipuAI(api_key=api_key)
    model = os.getenv("GLM_MODEL", "glm-5")

    # Monta bloco com conteúdo das 3 signatures
    sig_blocks = []
    for key, path in SIGNATURE_FILES.items():
        agent_label = key.replace("_signature", "").replace("_", " ").title()
        sig_blocks.append(f"=== {agent_label} ({path.name}) ===\n{read_file(path)}")

    user_content = (
        "Arquivos signatures.py atuais:\n\n"
        + "\n\n".join(sig_blocks)
        + f"\n\n---\n\nRelatório de falhas do eval:\n\n```\n{report}\n```\n\n"
        "Analise as falhas, melhore as instruções e retorne os 3 arquivos + explicação."
    )

    print(f"  Chamando GLM ({model})...", flush=True)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_tokens=4096,
        temperature=0.3,
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  ERRO: JSON inválido do GLM: {e}")
        print(f"  Resposta bruta (primeiros 500 chars): {raw[:500]}")
        return {}


# ─── Validação e aplicação ────────────────────────────────────────────────────

def validate_python_syntax(code: str, filename: str) -> Optional[str]:
    """Valida sintaxe Python. Retorna mensagem de erro ou None se OK."""
    try:
        compile(code, filename, "exec")
        return None
    except SyntaxError as e:
        return f"SyntaxError em {filename} linha {e.lineno}: {e.msg}"


def show_string_diffs(path: Path, old: str, new: str) -> None:
    """Mostra linhas que mudaram (somente strings de instrução)."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    shown = 0
    max_diffs = 6

    for i, (ol, nl) in enumerate(zip(old_lines, new_lines)):
        if ol != nl and shown < max_diffs:
            print(f"    L{i+1:3d} - {ol.strip()[:90]}")
            print(f"    L{i+1:3d} + {nl.strip()[:90]}")
            shown += 1

    added = max(0, len(new_lines) - len(old_lines))
    removed = max(0, len(old_lines) - len(new_lines))
    changed = sum(1 for a, b in zip(old_lines, new_lines) if a != b)
    print(f"    ({changed} linhas alteradas, +{added} adicionadas, -{removed} removidas)")


def backup_signatures() -> Dict[str, str]:
    """Cria cópias .bak das signatures. Retorna {key: conteúdo_original}."""
    originals: Dict[str, str] = {}
    for key, path in SIGNATURE_FILES.items():
        originals[key] = read_file(path)
        bak = path.with_suffix(".py.bak")
        shutil.copy2(path, bak)
        print(f"  backup: {bak.relative_to(REPO_ROOT)}")
    return originals


def restore_backups() -> None:
    """Restaura arquivos .bak."""
    for key, path in SIGNATURE_FILES.items():
        bak = path.with_suffix(".py.bak")
        if bak.exists():
            shutil.copy2(bak, path)
            print(f"  restaurado: {path.relative_to(REPO_ROOT)}")
        else:
            print(f"  AVISO: backup não encontrado para {path.name}")


def extract_valid_fixes(glm_data: dict) -> Dict[str, str]:
    """
    Valida sintaxe de cada signature proposta pelo GLM.
    Retorna somente as que passaram na validação.
    """
    valid: Dict[str, str] = {}
    for key, path in SIGNATURE_FILES.items():
        new_content = glm_data.get(key, "")
        if not new_content or not isinstance(new_content, str):
            print(f"  AVISO: GLM não retornou conteúdo para '{key}'. Mantendo original.")
            continue
        err = validate_python_syntax(new_content, path.name)
        if err:
            print(f"  ERRO de sintaxe — {err}. Mantendo {path.name} original.")
            continue
        # Verifica que o conteúdo realmente mudou
        old_content = read_file(path)
        if new_content.strip() == old_content.strip():
            print(f"  INFO: GLM não alterou {path.name} (idêntico ao original).")
            continue
        valid[key] = new_content
    return valid


def apply_fixes_dry_run(valid_fixes: Dict[str, str]) -> None:
    """Exibe as mudanças propostas sem escrever."""
    if not valid_fixes:
        print("  Nenhuma mudança válida a exibir.")
        return

    print("\n[dry-run] Mudanças propostas (não serão escritas):\n")
    for key, new_content in valid_fixes.items():
        path = SIGNATURE_FILES[key]
        old_content = read_file(path)
        print(f"  === {path.relative_to(REPO_ROOT)} ===")
        show_string_diffs(path, old_content, new_content)
        print()


def apply_fixes_write(valid_fixes: Dict[str, str]) -> None:
    """Escreve as novas signatures em disco."""
    for key, new_content in valid_fixes.items():
        path = SIGNATURE_FILES[key]
        old_content = read_file(path)
        print(f"  Aplicando {path.relative_to(REPO_ROOT)}:")
        show_string_diffs(path, old_content, new_content)
        path.write_text(new_content, encoding="utf-8")
        print(f"  Escrito: {path.relative_to(REPO_ROOT)}")


# ─── Iteração ─────────────────────────────────────────────────────────────────

def run_one_iteration(
    iteration: int,
    max_iterations: int,
    score_before: float,
    args: argparse.Namespace,
) -> Tuple[bool, float]:
    """
    Executa uma iteração do ciclo: GLM → validação → [confirm].
    Retorna (improved, score_after).
    """
    label = f"[iter {iteration}/{max_iterations}]"
    print(f"\n{label} Score antes: {score_before:.2f}", flush=True)

    # 1. Lê relatório atual
    report = read_report()
    if not report:
        print(f"{label} ERRO: {EVAL_REPORT_FILE} não encontrado. Execute eval_agents.py primeiro.")
        return False, score_before

    # 2. Chama GLM
    try:
        glm_data = call_glm(report)
    except Exception as e:
        print(f"{label} ERRO ao chamar GLM: {e}")
        return False, score_before

    if not glm_data:
        print(f"{label} GLM retornou resposta vazia. Parando.")
        return False, score_before

    # 3. Exibe explicação
    explanation = glm_data.get("explanation", "")
    if explanation:
        print(f"\n{label} Explicação do GLM:\n")
        # Imprime até 1000 chars
        print(explanation[:1000])
        if len(explanation) > 1000:
            print("  ... (truncado)")
        print()

    # 4. Valida as fixes
    valid_fixes = extract_valid_fixes(glm_data)
    if not valid_fixes:
        print(f"{label} GLM não propôs mudanças válidas.")
        return False, score_before

    # 5. dry-run: apenas mostra
    if args.dry_run:
        apply_fixes_dry_run(valid_fixes)
        return True, score_before

    # 6. Cria backups e aplica
    backup_signatures()
    apply_fixes_write(valid_fixes)

    # 7. Se --confirm, re-roda eval e compara
    if args.confirm:
        print(f"\n{label} Re-rodando eval para confirmar melhoria...")
        score_after, _ = run_eval()
        delta = score_after - score_before
        sign = "+" if delta >= 0 else ""
        icon = "✓" if delta >= 0 else "✗"
        print(f"{label} Score depois: {score_after:.2f} {icon} ({sign}{delta:.2f})")

        if score_after < score_before:
            print(f"{label} Score piorou ({score_before:.2f} → {score_after:.2f}). Revertendo backups...")
            restore_backups()
            return False, score_before
        else:
            print(f"{label} Melhoria confirmada. Mantendo mudanças.")
            return True, score_after
    else:
        print(f"{label} Mudanças aplicadas (sem --confirm, score não re-verificado).")
        return True, score_before


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sofia Improve Agents — loop de auto-melhoria de signatures DSPy via GLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Mostra mudanças propostas sem aplicar
  python scripts/improve_agents.py --dry-run

  # Aplica + re-valida (reverte se piorou)
  python scripts/improve_agents.py --confirm

  # Loop automático até score 0.85, expandindo casos a cada rodada
  python scripts/improve_agents.py --loop 5 --target-score 0.85 --confirm --expand
""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exibe mudanças propostas sem escrever arquivos",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Aplica mudanças, re-roda eval e reverte se score piorou",
    )
    parser.add_argument(
        "--eval-first",
        action="store_true",
        help="Roda eval_agents.py antes de chamar o GLM (atualiza ci-eval-report.txt)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=1,
        metavar="N",
        help="Número máximo de iterações (default: 1)",
    )
    parser.add_argument(
        "--target-score",
        type=float,
        default=0.85,
        metavar="F",
        help="Para o loop antecipado se este score for atingido (default: 0.85)",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Roda expand_eval_cases.py ao final de cada iteração",
    )
    args = parser.parse_args()

    # Valida pré-requisitos
    if not os.getenv("GLM_API_KEY") and not args.dry_run:
        # dry-run ainda pode mostrar as signatures atuais sem chamar o GLM
        # mas para chamar o GLM precisamos da chave
        pass  # será verificado antes de chamar call_glm

    glm_model = os.getenv("GLM_MODEL", "glm-5")
    print("Sofia Improve Agents")
    print(f"  loop={args.loop}  target={args.target_score}  confirm={args.confirm}"
          f"  dry_run={args.dry_run}  expand={args.expand}")
    print(f"  modelo GLM : {glm_model}")
    print(f"  report     : {EVAL_REPORT_FILE}")

    # Validação de GLM_API_KEY (necessária para chamadas reais)
    if not os.getenv("GLM_API_KEY"):
        print("\nERRO: GLM_API_KEY não configurada.")
        print("  Configure com: export GLM_API_KEY=<sua-chave-zhipuai>")
        return 1

    # Roda eval inicial se solicitado ou se loop > 1 e não há report
    run_eval_at_start = args.eval_first or (args.loop > 1 and not EVAL_REPORT_FILE.exists())
    if run_eval_at_start:
        print("\n[pre-loop] Rodando eval_agents.py para obter score base...")
        score_current, _ = run_eval()
        print(f"[pre-loop] Score base: {score_current:.2f}")
    else:
        # Tenta ler score do report existente
        existing_report = read_report()
        if existing_report:
            score_current = extract_score_from_report(existing_report) or 0.0
            print(f"\n[pre-loop] Score lido do report existente: {score_current:.2f}")
        else:
            print(
                "\nAVISO: ci-eval-report.txt não encontrado. "
                "Execute eval_agents.py primeiro ou use --eval-first."
            )
            if not args.dry_run:
                return 1
            score_current = 0.0

    max_iterations = args.loop
    target_score = args.target_score

    # ─── Loop principal ───────────────────────────────────────────────────────
    for iteration in range(1, max_iterations + 1):

        # Verifica se já atingiu o target (antes de cada iteração)
        # dry-run ignora o target — o objetivo é inspecionar as propostas do GLM
        if not args.dry_run and score_current >= target_score:
            print(f"\nTarget {target_score} atingido antes da iteração {iteration}. Parando.")
            break

        # Para iterações > 1 com --confirm: roda eval para obter score fresco
        if iteration > 1 and args.confirm:
            print(f"\n[iter {iteration}/{max_iterations}] Rodando eval para score fresco...")
            score_current, _ = run_eval()

        # Executa a iteração
        improved, score_current = run_one_iteration(
            iteration=iteration,
            max_iterations=max_iterations,
            score_before=score_current,
            args=args,
        )

        # Expande casos de eval se solicitado (e não é dry-run)
        if args.expand and not args.dry_run:
            print(f"\n[iter {iteration}/{max_iterations}] Expandindo casos de eval...")
            run_expand()

        # Para o loop se o GLM não conseguiu melhorar
        if not improved and not args.dry_run:
            print(f"\nGLM não conseguiu melhorar na iteração {iteration}. Parando loop.")
            break

        # dry-run: executa apenas uma vez
        if args.dry_run:
            break

    # ─── Resultado final ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    if score_current >= target_score:
        print(f"Score final: {score_current:.2f} ✓  (target {target_score} atingido)")
    else:
        print(f"Score final: {score_current:.2f}  (target {target_score} não atingido)")
    print("=" * 55)

    return 0


if __name__ == "__main__":
    sys.exit(main())
