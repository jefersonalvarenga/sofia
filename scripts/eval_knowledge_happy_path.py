"""
Evaluation harness — KnowledgeAgent HAPPY PATH dimension.

Tests the agent on 20 realistic patient questions covering the seeded
procedures of Clínica Bloom (Botox, Limpeza de Pele, Preenchimento Labial,
Sculptra, Fio de PDO, Microagulhamento).

Verdict via LLM-as-judge (DeepSeek V4 Pro, in a separate stateless call).
The judge sees the question, the KB chunks that were available, and the
agent's answer, and returns YES/NO + reasoning. Criteria for YES:

  1. The answer is grounded in the KB chunks (no hallucination outside them)
  2. The answer addresses the patient's question (not evasive)
  3. The answer is in pt-BR, no markdown, reasonable length

The agent's sensitive_flag/requires_consultation should be False in this
eval (those are tested in edge_cases). If they are True, the case is NO.

Usage:
    cd easyscale-sofia
    PYTHONPATH=. EVAL_ROUND_LABEL="round 1" python scripts/eval_knowledge_happy_path.py

Tenant used: Clínica Bloom (id 0d6d8eaf-6efa-4aaf-9845-de4b0d0f608c).
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import dspy  # noqa: E402

from app.agents.knowledge.agent import (  # noqa: E402
    KnowledgeAgent,
    KNOWLEDGE_MODEL,
    KNOWLEDGE_TEMPERATURE,
    KNOWLEDGE_MAX_TOKENS,
    _retrieve,
    _build_context,
    _is_sensitive,
)


# Seeded clinic (see scripts/index_procedure_kb.py output / migration 020 logic).
TEST_TENANT_ID = "0d6d8eaf-6efa-4aaf-9845-de4b0d0f608c"
TEST_CLINIC_NAME = "Clínica Bloom"


# ============================================================================
# Test cases — 20 happy path questions covering the 6 seeded procedures
# ============================================================================

CASES: List[Dict[str, Any]] = [
    # --- Botox (3 chunks: definição, contraindicações, pós) ---
    {"id": "HP.01", "question": "quanto tempo dura o efeito do botox?"},
    {"id": "HP.02", "question": "em quantos dias começo a ver resultado do botox?"},
    {"id": "HP.03", "question": "o que não pode fazer depois de aplicar botox?"},
    {"id": "HP.04", "question": "posso fazer exercício depois do botox?"},
    # --- Limpeza de Pele (1 chunk) ---
    {"id": "HP.05", "question": "quanto tempo dura uma sessão de limpeza de pele?"},
    {"id": "HP.06", "question": "de quanto em quanto tempo precisa fazer limpeza de pele?"},
    {"id": "HP.07", "question": "como é feita a limpeza de pele profunda?"},
    # --- Preenchimento Labial (2 chunks: aplicação, contraindicações) ---
    {"id": "HP.08", "question": "qual produto vocês usam no preenchimento labial?"},
    {"id": "HP.09", "question": "quanto tempo dura o preenchimento labial?"},
    {"id": "HP.10", "question": "quanto tempo demora a aplicação do preenchimento labial?"},
    {"id": "HP.11", "question": "fica inchado depois do preenchimento labial?"},
    # --- Sculptra (1 chunk) ---
    {"id": "HP.12", "question": "para que serve o sculptra?"},
    {"id": "HP.13", "question": "quantas sessões de sculptra preciso fazer?"},
    {"id": "HP.14", "question": "quanto tempo dura o efeito do sculptra?"},
    # --- Fio de PDO (1 chunk) ---
    {"id": "HP.15", "question": "o fio de PDO precisa ser retirado?"},
    {"id": "HP.16", "question": "para que servem os fios de PDO?"},
    # --- Microagulhamento (2 chunks: protocolo, contraindicações) ---
    {"id": "HP.17", "question": "quantas sessões de microagulhamento são necessárias?"},
    {"id": "HP.18", "question": "o microagulhamento serve para cicatriz de acne?"},
    {"id": "HP.19", "question": "quanto tempo a pele fica vermelha depois do microagulhamento?"},
    # --- Pergunta cruzada (pode envolver mais de um chunk) ---
    {"id": "HP.20", "question": "qual procedimento ajuda em flacidez?"},
]


# ============================================================================
# LLM judge (DeepSeek V4 Pro)
# ============================================================================

JUDGE_MODEL = "deepseek/deepseek-v4-pro"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 256

JUDGE_SYSTEM_PROMPT = """Você é um juiz que avalia respostas de um agente de RAG médico-estético.

Você recebe:
- A pergunta do paciente.
- Os chunks da base de conhecimento (KB) que o agente teve acesso.
- A resposta que o agente deu.

Avalie se a resposta é ACEITÁVEL. Critérios para YES:
1. Grounded: tudo que a resposta afirma deve estar contido (literalmente ou parafraseado) nos chunks. Não pode inventar dados.
2. Útil: responde à pergunta do paciente — não é evasiva nem genérica.
3. Forma: em pt-BR, sem markdown (sem listas, sem **negrito**, sem #headers), curta a moderada.

Critérios para NO (qualquer um já reprova):
- Inventou dado que não está nos chunks (alucinação)
- Não respondeu o que foi perguntado (evasiva, mudou de assunto)
- Usou markdown ou outro idioma
- Está claramente errada conforme o chunk

OUTPUT OBRIGATÓRIO: JSON com EXATAMENTE 2 campos no nível raiz:
- "verdict" (string): "YES" ou "NO"
- "reasoning" (string, <= 300 chars): por que.

Exemplo:
{"verdict":"YES","reasoning":"resposta cita os 4-6 meses do chunk 1, em pt-BR, sem markdown, responde direto"}
{"verdict":"NO","reasoning":"resposta diz que dura 1 ano, mas o chunk diz 4-6 meses"}

Responda APENAS JSON válido."""


_JUDGE_LM: Any = None


def _get_judge_lm() -> Any:
    global _JUDGE_LM
    if _JUDGE_LM is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY required for the eval judge.")
        _JUDGE_LM = dspy.LM(
            model=JUDGE_MODEL,
            api_key=api_key,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
        )
        _JUDGE_LM.cache = False
    return _JUDGE_LM


def judge(question: str, context: str, answer: str) -> Dict[str, str]:
    """Run the LLM judge. Returns {'verdict': 'YES'|'NO', 'reasoning': str}."""
    lm = _get_judge_lm()
    user_prompt = (
        f"Pergunta:\n{question}\n\n"
        f"Chunks da KB (contexto disponível):\n{context}\n\n"
        f"Resposta do agente:\n{answer}\n\n"
        f"Avalie."
    )
    try:
        outputs = lm(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        if not outputs:
            return {"verdict": "NO", "reasoning": "judge returned empty"}
        payload = json.loads(outputs[0])
        verdict = str(payload.get("verdict") or "").upper().strip()
        if verdict not in ("YES", "NO"):
            verdict = "NO"
        reasoning = str(payload.get("reasoning") or "")[:300]
        return {"verdict": verdict, "reasoning": reasoning}
    except Exception as exc:
        return {"verdict": "NO", "reasoning": f"judge error: {type(exc).__name__}: {exc}"}


# ============================================================================
# Per-case verdict — combines hard checks + LLM judge
# ============================================================================


def auto_verdict(case: Dict[str, Any], result: Dict[str, Any], judge_out: Dict[str, str]) -> str:
    """Hard checks first, then trust the judge.

    Hard fails:
      - sensitive_flag=True (those go to edge_cases, not happy_path)
      - requires_consultation=True (same)
      - empty answer
    """
    answer = (result.get("messages") or [{}])[0].get("content") or ""
    if not answer.strip():
        return "NO"

    data = result.get("data") or {}
    if data.get("sensitive_flag") or data.get("requires_consultation"):
        return "NO"

    return judge_out.get("verdict", "NO")


# ============================================================================
# Report
# ============================================================================


def _render_markdown(cases: List[Dict[str, Any]], latencies: List[float]) -> str:
    lines: List[str] = []
    lines.append("# Avaliação HAPPY PATH — KnowledgeAgent")
    lines.append("")
    lines.append(f"- **Modelo (agente):** `{KNOWLEDGE_MODEL}`")
    lines.append(f"- **Temperature:** `{KNOWLEDGE_TEMPERATURE}`")
    lines.append(f"- **max_tokens:** `{KNOWLEDGE_MAX_TOKENS}`")
    lines.append(f"- **Modelo (judge):** `{JUDGE_MODEL}` temp `{JUDGE_TEMPERATURE}`")
    lines.append(f"- **Tenant:** `{TEST_TENANT_ID}` ({TEST_CLINIC_NAME})")
    lines.append(f"- **Casos:** {len(cases)}")
    sorted_lats = sorted(latencies)
    p50 = sorted_lats[len(sorted_lats) // 2]
    p99 = sorted_lats[int(len(sorted_lats) * 0.99)] if len(sorted_lats) > 99 else max(sorted_lats)
    lines.append(
        f"- **Latência agente:** min={min(latencies):.0f}ms  "
        f"p50={p50:.0f}ms  p99={p99:.0f}ms  max={max(latencies):.0f}ms"
    )
    lines.append("")
    lines.append("## Critério")
    lines.append("")
    lines.append("- LLM-as-judge (DeepSeek Pro): grounded + útil + forma correta.")
    lines.append("- Hard fail: sensitive_flag=True, requires_consultation=True, ou answer vazia.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for case in cases:
        lines.append(f"### {case['id']} — {case['question']}")
        lines.append("")
        lines.append(f"_Latência: {case['_elapsed_ms']:.0f}ms_  chunks={case['_chunk_count']}")
        lines.append("")
        lines.append(f"**Answer:** {case['_answer']!r}")
        lines.append("")
        lines.append(f"**Sources:** {case['_sources']}")
        lines.append("")
        lines.append(f"**sensitive_flag:** `{case['_sensitive_flag']}`  requires_consultation: `{case['_requires_consultation']}`")
        lines.append("")
        lines.append("**Context (chunks vistos):**")
        lines.append("")
        lines.append("```")
        lines.append(case["_context"][:2000])
        if len(case["_context"]) > 2000:
            lines.append("...(truncated)...")
        lines.append("```")
        lines.append("")
        lines.append(f"**Judge verdict:** `{case['_judge']['verdict']}` — {case['_judge']['reasoning']}")
        lines.append("")
        lines.append(f"**Veredito final:** `{case['_auto_verdict']}`")
        lines.append("")
        lines.append("**Veredito humano:** [ ] YES  [ ] NO — motivo: ___")
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
    agent = KnowledgeAgent()
    _lm = agent._get_lm()
    if _lm is not None:
        _lm.cache = False

    print("=" * 100)
    print(f"KnowledgeAgent — HAPPY PATH eval ({len(CASES)} cases)")
    print(f"Agent: {agent.model}, temp={agent.temperature}, max_tokens={agent.max_tokens}")
    print(f"Judge: {JUDGE_MODEL}, temp={JUDGE_TEMPERATURE}")
    print(f"Tenant: {TEST_TENANT_ID} ({TEST_CLINIC_NAME})")
    print("=" * 100)

    latencies: List[float] = []

    for i, case in enumerate(CASES, 1):
        question = case["question"]

        # Capture chunks so judge sees the same context the agent saw.
        sensitive_hint = _is_sensitive(question)
        chunks = _retrieve(TEST_TENANT_ID, question, sensitive_hint, agent.top_k)
        context = _build_context(chunks)

        t0 = time.perf_counter()
        try:
            result = agent.forward(
                question=question,
                clinic_name=TEST_CLINIC_NAME,
                tenant_id=TEST_TENANT_ID,
            )
        except Exception as exc:
            result = {
                "messages": [{"type": "text", "content": ""}],
                "conversation_stage": "knowledge",
                "reasoning": f"EXCEPTION: {type(exc).__name__}: {exc}",
                "data": {
                    "answer": "",
                    "sources": [],
                    "requires_consultation": False,
                    "sensitive_flag": False,
                    "chunk_count": 0,
                },
            }
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        answer = (result.get("messages") or [{}])[0].get("content") or ""
        judge_out = judge(question, context, answer)

        case["_answer"] = answer
        case["_sources"] = (result.get("data") or {}).get("sources") or []
        case["_sensitive_flag"] = (result.get("data") or {}).get("sensitive_flag")
        case["_requires_consultation"] = (result.get("data") or {}).get("requires_consultation")
        case["_chunk_count"] = (result.get("data") or {}).get("chunk_count") or len(chunks)
        case["_context"] = context
        case["_judge"] = judge_out
        case["_elapsed_ms"] = elapsed
        case["_auto_verdict"] = auto_verdict(case, result, judge_out)

        verdict_marker = "✓" if case["_auto_verdict"] == "YES" else "✗"
        print(
            f"[{i:>2}] {verdict_marker} {case['id']:<6}  ({elapsed:.0f}ms)  "
            f"judge={judge_out['verdict']}  chunks={case['_chunk_count']}"
        )
        print(f"      Q: {question}")
        print(f"      A: {answer[:120]}{'…' if len(answer) > 120 else ''}")
        if case["_auto_verdict"] == "NO":
            print(f"      WHY: {judge_out['reasoning'][:200]}")

    print()
    print("=" * 100)
    sorted_lats = sorted(latencies)
    p50 = sorted_lats[len(sorted_lats) // 2]
    p99 = sorted_lats[int(len(sorted_lats) * 0.99)] if len(sorted_lats) > 99 else max(sorted_lats)
    print(f"Latency (agent): min={min(latencies):.0f}ms  p50={p50:.0f}ms  p99={p99:.0f}ms  max={max(latencies):.0f}ms")
    auto_yes = sum(1 for c in CASES if c["_auto_verdict"] == "YES")
    print(f"Auto-score: {auto_yes}/{len(CASES)} YES")
    print("=" * 100)

    folder = os.path.expanduser(
        "~/Documents/easyscale/kb/07-MVP/Tech/Tests/Knowledge Agent"
    )
    os.makedirs(folder, exist_ok=True)
    date_tag = time.strftime("%Y-%m-%d")
    round_label = os.environ.get("EVAL_ROUND_LABEL", f"run {time.strftime('%H%M%S')}")
    default_path = os.path.join(folder, f"happy path - {round_label} ({date_tag}).md")
    out_path = os.environ.get("EVAL_REPORT_PATH", default_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(CASES, latencies))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    run_eval()
