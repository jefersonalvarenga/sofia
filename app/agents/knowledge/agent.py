"""
KnowledgeAgent — RAG sobre sf_procedure_kb (pgvector).

Responde perguntas técnicas de pacientes sobre procedimentos estéticos usando
DeepSeek V4 Pro como modelo de geração. Stack consistente com os outros agentes
da Iris (greeting, router, schedule-router) — o agent auto-gerencia seu LM
via DEEPSEEK_API_KEY e não depende de init_dspy() global.

Retrieval híbrido (mesma lógica preservada do design original):
  1. Pgvector cosine similarity via RPC `match_procedure_kb` (preferido).
  2. Fallback ILIKE em title/body quando embeddings ainda não foram indexados
     ou a query embedding falha.

Guardrails médicos:
  - Regex de termos sensíveis (`_SENSITIVE_RE`) detecta gravidez, anticoagulante,
    doenças crônicas etc. Marca `sensitive_flag=True` e força CTA de avaliação
    presencial.
  - Modelo é instruído a NUNCA diagnosticar / prescrever, e a só usar info do
    contexto retornado pelo RAG (não pode alucinar fora dos chunks).
  - Se a base não tem informação relevante, agente diz isso honestamente e
    pergunta se o paciente quer falar com a equipe.

Output envelope segue padrão Greeting/Router/Schedule (messages + data),
para o pipeline consumir de forma uniforme.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import dspy
from pydantic import BaseModel, Field, ValidationError

from app.core.config import get_settings
from app.core.supabase_client import get_supabase
from app.core.telemetry import log


# ============================================================================
# Configuration
# ============================================================================

KNOWLEDGE_MODEL = "deepseek/deepseek-v4-pro"
KNOWLEDGE_TEMPERATURE = 0.2  # baixa mas não-zero — respostas técnicas com leveza
KNOWLEDGE_MAX_TOKENS = 512  # margem confortável para resposta + sources + flags

# DeepSeek thinking-mode disabled for low-latency single-turn answer.
KNOWLEDGE_EXTRA_BODY: Dict[str, Any] = {"thinking": {"type": "disabled"}}

EMBED_MODEL = "text-embedding-3-small"
TOP_K_DEFAULT = 4
TECHNICAL_FALLBACK = (
    "Desculpa, tive um problema técnico aqui. Quer falar com nossa equipe?"
)


# ============================================================================
# Sensitive-condition regex (preserved from previous implementation)
#
# Triggers `sensitive_flag=True` and forces the in-person CTA even if the LLM
# decides otherwise. Prefixes (not anchored to \b at the end) because most
# Portuguese terms appear inflected (grávida, anticoagulante, etc.).
# ============================================================================

_SENSITIVE_RE = re.compile(
    r"\b(gr[aá]vid|gestant|gestação|lactan|amament|"
    r"anticoagulant|varfarin|warfarin|heparin|aspirina|"
    r"al[eé]rgi|anafilax|"
    r"diabet|hipertensão|pressão alta|"
    r"cardí[ao]c|coração|infarto|"
    r"epilepsi|convuls|"
    r"câncer|cancer|oncolog|quimio|"
    r"transplante|imunossupressor|"
    r"lúpus|lupus|autoimune|esclerose|"
    r"miastenia|parkinson|alzheimer|"
    r"isotretinoína|roacutan)",
    re.IGNORECASE | re.UNICODE,
)


def _is_sensitive(question: str) -> bool:
    """Best-effort detection of clinically sensitive context in the question."""
    return bool(_SENSITIVE_RE.search(question or ""))


# ============================================================================
# Pydantic schema for the LLM output (mirrors what greeting/router use)
# ============================================================================


class KnowledgeOutput(BaseModel):
    answer: str = Field(..., description="Resposta final para o paciente (pt-BR).")
    requires_consultation: bool = Field(
        default=False,
        description="True quando a pergunta exige avaliação presencial.",
    )
    sensitive_flag: bool = Field(
        default=False,
        description="True quando há condição de saúde sensível mencionada.",
    )
    sources_used: List[str] = Field(
        default_factory=list,
        description="Títulos dos chunks consultados (auditoria).",
    )
    reasoning: str = Field(
        default="",
        max_length=400,
        description="Justificativa curta da resposta (debug).",
    )


# ============================================================================
# System prompt — gives the LLM the contract + guardrails
# ============================================================================

SYSTEM_PROMPT = """Você é a Iris, assistente da {clinic_name}, respondendo a uma pergunta técnica do paciente sobre um procedimento estético.

REGRAS NÃO-NEGOCIÁVEIS:
1. Use APENAS as informações fornecidas no bloco "Contexto da base de procedimentos". NUNCA invente dados.
2. Se o Contexto não tem a informação necessária, diga honestamente que não tem essa info disponível e pergunte se o paciente quer falar com a equipe.
3. Nunca diagnostique condições. Nunca prescreva medicamentos ou tratamentos específicos.
4. Se a pergunta menciona condição sensível (gravidez, anticoagulante, doença crônica, alergia, isotretinoína, etc.) -> sensitive_flag=true E requires_consultation=true E inclua na resposta: "isso depende de avaliação presencial com a equipe médica — quer agendar uma consulta?"
5. Responda em pt-BR claro, objetivo e empático. Máximo de 3 frases curtas, exceto quando a info técnica exigir mais.
6. NUNCA use markdown na resposta (sem listas, negritos, headers).

OUTPUT OBRIGATÓRIO: JSON com EXATAMENTE estes 5 campos no nível raiz:
- "answer" (string): a mensagem que vai ao paciente.
- "requires_consultation" (boolean): true se exige avaliação presencial.
- "sensitive_flag" (boolean): true se há condição sensível.
- "sources_used" (array of strings): títulos dos chunks que você usou.
- "reasoning" (string, <= 400 chars): justificativa curta da decisão.

Exemplos de saída:

{{"answer":"O Botox dura em média 4 a 6 meses. O efeito começa a aparecer entre 3 e 7 dias após a aplicação, com resultado final em 7 a 14 dias.","requires_consultation":false,"sensitive_flag":false,"sources_used":["O que é Botox?"],"reasoning":"Pergunta direta sobre duracao, info no contexto chunk 1."}}

{{"answer":"Como você está grávida, isso depende de avaliação presencial com a equipe médica — quer agendar uma consulta?","requires_consultation":true,"sensitive_flag":true,"sources_used":["Quem pode fazer Botox?"],"reasoning":"Gestante mencionada, contraindicacao no chunk 2, escalando."}}

{{"answer":"Não tenho essa informação específica na nossa base agora. Posso te conectar com nossa equipe pra tirar essa dúvida — quer?","requires_consultation":false,"sensitive_flag":false,"sources_used":[],"reasoning":"Pergunta sobre seguro estetico, sem chunks relevantes."}}

Responda APENAS JSON válido com os 5 campos obrigatórios."""


# ============================================================================
# Retrieval (preserved logic — pgvector first, ILIKE fallback)
# ============================================================================


def _embed_query(question: str) -> Optional[List[float]]:
    """Generate embedding via OpenAI text-embedding-3-small. None on failure."""
    try:
        import openai  # lazy import; only used when key present

        api_key = get_settings().openai_api_key
        if not api_key:
            return None
        client = openai.OpenAI(api_key=api_key)
        resp = client.embeddings.create(model=EMBED_MODEL, input=question)
        return resp.data[0].embedding
    except Exception as exc:  # noqa: BLE001 (telemetry-only, fall through to keyword)
        log.warning("knowledge.embed_query.failed", error=str(exc))
        return None


def _retrieve_by_vector(
    tenant_id: str, embedding: List[float], top_k: int
) -> List[Dict[str, Any]]:
    """Pgvector cosine similarity via Supabase RPC."""
    try:
        sb = get_supabase()
        vector_str = "[" + ",".join(str(v) for v in embedding) + "]"
        result = sb.rpc(
            "match_procedure_kb",
            {
                "p_tenant_id": tenant_id,
                "p_embedding": vector_str,
                "p_top_k": top_k,
            },
        ).execute()
        return result.data or []
    except Exception as exc:
        log.warning("knowledge.retrieve_vector.failed", error=str(exc))
        return []


def _retrieve_by_keyword(
    tenant_id: str, question: str, top_k: int
) -> List[Dict[str, Any]]:
    """ILIKE fallback. Used when embeddings missing OR vector retrieval fails."""
    try:
        sb = get_supabase()
        words = [
            w
            for w in re.findall(r"[a-záéíóúâêîôûãẽõüçñ]+", question.lower())
            if len(w) >= 4
        ]
        if not words:
            words = question.lower().split()[:3]

        query = (
            sb.table("sf_procedure_kb")
            .select("procedure, title, body")
            .eq("tenant_id", tenant_id)
        )
        if words:
            filters = [f"body.ilike.%{w}%,title.ilike.%{w}%" for w in words[:3]]
            query = query.or_(",".join(filters))

        result = query.limit(top_k).execute()
        return result.data or []
    except Exception as exc:
        log.warning("knowledge.retrieve_keyword.failed", error=str(exc))
        return []


def _retrieve(
    tenant_id: str, question: str, sensitive: bool, top_k: int
) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks. Vector first; keyword fallback."""
    embedding = _embed_query(question)
    if embedding:
        chunks = _retrieve_by_vector(tenant_id, embedding, top_k)
        if chunks:
            return chunks

    # When sensitive, bias the keyword query toward contraindication chunks.
    augmented = question + " contraindicação gestante alergia" if sensitive else question
    return _retrieve_by_keyword(tenant_id, augmented, top_k)


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks as a single context block for the LLM."""
    if not chunks:
        return "(base de procedimentos sem entradas relevantes para essa pergunta)"
    parts = []
    for c in chunks:
        proc = (c.get("procedure") or "").strip()
        title = (c.get("title") or "").strip()
        body = (c.get("body") or "").strip()
        header = f"{proc} — {title}" if proc and title else (proc or title or "(sem título)")
        parts.append(f"### {header}\n{body}")
    return "\n\n".join(parts)


# ============================================================================
# LM management (same pattern as RouterAgent / GreetingAgent)
# ============================================================================


def _build_default_lm(model: str, max_tokens: int) -> Optional[dspy.LM]:
    """Build the LM the knowledge agent uses by default."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    try:
        return dspy.LM(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=KNOWLEDGE_TEMPERATURE,
        )
    except Exception as exc:
        log.error("knowledge.lm_init_failed", error=str(exc))
        return None


# ============================================================================
# Agent
# ============================================================================


class KnowledgeAgent:
    """Iris knowledge agent — RAG-backed answers on procedure questions.

    Drop-in replacement for the legacy ``KnowledgeSpecialist`` (Anthropic).
    Runs on ``deepseek/deepseek-v4-pro`` (non-thinking). Auto-manages LM via
    ``DEEPSEEK_API_KEY``.

    Usage:
        agent = KnowledgeAgent()
        out = agent.forward(
            question="quanto tempo dura o efeito do botox?",
            clinic_name="Clínica Bloom",
            tenant_id="0d6d8eaf-6efa-4aaf-9845-de4b0d0f608c",
        )
        # out["messages"] -> [{"type": "text", "content": "..."}]
        # out["data"] -> {"answer", "sources", "requires_consultation",
        #                 "sensitive_flag", "routing_hint", ...}
    """

    def __init__(
        self,
        lm: Optional[dspy.LM] = None,
        model: str = KNOWLEDGE_MODEL,
        max_tokens: int = KNOWLEDGE_MAX_TOKENS,
        temperature: float = KNOWLEDGE_TEMPERATURE,
        top_k: int = TOP_K_DEFAULT,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self._lm_override = lm
        self._default_lm: Optional[dspy.LM] = None

    def _get_lm(self) -> dspy.LM:
        if self._lm_override is not None:
            return self._lm_override
        if self._default_lm is None:
            self._default_lm = _build_default_lm(self.model, self.max_tokens)
        if self._default_lm is not None:
            return self._default_lm
        lm = dspy.settings.lm
        if lm is None:
            raise RuntimeError(
                "KnowledgeAgent: no LM available. Set DEEPSEEK_API_KEY, call init_dspy(), "
                "or pass lm= to constructor."
            )
        return lm

    def _build_user_prompt(
        self,
        question: str,
        history: List[Dict[str, str]],
        context: str,
    ) -> str:
        history_block = self._format_history(history)
        return (
            f"Contexto da base de procedimentos:\n{context}\n\n"
            f"Histórico recente (últimos turnos, opcional):\n{history_block}\n\n"
            f"Pergunta do paciente:\n{question}\n\n"
            f"Responda em JSON conforme as regras."
        )

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "(sem histórico)"
        lines = []
        for turn in history[-5:]:
            role = turn.get("role", "?")
            content = turn.get("content", "")
            prefix = "Paciente" if role in ("human", "patient") else role
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def _call_lm(self, system: str, user_prompt: str) -> str:
        lm = self._get_lm()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        call_kwargs: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if KNOWLEDGE_EXTRA_BODY:
            call_kwargs["extra_body"] = KNOWLEDGE_EXTRA_BODY
        outputs = lm(**call_kwargs)
        if not outputs:
            raise ValueError("knowledge LM returned no outputs")
        return outputs[0]

    def _parse(self, raw_content: str) -> KnowledgeOutput:
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"knowledge LM returned non-JSON content: {exc}") from exc
        try:
            return KnowledgeOutput.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"knowledge output failed Pydantic validation: {exc}"
            ) from exc

    def forward(
        self,
        question: str,
        clinic_name: str,
        tenant_id: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        history = history or []
        sensitive = _is_sensitive(question)

        chunks = _retrieve(tenant_id, question, sensitive, self.top_k)
        context = _build_context(chunks)
        chunk_titles = [c.get("title", "") for c in chunks if c.get("title")]

        system_prompt = SYSTEM_PROMPT.format(clinic_name=clinic_name)
        user_prompt = self._build_user_prompt(question, history, context)

        try:
            raw_content = self._call_lm(system_prompt, user_prompt)
            parsed = self._parse(raw_content)
        except Exception as exc:
            log.error(
                "knowledge.forward.failed",
                error=str(exc),
                error_type=type(exc).__name__,
                model=self.model,
            )
            return {
                "messages": [{"type": "text", "content": TECHNICAL_FALLBACK}],
                "conversation_stage": "knowledge",
                "reasoning": f"source=fallback | error={type(exc).__name__}",
                "data": {
                    "answer": TECHNICAL_FALLBACK,
                    "sources": chunk_titles,
                    "requires_consultation": False,
                    "sensitive_flag": sensitive,
                    "routing_hint": None,
                    "chunk_count": len(chunks),
                },
            }

        # Regex-driven guardrail wins over the LLM: if we detected sensitive
        # context, force the flags + append CTA. This protects against the LLM
        # accidentally minimizing a contraindication.
        if sensitive:
            parsed.sensitive_flag = True
            parsed.requires_consultation = True
            cta_phrase = "avaliação presencial"
            if cta_phrase not in parsed.answer.lower():
                parsed.answer = parsed.answer.rstrip() + (
                    " Isso depende de uma avaliação presencial com a nossa equipe "
                    "médica — quer agendar uma consulta?"
                )

        routing_hint = "SCHEDULE_NEXT" if parsed.sensitive_flag else None

        log.info(
            "knowledge.forward.ok",
            model=self.model,
            sensitive=sensitive,
            requires_consultation=parsed.requires_consultation,
            sensitive_flag=parsed.sensitive_flag,
            chunk_count=len(chunks),
        )

        return {
            "messages": [{"type": "text", "content": parsed.answer}],
            "conversation_stage": "knowledge",
            "reasoning": (
                f"source=llm | model={self.model} | sensitive={sensitive} | "
                f"chunks={len(chunks)} | llm_reasoning={parsed.reasoning}"
            ),
            "data": {
                "answer": parsed.answer,
                "sources": parsed.sources_used or chunk_titles,
                "requires_consultation": parsed.requires_consultation,
                "sensitive_flag": parsed.sensitive_flag,
                "routing_hint": routing_hint,
                "chunk_count": len(chunks),
            },
        }


# Back-compat alias — pipeline may still import KnowledgeSpecialist.
KnowledgeSpecialist = KnowledgeAgent
