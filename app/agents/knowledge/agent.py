"""
KnowledgeSpecialist — RAG sobre sf_procedure_kb (pgvector).

Responde perguntas de pacientes sobre procedimentos estéticos.
Guardrails: não diagnostica, não prescreve, escala condições sensíveis.

Retrieval:
  1. Tenta pgvector similarity search se embeddings existem (OpenAI text-embedding-3-small).
  2. Fallback: ILIKE keyword para quando embeddings ainda não foram indexados.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.supabase_client import get_supabase
from app.core.telemetry import log

KNOWLEDGE_HAIKU = "claude-haiku-4-5-20251001"
KNOWLEDGE_SONNET = "claude-sonnet-4-6"

# Conditions that require in-person evaluation — always set sensitive_flag=True.
# Prefixes — no trailing \b because most terms appear as inflected forms
# (grávida, anticoagulante, alérgica, diabetes, etc.).
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
    r"miastenia|parkinson|alzheimer)",
    re.IGNORECASE | re.UNICODE,
)

TOP_K = 4  # max chunks to retrieve

SYSTEM_PROMPT = """Você é a Iris, assistente de IA da {clinic_name}.
Você responde perguntas de pacientes sobre procedimentos estéticos.

GUARDRAILS MÉDICOS (não negociáveis):
- Jamais diagnostique condições médicas.
- Jamais prescreva medicamentos ou tratamentos.
- Se a pergunta menciona condição de saúde sensível (gravidez, medicação, doença crônica, alergia),
  explique que é necessária avaliação presencial com a equipe médica e que o paciente pode agendar.
- Você pode descrever procedimentos: benefícios, duração de efeito, processo geral, cuidados pós.
- Nunca invente informações que não estejam no contexto fornecido.
- Se a informação não estiver no contexto, diga honestamente que não tem essa informação disponível.

Responda em português claro e acessível. Seja objetiva e empática."""

ANSWER_TOOL: Dict[str, Any] = {
    "name": "knowledge_answer",
    "description": "Resposta estruturada sobre procedimento estético com guardrails médicos.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "Resposta clara e objetiva. "
                    "Se sensitive_flag=true, inclua: 'isso depende de avaliação presencial "
                    "— quer agendar uma consulta?'"
                ),
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Títulos dos chunks consultados para formular a resposta.",
            },
            "requires_consultation": {
                "type": "boolean",
                "description": "True se a pergunta requer avaliação presencial para ser adequadamente respondida.",
            },
            "sensitive_flag": {
                "type": "boolean",
                "description": (
                    "True se a pergunta menciona condição de saúde sensível "
                    "(gravidez, medicação, doença crônica, alergia)."
                ),
            },
        },
        "required": ["answer", "sources", "requires_consultation", "sensitive_flag"],
    },
}


class KnowledgeOutput(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    requires_consultation: bool = False
    sensitive_flag: bool = False


def _is_sensitive(question: str) -> bool:
    return bool(_SENSITIVE_RE.search(question))


def _embed_query(question: str) -> Optional[List[float]]:
    """Generate embedding via OpenAI text-embedding-3-small. Returns None on failure."""
    try:
        import openai  # optional dep; only used if key present
        api_key = get_settings().openai_api_key
        if not api_key:
            return None
        client = openai.OpenAI(api_key=api_key)
        resp = client.embeddings.create(model="text-embedding-3-small", input=question)
        return resp.data[0].embedding
    except Exception as exc:
        log.warning("knowledge.embed_query.failed", error=str(exc))
        return None


def _retrieve_by_vector(
    tenant_id: str, embedding: List[float], top_k: int
) -> List[Dict[str, str]]:
    """Pgvector cosine similarity search."""
    try:
        sb = get_supabase()
        vector_str = "[" + ",".join(str(v) for v in embedding) + "]"
        result = (
            sb.rpc(
                "match_procedure_kb",
                {
                    "p_tenant_id": tenant_id,
                    "p_embedding": vector_str,
                    "p_top_k": top_k,
                },
            )
            .execute()
        )
        return result.data or []
    except Exception as exc:
        log.warning("knowledge.retrieve_vector.failed", error=str(exc))
        return []


def _retrieve_by_keyword(
    tenant_id: str, question: str, top_k: int
) -> List[Dict[str, str]]:
    """ILIKE keyword fallback when embeddings are absent."""
    try:
        sb = get_supabase()
        # Extract meaningful words (≥4 chars, non-stopwords)
        words = [w for w in re.findall(r"[a-záéíóúâêîôûãẽõüçñ]+", question.lower()) if len(w) >= 4]
        if not words:
            words = question.lower().split()[:3]

        query = sb.table("sf_procedure_kb").select("procedure, title, body").eq("tenant_id", tenant_id)
        # OR across words — take first word as filter, extend with or_
        if words:
            filters = [f"body.ilike.%{w}%,title.ilike.%{w}%" for w in words[:3]]
            query = query.or_(",".join(filters))

        result = query.limit(top_k).execute()
        return result.data or []
    except Exception as exc:
        log.warning("knowledge.retrieve_keyword.failed", error=str(exc))
        return []


def _retrieve(tenant_id: str, question: str, sensitive: bool) -> List[Dict[str, str]]:
    """Retrieve relevant KB chunks; vector first, keyword fallback."""
    embedding = _embed_query(question)
    if embedding:
        chunks = _retrieve_by_vector(tenant_id, embedding, TOP_K)
        if chunks:
            return chunks

    # When sensitive, also fetch contraindication chunks explicitly.
    if sensitive:
        return _retrieve_by_keyword(tenant_id, question + " contraindicação gestante alergia", TOP_K)
    return _retrieve_by_keyword(tenant_id, question, TOP_K)


def _build_context(chunks: List[Dict[str, str]]) -> str:
    if not chunks:
        return "Nenhuma informação encontrada na base de procedimentos."
    parts = []
    for c in chunks:
        proc = c.get("procedure", "")
        title = c.get("title", "")
        body = c.get("body", "")
        parts.append(f"### {proc} — {title}\n{body}")
    return "\n\n".join(parts)


class KnowledgeSpecialist:
    """RAG specialist for procedure questions. Uses Haiku; upgrades to Sonnet for sensitive queries."""

    def __init__(self, client: Optional[Anthropic] = None) -> None:
        self.client = client or Anthropic()
        self.last_response: Any = None

    def forward(
        self,
        question: str,
        clinic_name: str,
        tenant_id: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        sensitive = _is_sensitive(question)
        model = KNOWLEDGE_SONNET if sensitive else KNOWLEDGE_HAIKU

        chunks = _retrieve(tenant_id, question, sensitive)
        context = _build_context(chunks)
        source_titles = [c.get("title", "") for c in chunks if c.get("title")]

        system = SYSTEM_PROMPT.format(clinic_name=clinic_name)
        user_content = (
            f"Informações de procedimentos disponíveis:\n{context}\n\n"
            f"Pergunta do paciente: {question}"
        )

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=512,
                system=system,
                tools=[ANSWER_TOOL],
                tool_choice={"type": "tool", "name": "knowledge_answer"},
                messages=[{"role": "user", "content": user_content}],
            )
            self.last_response = response

            payload = self._extract_tool_input(response)
            if payload is None:
                raise ValueError("knowledge_answer tool call missing")

            out = KnowledgeOutput.model_validate(payload)

            # Regex guard: sensitive_flag must be True if question contains triggers.
            if sensitive and not out.sensitive_flag:
                out.sensitive_flag = True
                out.requires_consultation = True

            # Append in-person CTA when flagged.
            if out.sensitive_flag and "avaliação presencial" not in out.answer:
                out.answer += (
                    "\n\nIsso depende de uma avaliação presencial com a nossa equipe médica "
                    "— quer agendar uma consulta?"
                )

        except Exception as exc:
            log.error("knowledge.forward.failed", error=str(exc), model=model)
            self.last_response = None
            fallback_answer = (
                "No momento não consigo acessar nossa base de procedimentos. "
                "Posso te ajudar com mais alguma coisa, ou prefere falar com nossa equipe?"
            )
            out = KnowledgeOutput(
                answer=fallback_answer,
                sources=source_titles,
                requires_consultation=False,
                sensitive_flag=False,
            )

        log.info(
            "knowledge.forward.ok",
            model=model,
            sensitive=sensitive,
            requires_consultation=out.requires_consultation,
            sensitive_flag=out.sensitive_flag,
            chunk_count=len(chunks),
        )

        return {
            "messages": [{"type": "text", "content": out.answer}],
            "conversation_stage": "knowledge",
            "reasoning": f"KnowledgeSpecialist ({model}); sensitive={sensitive}; chunks={len(chunks)}",
            "data": {
                "answer": out.answer,
                "sources": out.sources,
                "requires_consultation": out.requires_consultation,
                "sensitive_flag": out.sensitive_flag,
                "routing_hint": "SCHEDULE_NEXT" if out.sensitive_flag else None,
            },
        }

    def _extract_tool_input(self, response: Any) -> Optional[Dict[str, Any]]:
        for block in getattr(response, "content", []) or []:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == "knowledge_answer"
            ):
                payload = getattr(block, "input", None)
                if isinstance(payload, dict):
                    return payload
        return None
