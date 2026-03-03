"""
Sofia State, Request and Response models — v2.0
"""

from typing import TypedDict, Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# AGENT RUN SCHEMA
# ============================================================================

class AgentRunMessage(TypedDict):
    type: str       # "text" | "audio" | "reaction" | "image"
    content: str


class AgentRun(TypedDict):
    # Identity
    agent: str                       # "FAQResponder", "Scheduler", etc.
    reason: str                      # why this agent was triggered
    status: str                      # "success" | "error"
    # Output
    messages: List[AgentRunMessage]  # messages to send to patient (in order)
    data: Optional[Dict[str, Any]]   # structured payload for n8n actions
    # Timing
    started_at: str                  # ISO UTC timestamp
    duration_ms: float
    # Token usage (0 for deterministic agents)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Observability
    trace_id: str                    # UUID per API call — correlation key
    clinic_id: str                   # tenant
    session_id: str                  # conversation
    language: str                    # detected language e.g. "pt-BR"
    sofia_version: str               # "2.0"


# ============================================================================
# LANGGRAPH STATE v2
# ============================================================================

class SofiaState(TypedDict):
    # ---- Inputs from n8n ----
    instance_id: str
    clinic_id: str
    remote_jid: str
    push_name: Optional[str]
    message: str
    message_type: str
    wamid: str
    available_slots: List[str]
    conversation_type: str           # "first_contact" | "confirmation" | "reminder" | "reengage" | "upsell"

    # ---- Observability ----
    trace_id: str                    # UUID generated at request entry in main.py
    language: str                    # detected by Router (e.g. "pt-BR")

    # ---- Context loaded from Supabase ----
    session_id: str
    clinic_name: str
    assistant_name: str
    history: List[Dict[str, str]]    # [{"role": "human|AgentName", "content": "..."}]
    conversation_stage: str
    patient_name: Optional[str]
    customer_id: Optional[str]

    # ---- Routing ----
    detected_intents: List[str]      # sorted: informational first, CTA (most important) last

    # ---- Agent outputs ----
    agent_runs: List[AgentRun]
    requires_human: bool             # convenience flag derived from agent_runs


# ============================================================================
# API REQUEST / RESPONSE v2
# ============================================================================

class SofiaRequest(BaseModel):
    instance_id: str = Field(..., description="Evolution API instance ID (slug da clínica)")
    clinic_id: str = Field(..., description="UUID da clínica no Supabase")
    remote_jid: str = Field(..., description="WhatsApp JID do paciente (ex: 5511999999999@s.whatsapp.net)")
    push_name: Optional[str] = Field(None, description="Nome exibido no WhatsApp do paciente")
    message: str = Field(..., description="Mensagem recebida do paciente")
    message_type: str = Field(default="text", description="Tipo da mensagem: text, image, audio, document")
    wamid: str = Field(..., description="WhatsApp message ID único")
    available_slots: List[str] = Field(
        default_factory=list,
        description="Horários disponíveis para agendamento (gerados pelo n8n)"
    )
    conversation_type: str = Field(
        default="first_contact",
        description="Tipo de conversa: first_contact | confirmation | reminder | reengage | upsell"
    )


class SofiaResponse(BaseModel):
    agent_runs: List[Dict[str, Any]] = Field(
        ...,
        description="Lista de execuções de agentes — cada item contém messages (para enviar) e data (para ações n8n)"
    )
    session_id: str = Field(..., description="ID da sessão no Supabase")
    conversation_stage: str = Field(..., description="Estágio atual da conversa")
    requires_human: bool = Field(default=False, description="Se requer escalada para atendente humano")
    processing_time_ms: float = Field(..., description="Tempo de processamento total em milissegundos")
    trace_id: str = Field(..., description="UUID de correlação para observabilidade")
    language: str = Field(..., description="Idioma detectado na conversa")
