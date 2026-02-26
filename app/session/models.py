"""
Sofia State, Request and Response models.
"""

from typing import TypedDict, Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# LANGGRAPH STATE
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

    # ---- Context loaded from Supabase ----
    session_id: str
    clinic_name: str
    assistant_name: str
    services_context: str        # JSON-serialized clinic_services + clinic_offers
    business_rules: str          # JSON-serialized clinic_business_rules
    history: List[Dict[str, str]]  # [{"role": "human|AgentName", "content": "..."}]
    conversation_stage: str
    patient_name: Optional[str]
    customer_id: Optional[str]

    # ---- Outputs produced by agents ----
    intent: Optional[str]
    confidence: Optional[float]
    response_message: Optional[str]
    agent_name: Optional[str]
    requires_human: bool
    appointment_created: Optional[Dict[str, Any]]
    reasoning: Optional[str]
    processing_time_ms: float


# ============================================================================
# API REQUEST / RESPONSE
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


class SofiaResponse(BaseModel):
    response_message: str = Field(..., description="Mensagem de resposta a ser enviada ao paciente")
    should_send: bool = Field(..., description="Se deve enviar a mensagem via WhatsApp")
    agent_name: str = Field(..., description="Nome do agente que gerou a resposta")
    conversation_stage: str = Field(..., description="Estágio atual da conversa")
    session_id: str = Field(..., description="ID da sessão no Supabase")
    requires_human: bool = Field(default=False, description="Se requer escalada para atendente humano")
    appointment_created: Optional[Dict[str, Any]] = Field(None, description="Dados do agendamento criado, se houver")
    processing_time_ms: float = Field(..., description="Tempo de processamento em milissegundos")
