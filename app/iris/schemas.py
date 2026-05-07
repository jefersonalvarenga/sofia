"""
Evolution API webhook payload schemas (subset Iris consumes).

Pydantic flexes around two Evolution-emitted shapes:
  - top-level `messages.upsert` event: { event, instance, data: { key, message, ... } }
  - legacy/batched: { instance, data: { messages: [...] } }
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvolutionMessageKey(BaseModel):
    model_config = ConfigDict(extra="ignore")

    remoteJid: str = ""
    fromMe: bool = False
    id: str = ""


class EvolutionMessageNode(BaseModel):
    """One message envelope inside the Evolution payload."""

    model_config = ConfigDict(extra="ignore")

    key: EvolutionMessageKey = Field(default_factory=EvolutionMessageKey)
    pushName: Optional[str] = None
    message: Dict[str, Any] = Field(default_factory=dict)
    messageType: Optional[str] = None


class EvolutionData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: Optional[EvolutionMessageKey] = None
    pushName: Optional[str] = None
    message: Optional[Dict[str, Any]] = None
    messageType: Optional[str] = None
    messages: Optional[List[EvolutionMessageNode]] = None


class EvolutionWebhookPayload(BaseModel):
    """Top-level webhook envelope. Permissive — Evolution variants exist."""

    model_config = ConfigDict(extra="ignore")

    event: Optional[str] = None
    instance: Optional[str] = None
    instanceName: Optional[str] = None
    data: Optional[EvolutionData] = None


class ParsedMessage(BaseModel):
    """Normalized inbound message after parse + filter."""

    model_config = ConfigDict(extra="ignore")

    instance_name: str
    remote_jid: str
    wamid: str
    push_name: str = ""
    message_content: str
    message_type: str = "text"
    phone: str
