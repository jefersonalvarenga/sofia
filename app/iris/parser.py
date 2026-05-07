"""
Evolution payload parser.

Pure function: takes raw dict, returns either ParsedMessage or a skip reason.
Mirrors filtering logic from `workflows/sofia-inbound.json` (n8n) so Iris
keeps parity with the legacy path.
"""

from typing import Any, Dict, Optional, Tuple

from app.iris.schemas import (
    EvolutionMessageKey,
    EvolutionMessageNode,
    EvolutionWebhookPayload,
    ParsedMessage,
)


def _extract_message_node(payload: EvolutionWebhookPayload) -> Optional[EvolutionMessageNode]:
    """Pick the single message envelope from any of Evolution's payload shapes."""
    data = payload.data
    if data is None:
        return None

    if data.messages:
        return data.messages[0]

    if data.key is not None or data.message is not None:
        return EvolutionMessageNode(
            key=data.key or EvolutionMessageKey(),
            pushName=data.pushName,
            message=data.message or {},
            messageType=data.messageType,
        )

    return None


def _extract_text_and_type(message: Dict[str, Any]) -> Tuple[str, str]:
    """Map Evolution `message` dict to (content, type) or ("", "unsupported")."""
    if not isinstance(message, dict):
        return "", "unsupported"

    if message.get("conversation"):
        return str(message["conversation"]), "text"

    extended = message.get("extendedTextMessage") or {}
    if extended.get("text"):
        return str(extended["text"]), "text"

    image = message.get("imageMessage") or {}
    if image.get("caption"):
        return str(image["caption"]), "image"

    if message.get("audioMessage"):
        return "[Áudio]", "audio"

    document = message.get("documentMessage") or {}
    if document.get("caption"):
        return str(document["caption"]), "document"

    return "", "unsupported"


def parse_evolution_payload(
    raw: Dict[str, Any],
) -> Tuple[Optional[ParsedMessage], Optional[str]]:
    """
    Parse + filter. Returns (ParsedMessage, None) on success, (None, skip_reason) on filter hit.

    Skip reasons match the n8n workflow:
      - invalid_payload
      - no_message
      - from_me
      - group_message
      - status_broadcast
      - no_text_content
      - missing_instance
      - missing_remote_jid
      - missing_wamid
    """
    try:
        payload = EvolutionWebhookPayload.model_validate(raw)
    except Exception:
        return None, "invalid_payload"

    node = _extract_message_node(payload)
    if node is None:
        return None, "no_message"

    instance_name = payload.instance or payload.instanceName or ""
    if not instance_name:
        return None, "missing_instance"

    if node.key.fromMe:
        return None, "from_me"

    remote_jid = node.key.remoteJid or ""
    if "@g.us" in remote_jid:
        return None, "group_message"
    if remote_jid == "status@broadcast":
        return None, "status_broadcast"
    if not remote_jid:
        return None, "missing_remote_jid"

    wamid = node.key.id or ""
    if not wamid:
        return None, "missing_wamid"

    content, mtype = _extract_text_and_type(node.message)
    if not content:
        return None, "no_text_content"

    phone = remote_jid.split("@", 1)[0]

    return (
        ParsedMessage(
            instance_name=instance_name,
            remote_jid=remote_jid,
            wamid=wamid,
            push_name=(node.pushName or ""),
            message_content=content,
            message_type=mtype,
            phone=phone,
        ),
        None,
    )
