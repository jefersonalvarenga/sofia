"""
Notification webhook stub for receptionist escalation.

Pure function — no external dependencies beyond stdlib logging and os.
Logs a structured JSON event to stdout (captured by Railway).
HTTP POST integration is out-of-scope for now.
"""

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_log_handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False


def notify_receptionist(
    tenant_id: str,
    conversation_id: str,
    patient_name: str | None,
    trigger: str,
) -> None:
    """Log a structured escalation notification event.

    Args:
        tenant_id:       Clinic/tenant identifier.
        conversation_id: WhatsApp conversation id (remote_jid).
        patient_name:    Patient display name, or None.
        trigger:         One of explicit_request, clinical_urgency,
                         complaint_tone, sensitive_flag.
    """
    webhook_url = os.getenv(f"WEBHOOK_URL_{tenant_id}") or os.getenv("WEBHOOK_URL_DEFAULT")

    event: dict[str, object] = {
        "event": "escalation.notify",
        "tenant_id": tenant_id,
        "conversation_id": conversation_id,
        "patient_name": patient_name,
        "trigger": trigger,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if webhook_url:
        event["webhook_url"] = webhook_url
        logger.debug("webhook url resolved for tenant %s", tenant_id)
    else:
        logger.debug("no webhook url configured for tenant %s", tenant_id)

    logger.info(json.dumps(event))
