"""
ClosureAgent — deterministic agent for short social messages.
Handles "obrigado", "ok", "até logo", etc. with zero token cost.
Falls back to FAQResponder for unrecognized UNCLASSIFIED messages.
"""

import re
from typing import Dict, Any, Optional


# Common social closure tokens in PT-BR, EN, and ES.
# Match is done against individual words/tokens in the message.
_CLOSURE_TOKENS = {
    # PT-BR
    "obrigado", "obrigada", "valeu", "vlw", "ótimo", "otimo",
    "perfeito", "entendido", "entendi", "certo", "combinado",
    "ok", "okay", "tá", "ta", "tudo", "show",
    "tchau", "até", "flw", "abraço", "abracos",
    # EN
    "thanks", "thank", "perfect", "great", "understood",
    "got", "bye", "goodbye", "cheers",
    # ES
    "gracias", "perfecto", "entendido", "ok",
}

# Max word count to even attempt closure detection (long messages = real questions)
_MAX_WORDS = 6


def is_closure_message(message: str) -> bool:
    """
    Returns True if the message is a short social closure with no real question.
    Fast, deterministic — no LLM.
    """
    stripped = message.strip()
    words = re.findall(r"\w+", stripped.lower())
    if not words or len(words) > _MAX_WORDS:
        return False
    # At least one closure token must be present
    return bool(_CLOSURE_TOKENS.intersection(words))


class ClosureAgent:
    """Deterministic closure agent — no LLM, zero token cost."""

    def forward(
        self,
        patient_name: Optional[str],
        conversation_stage: str,
    ) -> Dict[str, Any]:
        name = patient_name if patient_name and patient_name != "Paciente" else ""

        if conversation_stage == "booked":
            if name:
                content = f"Por nada, {name}! Até breve e boa consulta! 😊"
            else:
                content = "Por nada! Até breve e boa consulta! 😊"
        else:
            if name:
                content = f"Por nada, {name}! Se precisar de mais alguma coisa, é só chamar. 😊"
            else:
                content = "Por nada! Se precisar de mais alguma coisa, é só chamar. 😊"

        return {
            "messages": [{"type": "text", "content": content}],
            "conversation_stage": conversation_stage,  # preserve current stage
            "reasoning": "Short social closure detected — deterministic response, no LLM needed.",
            "data": None,
        }
