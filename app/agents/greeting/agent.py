"""
GreetingAgent — deterministic agent for pure greeting messages.
No LLM — fast, zero token cost.
"""

from typing import Dict, Any, Optional


GREETING_TEMPLATES = [
    "Olá, {name}! 😊 Seja bem-vindo(a)! Como posso ajudar você hoje?",
    "Oi, {name}! Tudo bem? Estou aqui para ajudar. O que você precisa?",
    "Olá, {name}! Que bom ter você aqui! Como posso te ajudar?",
]


class GreetingAgent:
    """Deterministic greeting agent — no LLM, zero token cost."""

    def forward(
        self,
        patient_name: Optional[str],
        clinic_name: str,
        assistant_name: str,
        history_length: int = 0,
    ) -> Dict[str, Any]:
        name = patient_name if patient_name and patient_name != "Paciente" else ""

        # Pick template based on history length (first contact vs returning)
        if history_length == 0:
            # First ever message — warm welcome
            if name:
                content = f"Olá, {name}! 😊 Seja bem-vindo(a) à {clinic_name}! Sou a {assistant_name}. Como posso ajudar você hoje?"
            else:
                content = f"Olá! 😊 Seja bem-vindo(a) à {clinic_name}! Sou a {assistant_name}. Como posso ajudar?"
        else:
            # Returning patient — shorter greeting
            if name:
                content = f"Olá, {name}! Como posso ajudar você hoje? 😊"
            else:
                content = "Olá! Como posso ajudar? 😊"

        return {
            "messages": [{"type": "text", "content": content}],
            "conversation_stage": "greeting",
            "reasoning": "Pure greeting detected — deterministic response, no LLM needed.",
            "data": None,
        }
