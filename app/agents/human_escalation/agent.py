"""
HumanEscalationAgent — deterministic agent, no LLM.
Generates a transition message and sets requires_human=True.
"""

from typing import Dict, Any, List


class HumanEscalationAgent:
    """Deterministic agent — reads assistant_name from state and builds transition message."""

    def forward(
        self,
        patient_name: str,
        assistant_name: str,
        clinic_name: str,
    ) -> Dict[str, Any]:
        patient_greeting = f"{patient_name}, " if patient_name and patient_name != "Paciente" else ""

        message = (
            f"{patient_greeting}vou transferir você para um de nossos atendentes agora! "
            f"Em breve alguém da equipe da {clinic_name} entrará em contato. "
            f"Obrigado(a) por nos contatar! 😊"
        )

        return {
            "messages": [{"type": "text", "content": message}],
            "conversation_stage": "human_escalation",
            "reasoning": "Patient explicitly requested human agent.",
            "requires_human": True,
            "data": {
                "type": "escalation",
                "reason": "patient_request",
            },
        }
