"""
HumanEscalationAgent — deterministic agent, no LLM.
Generates the acolhimento (welcome) message and marks the conversation as paused.
"""

from typing import Dict, Any, List


class HumanEscalationAgent:
    """Deterministic agent — reads patient/assistant/clinic from state and builds transition message."""

    def forward(
        self,
        patient_name: str,
        assistant_name: str,
        clinic_name: str,
    ) -> Dict[str, Any]:
        patient_greeting = ""
        if patient_name and patient_name != "Paciente":
            patient_greeting = f"{patient_name}, "

        message = (
            f"{patient_greeting}"
            f"Vou te conectar com nossa recepcionista, "
            f"ela te responde em instantes."
        )

        return {
            "messages": [{"type": "text", "content": message}],
            "response_message": message,
            "agent_name": "HumanEscalation",
            "conversation_stage": "human_escalation",
            "reasoning": "Patient escalated to human — acolhimento sent, conversation paused.",
            "requires_human": True,
            "data": {
                "type": "escalation",
                "reason": "patient_request",
            },
        }
