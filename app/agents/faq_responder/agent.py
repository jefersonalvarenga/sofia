"""
FAQResponderAgent — answers patient questions about clinic services, prices, etc.
"""

import dspy
from typing import List, Dict, Any

from .signatures import FAQResponderSignature


class FAQResponderAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process = dspy.ChainOfThought(FAQResponderSignature)

    def forward(
        self,
        patient_message: str,
        history: List[Dict[str, str]],
        clinic_name: str,
        patient_name: str,
        services_context: str,
        business_rules: str,
    ) -> Dict[str, Any]:
        history_str = self._format_history(history)

        try:
            result = self.process(
                patient_message=patient_message,
                history_str=history_str,
                clinic_name=clinic_name,
                patient_name=patient_name or "Paciente",
                services_context=services_context,
                business_rules=business_rules,
            )
            return {
                "messages": [{"type": "text", "content": str(result.response_message).strip()}],
                "conversation_stage": "faq",
                "reasoning": str(result.reasoning).strip(),
            }
        except Exception as e:
            print(f"FAQResponderAgent error: {e}")
            return {
                "messages": [{"type": "text", "content": "Olá! Como posso ajudar?"}],
                "conversation_stage": "faq",
                "reasoning": f"Erro: {str(e)}",
            }

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "Sem histórico anterior."
        lines = []
        for turn in history[-20:]:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            prefix = "Paciente" if role == "human" else role
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)
