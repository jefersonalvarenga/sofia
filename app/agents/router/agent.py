"""
SofiaRouterAgent — classifies patient message into a single intent.
"""

import re
import dspy
from typing import List, Dict, Any

from .signatures import SofiaRouterSignature, SofiaIntentType


class SofiaRouterAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process = dspy.ChainOfThought(SofiaRouterSignature)

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "Sem histórico anterior."
        lines = []
        for turn in history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            prefix = "Paciente" if role == "human" else role
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines[-20:])  # last 20 turns

    def _parse_intent(self, raw: Any) -> str:
        valid = {item.value for item in SofiaIntentType}
        if isinstance(raw, str):
            candidate = raw.strip().upper()
            if candidate in valid:
                return candidate
            # Try to extract from phrase e.g. "The intent is SCHEDULE"
            for v in valid:
                if v in candidate:
                    return v
        return SofiaIntentType.UNCLASSIFIED.value

    def _parse_confidence(self, raw: Any) -> float:
        try:
            if isinstance(raw, (int, float)):
                return float(raw)
            match = re.search(r"[\d.]+", str(raw))
            return float(match.group()) if match else 0.0
        except (ValueError, TypeError):
            return 0.0

    def forward(
        self,
        latest_message: str,
        history: List[Dict[str, str]],
        conversation_stage: str,
        language: str = "pt-BR",
    ) -> Dict[str, Any]:
        history_str = self._format_history(history)

        try:
            result = self.process(
                latest_message=latest_message,
                history_str=history_str,
                conversation_stage=conversation_stage,
                language=language,
            )
            intent = self._parse_intent(result.intent)
            confidence = self._parse_confidence(result.confidence)
            reasoning = str(result.reasoning).strip()
        except Exception as e:
            print(f"RouterAgent error: {e}")
            intent = SofiaIntentType.UNCLASSIFIED.value
            confidence = 0.0
            reasoning = f"Erro no roteamento: {str(e)}"

        return {
            "intent": intent,
            "reasoning": reasoning,
            "confidence": confidence,
        }
