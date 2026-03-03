"""
SofiaRouterAgent — classifies patient message into one or more intents and detects language.
"""

import re
import dspy
from typing import List, Dict, Any

from .signatures import SofiaRouterSignature, SofiaIntentType


INTENT_PRIORITY = {
    "HUMAN_ESCALATION": 1,
    "SCHEDULE": 2,
    "REENGAGE": 3,
    "FAQ": 4,
    "GREETING": 5,
    "UNCLASSIFIED": 6,
}


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

    def _parse_intents(self, raw: Any) -> List[str]:
        valid = {item.value for item in SofiaIntentType}
        seen = set()
        parsed = []

        if isinstance(raw, str):
            candidates = [part.strip().upper() for part in raw.split(",")]
            for candidate in candidates:
                # Exact match first
                if candidate in valid and candidate not in seen:
                    seen.add(candidate)
                    parsed.append(candidate)
                    continue
                # Substring match fallback
                for v in valid:
                    if v in candidate and v not in seen:
                        seen.add(v)
                        parsed.append(v)
                        break

        if not parsed:
            return [SofiaIntentType.UNCLASSIFIED.value]

        # Sort by INTENT_PRIORITY descending (highest number first = informational first,
        # lowest number = most important = CTA = last)
        parsed.sort(key=lambda x: INTENT_PRIORITY.get(x, 6), reverse=True)
        return parsed

    def _parse_language(self, raw: Any) -> str:
        if isinstance(raw, str):
            cleaned = raw.strip()
            if cleaned:
                return cleaned
        return "pt-BR"

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
    ) -> Dict[str, Any]:
        history_str = self._format_history(history)

        try:
            result = self.process(
                latest_message=latest_message,
                history_str=history_str,
                conversation_stage=conversation_stage,
            )
            detected_intents = self._parse_intents(result.detected_intents)
            language = self._parse_language(result.language)
            reasoning = str(result.reasoning).strip()
        except Exception as e:
            print(f"RouterAgent error: {e}")
            detected_intents = [SofiaIntentType.UNCLASSIFIED.value]
            language = "pt-BR"
            reasoning = f"Erro no roteamento: {str(e)}"

        return {
            "detected_intents": detected_intents,
            "language": language,
            "reasoning": reasoning,
        }
