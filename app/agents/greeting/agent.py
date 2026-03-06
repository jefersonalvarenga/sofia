"""
GreetingAgent — deterministic agent for pure greeting messages.
No LLM — fast, zero token cost.

When clinic_style.greeting_example is available, adapts the greeting
to match the clinic's tone (opener style + emoji usage) instead of
using the generic template.
"""

from typing import Dict, Any, Optional


class GreetingAgent:
    """Deterministic greeting agent — no LLM, zero token cost."""

    def forward(
        self,
        patient_name: Optional[str],
        clinic_name: str,
        assistant_name: str,
        history_length: int = 0,
        greeting_example: str = "",
    ) -> Dict[str, Any]:
        name = patient_name if patient_name and patient_name != "Paciente" else ""

        if greeting_example:
            content = self._style_greeting(
                greeting_example, name, clinic_name, assistant_name, history_length
            )
        elif history_length == 0:
            if name:
                content = f"Olá, {name}! 😊 Seja bem-vindo(a) à {clinic_name}! Sou a {assistant_name}. Como posso ajudar você hoje?"
            else:
                content = f"Olá! 😊 Seja bem-vindo(a) à {clinic_name}! Sou a {assistant_name}. Como posso ajudar?"
        else:
            content = f"Olá, {name}! Como posso ajudar você hoje? 😊" if name else "Olá! Como posso ajudar? 😊"

        return {
            "messages": [{"type": "text", "content": content}],
            "conversation_stage": "greeting",
            "reasoning": "Pure greeting detected — deterministic response, no LLM needed.",
            "data": None,
        }

    def _style_greeting(
        self,
        example: str,
        name: str,
        clinic_name: str,
        assistant_name: str,
        history_length: int,
    ) -> str:
        """Build a greeting that mirrors the clinic's example style."""
        # Detect opener ("Oi" vs "Olá") from the example
        opener = "Oi" if example.lower().startswith("oi") else "Olá"
        # Collect any emojis from the example (code points > U+2000)
        emojis = "".join(c for c in example if ord(c) > 0x2000)
        emoji_str = f" {emojis}" if emojis else ""

        name_part = f", {name}" if name else ""

        if history_length == 0:
            return (
                f"{opener}{name_part}!{emoji_str} "
                f"Sou a {assistant_name}, da {clinic_name}. "
                f"Como posso ajudar você hoje?"
            )
        return f"{opener}{name_part}!{emoji_str} Como posso ajudar?"
