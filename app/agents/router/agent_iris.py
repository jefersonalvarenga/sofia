"""
IrisRouterAgent — Anthropic SDK + tool use Pydantic for structured output.

Replaces SofiaRouterAgent (DSPy ChainOfThought) on the Iris pipeline. Sofia legacy
keeps using SofiaRouterAgent for now.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from pydantic import BaseModel, Field

from .signatures import SofiaIntentType


IRIS_ROUTER_MODEL = "claude-haiku-4-5-20251001"

INTENT_PRIORITY = {
    "HUMAN_ESCALATION": 1,
    "SCHEDULE": 2,
    "REENGAGE": 3,
    "FAQ": 4,
    "GREETING": 5,
    "UNCLASSIFIED": 6,
}

VALID_INTENTS = {item.value for item in SofiaIntentType}


class IrisRouterOutput(BaseModel):
    """Structured output schema for the Iris router tool call."""

    detected_intents: List[SofiaIntentType] = Field(
        ...,
        description=(
            "Lista de intents detectadas, ordenadas de informacional para mais "
            "importante (CTA por último). HUMAN_ESCALATION é sempre o último se "
            "presente. Valores válidos: GREETING, FAQ, SCHEDULE, REENGAGE, "
            "HUMAN_ESCALATION, UNCLASSIFIED."
        ),
        min_length=1,
    )
    language: str = Field(
        ...,
        description="Tag BCP-47 da língua do paciente (ex: 'pt-BR', 'es', 'en'). Default: 'pt-BR'.",
    )
    reasoning: str = Field(
        ...,
        description="Explicação curta da decisão de roteamento (≤200 chars).",
        max_length=400,
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confiança no intent primário (último/mais importante), entre 0.0 e 1.0.",
    )


SYSTEM_PROMPT = """Você é o Router Agent da Iris, recepcionista de IA para clínicas de estética/medicina no WhatsApp.
Sua função é classificar a última mensagem do paciente em uma ou mais intents e detectar a língua.

Definições:
- GREETING: paciente inicia conversa, cumprimenta, ou envia primeira mensagem.
- FAQ: paciente pergunta sobre serviços, preços, endereço, horários, convênios, procedimentos, recuperação ou qualquer informação geral.
- SCHEDULE: paciente quer agendar, confirmar consulta, ou está no meio de um agendamento (escolhendo serviço/horário).
- HUMAN_ESCALATION: paciente pede explicitamente para falar com um humano, atendente ou recepcionista.
- REENGAGE: paciente retoma uma conversa que estava pausada ou ociosa.
- UNCLASSIFIED: nenhuma das anteriores se aplica claramente.

Regras multi-intent:
1. Uma mensagem pode disparar múltiplas intents — detecte TODAS que se aplicam.
2. Ordene as intents do informacional para o mais importante (CTA por último).
   Prioridade (mais importante = último): HUMAN_ESCALATION > SCHEDULE > REENGAGE > FAQ > GREETING.
3. HUMAN_ESCALATION sempre aparece por último quando presente — sobrepõe outras ações.
4. Quando só uma intent se aplica, retorne apenas ela.
5. Use `conversation_stage` como contexto — paciente em meio a agendamento normalmente é SCHEDULE.
6. `reasoning` deve ser conciso (≤200 chars).

Detecção de língua:
- Detecte a partir do texto da mensagem.
- Use tags BCP-47 (ex: 'pt-BR', 'es', 'en').
- Default: 'pt-BR' se ambíguo.

Exemplos:
- "oi" → detected_intents=["GREETING"], language="pt-BR".
- "quanto custa limpeza?" → detected_intents=["FAQ"], language="pt-BR".
- "quero agendar uma limpeza, vocês aceitam Unimed?" → detected_intents=["FAQ","SCHEDULE"].
- "quero falar com atendente" → detected_intents=["HUMAN_ESCALATION"].

Sempre chame a tool `classify_intent` com o resultado estruturado. Não responda em texto livre."""


def _build_classify_tool() -> Dict[str, Any]:
    """Build the classify_intent tool schema from the Pydantic model."""
    schema = IrisRouterOutput.model_json_schema()
    return {
        "name": "classify_intent",
        "description": "Classify the patient message into intents and detect language.",
        "input_schema": {
            "type": "object",
            "properties": schema["properties"],
            "required": schema.get("required", []),
            "$defs": schema.get("$defs", {}),
        },
    }


CLASSIFY_TOOL = _build_classify_tool()


class IrisRouterAgent:
    """Drop-in replacement for SofiaRouterAgent on the Iris pipeline."""

    def __init__(
        self,
        client: Optional[Anthropic] = None,
        model: str = IRIS_ROUTER_MODEL,
        max_tokens: int = 512,
    ) -> None:
        self.client = client or Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.last_response: Any = None

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "Sem histórico anterior."
        lines = []
        for turn in history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            prefix = "Paciente" if role == "human" else role
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines[-20:])

    def _normalize_intents(self, raw_intents: List[Any]) -> List[str]:
        seen: set[str] = set()
        parsed: List[str] = []
        for item in raw_intents or []:
            value = item.value if isinstance(item, SofiaIntentType) else str(item).strip().upper()
            if value in VALID_INTENTS and value not in seen:
                seen.add(value)
                parsed.append(value)
        if not parsed:
            return [SofiaIntentType.UNCLASSIFIED.value]
        parsed.sort(key=lambda x: INTENT_PRIORITY.get(x, 6), reverse=True)
        return parsed

    def _extract_tool_input(self, response: Any) -> Optional[Dict[str, Any]]:
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "classify_intent":
                payload = getattr(block, "input", None)
                if isinstance(payload, dict):
                    return payload
        return None

    def forward(
        self,
        latest_message: str,
        history: List[Dict[str, str]],
        conversation_stage: str,
    ) -> Dict[str, Any]:
        history_str = self._format_history(history)
        user_prompt = (
            f"latest_message: {latest_message}\n"
            f"conversation_stage: {conversation_stage}\n"
            f"history:\n{history_str}"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=[CLASSIFY_TOOL],
                tool_choice={"type": "tool", "name": "classify_intent"},
                messages=[{"role": "user", "content": user_prompt}],
            )
            self.last_response = response

            payload = self._extract_tool_input(response)
            if payload is None:
                raise ValueError("classify_intent tool call missing in response")

            parsed = IrisRouterOutput.model_validate(payload)
            detected_intents = self._normalize_intents(parsed.detected_intents)
            language = parsed.language.strip() or "pt-BR"
            reasoning = parsed.reasoning.strip()
            confidence = max(0.0, min(1.0, float(parsed.confidence)))
        except Exception as e:
            print(f"IrisRouterAgent error: {e}")
            self.last_response = None
            return {
                "detected_intents": [SofiaIntentType.UNCLASSIFIED.value],
                "language": "pt-BR",
                "reasoning": f"Erro no roteamento: {e}",
                "confidence": 0.0,
            }

        return {
            "detected_intents": detected_intents,
            "language": language,
            "reasoning": reasoning,
            "confidence": confidence,
        }
