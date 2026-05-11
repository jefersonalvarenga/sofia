"""
IrisRouterAgent — Anthropic SDK + tool use Pydantic for structured output.

Replaces SofiaRouterAgent (DSPy ChainOfThought) on the Iris pipeline. Sofia legacy
keeps using SofiaRouterAgent for now.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from pydantic import BaseModel, Field

from app.core.telemetry import log

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


class Intent(BaseModel):
    """A single intent detected in the patient message, with its originating scope.

    The router can emit 1..N of these. `scope_text` is the substring (or close
    paraphrase) of the latest message that triggered this intent — downstream
    specialists receive `scope_text` rather than the full message so each
    specialist answers only its own slice of a multi-question message.
    """

    macro_state: SofiaIntentType = Field(
        ...,
        description=(
            "Macro state of the intent. Valid values: GREETING, FAQ, SCHEDULE, "
            "REENGAGE, HUMAN_ESCALATION, UNCLASSIFIED."
        ),
    )
    scope_text: str = Field(
        ...,
        description=(
            "Trecho da mensagem do paciente que originou essa intent. Prefira "
            "uma substring literal. Quando há só uma intent, pode ser a "
            "mensagem inteira."
        ),
        min_length=1,
    )


class IrisRouterOutput(BaseModel):
    """Structured output schema for the Iris router tool call."""

    intents: List[Intent] = Field(
        ...,
        description=(
            "Lista de intents detectadas (1..N), ordenadas de informacional para "
            "mais importante (CTA por último). HUMAN_ESCALATION é sempre o último "
            "se presente. Cada item tem {macro_state, scope_text}."
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
Sua função é classificar a última mensagem do paciente em uma ou mais intents, identificar o trecho que originou cada uma, e detectar a língua.

Definições:
- GREETING: paciente inicia conversa, cumprimenta, ou envia primeira mensagem.
- FAQ: paciente pergunta sobre serviços, preços, endereço, horários, convênios, procedimentos, recuperação ou qualquer informação geral.
- SCHEDULE: paciente quer agendar, confirmar consulta, ou está no meio de um agendamento (escolhendo serviço/horário).
- HUMAN_ESCALATION: paciente pede explicitamente para falar com um humano, atendente ou recepcionista, OU faz uma pergunta médica/clínica que exige avaliação profissional (gravidez, alergia grave, condição crônica, contraindicação).
- REENGAGE: paciente retoma uma conversa que estava pausada ou ociosa.
- UNCLASSIFIED: nenhuma das anteriores se aplica claramente.

Regras multi-intent:
1. Uma mensagem pode disparar múltiplas intents — detecte TODAS que se aplicam, retornando uma lista `intents` com {macro_state, scope_text}.
2. `scope_text` é o trecho literal (ou quase-literal) da mensagem que originou aquela intent. Quando só uma intent se aplica, `scope_text` pode ser a mensagem inteira.
3. Ordene as intents do informacional para o mais importante (CTA por último).
   Prioridade (mais importante = último): HUMAN_ESCALATION > SCHEDULE > REENGAGE > FAQ > GREETING.
4. HUMAN_ESCALATION sempre aparece por último quando presente — sobrepõe outras ações.
5. Use `conversation_stage` como contexto — paciente em meio a agendamento normalmente é SCHEDULE.
6. `reasoning` deve ser conciso (≤200 chars).

Detecção de língua:
- Detecte a partir do texto da mensagem.
- Use tags BCP-47 (ex: 'pt-BR', 'es', 'en').
- Default: 'pt-BR' se ambíguo.

Exemplos:
- "oi" → intents=[{"macro_state":"GREETING","scope_text":"oi"}], language="pt-BR".
- "quanto custa limpeza?" → intents=[{"macro_state":"FAQ","scope_text":"quanto custa limpeza?"}].
- "quero agendar uma limpeza, vocês aceitam Unimed?" → intents=[{"macro_state":"FAQ","scope_text":"vocês aceitam Unimed?"},{"macro_state":"SCHEDULE","scope_text":"quero agendar uma limpeza"}].
- "Quanto custa o botox? Posso fazer estando grávida?" → intents=[{"macro_state":"FAQ","scope_text":"Quanto custa o botox?"},{"macro_state":"HUMAN_ESCALATION","scope_text":"Posso fazer estando grávida?"}].
- "quero falar com atendente" → intents=[{"macro_state":"HUMAN_ESCALATION","scope_text":"quero falar com atendente"}].

Sempre chame a tool `classify_intent` com o resultado estruturado. Não responda em texto livre."""


def _build_classify_tool() -> Dict[str, Any]:
    """Build the classify_intent tool schema from the Pydantic model."""
    schema = IrisRouterOutput.model_json_schema()
    return {
        "name": "classify_intent",
        "description": "Classify the patient message into intents (with scope_text per intent) and detect language.",
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

    def _normalize_intents(
        self,
        raw_intents: List[Any],
        latest_message: str,
    ) -> List[Dict[str, str]]:
        """Dedup by macro_state, drop unknowns, sort priority (CTA last).

        Each item is a dict ``{"macro_state": str, "scope_text": str}``.
        `raw_intents` may contain Intent pydantic models or plain dicts.
        Falls back to a single UNCLASSIFIED intent (scope = full message) if
        nothing valid remains.
        """
        seen: set[str] = set()
        parsed: List[Dict[str, str]] = []
        for item in raw_intents or []:
            if isinstance(item, Intent):
                macro = item.macro_state.value
                scope = item.scope_text.strip()
            elif isinstance(item, dict):
                raw_macro = item.get("macro_state")
                macro = (
                    raw_macro.value
                    if isinstance(raw_macro, SofiaIntentType)
                    else str(raw_macro or "").strip().upper()
                )
                scope = str(item.get("scope_text") or "").strip()
            else:
                continue

            if macro not in VALID_INTENTS or macro in seen:
                continue
            if not scope:
                scope = latest_message
            seen.add(macro)
            parsed.append({"macro_state": macro, "scope_text": scope})

        if not parsed:
            return [
                {
                    "macro_state": SofiaIntentType.UNCLASSIFIED.value,
                    "scope_text": latest_message,
                }
            ]
        parsed.sort(
            key=lambda x: INTENT_PRIORITY.get(x["macro_state"], 6),
            reverse=True,
        )
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
            intents = self._normalize_intents(parsed.intents, latest_message)
            language = parsed.language.strip() or "pt-BR"
            reasoning = parsed.reasoning.strip()
            confidence = max(0.0, min(1.0, float(parsed.confidence)))
        except Exception as e:
            self.last_response = None
            log.error(
                "iris.router.failed",
                error=str(e),
                error_type=type(e).__name__,
                model=self.model,
            )
            raise

        return {
            "intents": intents,
            "detected_intents": [intent["macro_state"] for intent in intents],
            "language": language,
            "reasoning": reasoning,
            "confidence": confidence,
        }
