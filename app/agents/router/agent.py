"""
RouterAgent — Iris primary router.

Spec: kb/07-MVP/Tech/03-Discussoes/02 - Spec Router Primario.md

Classifies the patient's latest WhatsApp message into one or more intents
from the 8-value IntentType vocabulary. Runs on gpt-4o-mini via DSPy
(LiteLLM under the hood) for provider-agnostic structured output.

Differences vs the legacy IrisRouterAgent (deleted in this PR):
  - 8 intents (BUSINESS_INFO + TOPIC_KNOWLEDGE split, INTAKE added)
  - No language detection (MVP is pt-BR only)
  - No medical guardrail in the router (KnowledgeAgent owns that)
  - Confidence threshold (default 0.70) downgrades to UNCLASSIFIED
  - No DSPy fallback agent (legacy SofiaRouterAgent deleted)
  - Provider is OpenAI gpt-4o-mini, not Claude Haiku
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import dspy
from pydantic import ValidationError

from app.core.telemetry import log

from .intents import IntentType
from .normalize import normalize_intents
from .schemas import RouterOutput


ROUTER_MODEL = "gpt-4o-mini"

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
MIN_CONFIDENCE_THRESHOLD = 0.50


SYSTEM_PROMPT = """Router da Iris (recepcionista IA para clinicas de estetica). Classifique a ultima mensagem do paciente.

Intents (8):
- BUSINESS_INFO: preco, horario, endereco, convenio, lista de servicos.
- TOPIC_KNOWLEDGE: como servico funciona, contraindicacoes, recuperacao, "posso fazer X estando Y". Inclui gatilho clinico (NAO usar HUMAN_ESCALATION pra isso).
- REENGAGE: retomada de conversa inativa (use conversation_stage como dica).
- GREETING: "oi", "bom dia", primeira mensagem sem outra intencao.
- UNCLASSIFIED: fallback / sticker / fora de contexto.
- INTAKE: paciente declara INTERESSE esperando consultoria — "quero fazer botox". Difere de SCHEDULE (intake = qualificacao; schedule = marcacao).
- SCHEDULE: agendar/cancelar/remarcar — "quero marcar", "tem horario amanha?".
- HUMAN_ESCALATION: pedido EXPLICITO de humano/atendente.

Regras:
- Multi-intent: detecte TODAS as intents aplicaveis. Cada uma com {intent, scope_text} (scope_text = trecho literal que originou).
- Mesma intent nao repete.
- confidence em [0,1] agregado. Se ambiguo, use < 0.70.
- Sempre pt-BR. reasoning curto (<= 400 chars), opcional.

Exemplos:
- "voces aceitam Unimed? quero agendar" -> intents=[{intent:"BUSINESS_INFO",scope_text:"voces aceitam Unimed?"},{intent:"SCHEDULE",scope_text:"quero agendar"}], confidence=0.88
- "posso fazer botox amamentando?" -> intents=[{intent:"TOPIC_KNOWLEDGE",scope_text:"posso fazer botox amamentando?"}], confidence=0.88 (NAO HUMAN_ESCALATION)
- "quero fazer botox" -> intents=[{intent:"INTAKE",scope_text:"quero fazer botox"}], confidence=0.90
- "quero falar com atendente" -> intents=[{intent:"HUMAN_ESCALATION",scope_text:"quero falar com atendente"}], confidence=0.97

Responda APENAS JSON valido."""


def _read_threshold() -> float:
    """Read ROUTER_CONFIDENCE_THRESHOLD from env, clamped to [MIN, 1.0]."""
    raw = os.environ.get("ROUTER_CONFIDENCE_THRESHOLD")
    if not raw:
        return DEFAULT_CONFIDENCE_THRESHOLD
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.warn(
            "router.threshold.invalid",
            raw=raw,
            fallback=DEFAULT_CONFIDENCE_THRESHOLD,
        )
        return DEFAULT_CONFIDENCE_THRESHOLD
    if value < MIN_CONFIDENCE_THRESHOLD:
        log.warn(
            "router.threshold.too_low",
            raw=value,
            minimum=MIN_CONFIDENCE_THRESHOLD,
        )
        return MIN_CONFIDENCE_THRESHOLD
    return min(1.0, value)


class RouterAgent:
    """Iris primary router. Drop-in replacement for IrisRouterAgent.

    Usage:
        agent = RouterAgent()
        result = agent.forward(
            latest_message="quero marcar limpeza",
            history=[],
            conversation_stage="new",
        )
        # result["intents"] -> [{"intent": "SCHEDULE", "scope_text": "..."}]
        # result["detected_intents"] -> ["SCHEDULE"]
        # result["reasoning"] -> "..."
        # result["confidence"] -> 0.93
    """

    def __init__(
        self,
        lm: Optional[dspy.LM] = None,
        model: str = ROUTER_MODEL,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._lm_override = lm

    def _get_lm(self) -> dspy.LM:
        """Resolve the active LM (constructor override -> dspy.settings.lm)."""
        if self._lm_override is not None:
            return self._lm_override
        lm = dspy.settings.lm
        if lm is None:
            raise RuntimeError(
                "RouterAgent: no DSPy LM configured. Call init_dspy() or pass lm= to constructor."
            )
        return lm

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "Sem historico anterior."
        lines = []
        for turn in history[-20:]:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            prefix = "Paciente" if role == "human" else role
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def _build_user_prompt(
        self,
        latest_message: str,
        history: List[Dict[str, str]],
        conversation_stage: str,
    ) -> str:
        return (
            f"latest_message: {latest_message}\n"
            f"conversation_stage: {conversation_stage}\n"
            f"history:\n{self._format_history(history)}"
        )

    def _call_lm(self, user_prompt: str) -> str:
        """Call the LM with JSON response_format and return the raw content string."""
        lm = self._get_lm()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        outputs = lm(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        if not outputs:
            raise ValueError("router LM returned no outputs")
        return outputs[0]

    def _parse(self, raw_content: str) -> RouterOutput:
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"router LM returned non-JSON content: {exc}") from exc
        try:
            return RouterOutput.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"router output failed Pydantic validation: {exc}") from exc

    def forward(
        self,
        latest_message: str,
        history: List[Dict[str, str]],
        conversation_stage: str,
    ) -> Dict[str, Any]:
        """Run the router. Propagates exceptions on failure (no silent capture).

        Returns a dict shaped:
            {
                "intents": [{"intent": str, "scope_text": str}, ...],
                "detected_intents": [str, ...],
                "reasoning": str,
                "confidence": float,
            }
        """
        threshold = _read_threshold()
        user_prompt = self._build_user_prompt(latest_message, history, conversation_stage)

        try:
            raw_content = self._call_lm(user_prompt)
            parsed = self._parse(raw_content)
        except Exception as exc:
            log.error(
                "router.failed",
                error=str(exc),
                error_type=type(exc).__name__,
                model=self.model,
            )
            raise

        confidence = max(0.0, min(1.0, float(parsed.confidence)))
        reasoning = parsed.reasoning.strip()

        if confidence < threshold:
            intents = [
                {
                    "intent": IntentType.UNCLASSIFIED.value,
                    "scope_text": latest_message,
                }
            ]
        else:
            intents = normalize_intents(parsed.intents, latest_message)

        return {
            "intents": intents,
            "detected_intents": [item["intent"] for item in intents],
            "reasoning": reasoning,
            "confidence": confidence,
        }
