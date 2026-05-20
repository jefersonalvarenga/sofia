"""
RouterAgent — Iris primary router.

Spec: kb/07-MVP/Tech/03-Discussoes/02 - Spec Router Primario.md

Classifies the patient's latest WhatsApp message into one or more intents
from the 8-value IntentType vocabulary. Runs on DeepSeek V4 Flash
(non-thinking mode) via DSPy/LiteLLM. Same model+config as GreetingAgent
for stack consistency.

Differences vs the legacy IrisRouterAgent (deleted in this PR):
  - 8 intents (BUSINESS_INFO + TOPIC_KNOWLEDGE split, INTAKE added)
  - No language detection (MVP is pt-BR only)
  - No medical guardrail in the router (KnowledgeAgent owns that)
  - Confidence threshold (default 0.70) downgrades to UNCLASSIFIED
  - No DSPy fallback agent (legacy SofiaRouterAgent deleted)
  - Provider is DeepSeek V4 Flash non-thinking (was gpt-4o-mini in initial spec).
    The agent self-manages its LM: it does NOT rely on the global
    ``init_dspy()`` configuration.
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


ROUTER_MODEL = "deepseek/deepseek-v4-flash"
ROUTER_TEMPERATURE = 0.0
ROUTER_MAX_TOKENS = 384

# DeepSeek standard inference — no provider-specific extras.
ROUTER_EXTRA_BODY: Dict[str, Any] = {}

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
MIN_CONFIDENCE_THRESHOLD = 0.50


SYSTEM_PROMPT = """Router da Iris (recepcionista IA para clinicas de estetica). Classifique a ultima mensagem do paciente.

Intents (8):
- BUSINESS_INFO: preco, horario, endereco, convenio, lista de servicos.
- TOPIC_KNOWLEDGE: pergunta NEUTRA sobre como servico funciona, contraindicacoes, recuperacao. Paciente perguntando sem se colocar como interessado.
- REENGAGE: retomada de conversa inativa (use conversation_stage como dica).
- GREETING: "oi", "bom dia", primeira mensagem sem outra intencao.
- UNCLASSIFIED: fallback / sticker / fora de contexto.
- INTAKE: paciente expoe sintoma/queixa OU declara desejo de fazer procedimento, SEM verbo de agendamento. Ex: "quero fazer X", "tenho marca/ruga/mancha", "minha pele Y".
- SCHEDULE: paciente quer AGENDAR / CANCELAR / REMARCAR. Verbos-chave: "marcar", "agendar", "cancelar", "remarcar", "tem horario". Mesmo que aparecam servicos especificos.
- HUMAN_ESCALATION: pedido EXPLICITO de humano/atendente.

Regra-chave SCHEDULE vs INTAKE (CRITICA):
- "quero MARCAR/AGENDAR [servico]" -> SCHEDULE (verbo de agendamento)
- "quero FAZER [servico]" -> INTAKE (declara desejo, sem pedir agenda)
- "tem horario...?" -> SCHEDULE
- "tenho [problema], o que voces fazem?" -> INTAKE

Regra-chave INTAKE vs TOPIC_KNOWLEDGE:
- Paciente SE COLOCA no problema ("tenho X", "minha pele", "estou com Y", "quero fazer Z") -> INTAKE
- Pergunta NEUTRA sobre procedimento ("como funciona X?", "X doi?", "posso fazer X estando Y?") -> TOPIC_KNOWLEDGE

Regras gerais:
- Multi-intent: detecte TODAS as intents aplicaveis. Cada uma com {intent, scope_text}.
- Mesma intent nao repete.
- OUTPUT OBRIGATORIO: JSON com EXATAMENTE 3 campos no nivel raiz: "intents" (array), "confidence" (float 0-1), "reasoning" (string). NUNCA omita confidence.
- confidence em [0,1] agregado. Se ambiguo, use < 0.70. Use sempre numerico.
- Sempre pt-BR. reasoning curto (<= 400 chars).

Exemplos:
- "voces aceitam Unimed? quero agendar" -> {"intents":[{"intent":"BUSINESS_INFO","scope_text":"voces aceitam Unimed?"},{"intent":"SCHEDULE","scope_text":"quero agendar"}],"confidence":0.88,"reasoning":"info de convenio + intencao de agendamento"}
- "posso fazer botox amamentando?" -> {"intents":[{"intent":"TOPIC_KNOWLEDGE","scope_text":"posso fazer botox amamentando?"}],"confidence":0.88,"reasoning":"pergunta neutra sobre contraindicacao"}
- "quero fazer botox" -> {"intents":[{"intent":"INTAKE","scope_text":"quero fazer botox"}],"confidence":0.90,"reasoning":"declaracao de interesse, sem verbo de agendamento"}
- "quero marcar uma limpeza de pele" -> {"intents":[{"intent":"SCHEDULE","scope_text":"quero marcar uma limpeza de pele"}],"confidence":0.92,"reasoning":"verbo 'marcar' -> SCHEDULE, mesmo com servico especifico"}
- "tenho marcas de expressao na testa, o que voces fazem?" -> {"intents":[{"intent":"INTAKE","scope_text":"tenho marcas de expressao na testa, o que voces fazem?"}],"confidence":0.85,"reasoning":"paciente expoe problema proprio buscando solucao"}
- "como funciona o preenchimento?" -> {"intents":[{"intent":"TOPIC_KNOWLEDGE","scope_text":"como funciona o preenchimento?"}],"confidence":0.90,"reasoning":"pergunta neutra"}
- "quero falar com atendente" -> {"intents":[{"intent":"HUMAN_ESCALATION","scope_text":"quero falar com atendente"}],"confidence":0.97,"reasoning":"pedido explicito de humano"}

Responda APENAS JSON valido com os 3 campos obrigatorios."""


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


def _build_default_lm(model: str, max_tokens: int) -> Optional[dspy.LM]:
    """Build the LM the router uses by default (mirrors GreetingAgent pattern).

    Reads ``DEEPSEEK_API_KEY`` from the environment. Returns None when the
    key is absent so callers can fall back to ``dspy.settings.lm`` for tests.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    try:
        return dspy.LM(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=ROUTER_TEMPERATURE,
        )
    except Exception as exc:
        log.error("router.lm_init_failed", error=str(exc))
        return None


class RouterAgent:
    """Iris primary router. Drop-in replacement for IrisRouterAgent.

    Runs on ``deepseek/deepseek-v4-flash`` (non-thinking mode) — same stack
    as ``GreetingAgent``. The agent self-manages its LM via
    ``DEEPSEEK_API_KEY``; it does NOT rely on ``init_dspy()`` global config.

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
        max_tokens: int = ROUTER_MAX_TOKENS,
        temperature: float = ROUTER_TEMPERATURE,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._lm_override = lm
        self._default_lm: Optional[dspy.LM] = None

    def _get_lm(self) -> dspy.LM:
        """Resolve the active LM in priority order:
        1. constructor override
        2. lazily-built default LM from DEEPSEEK_API_KEY
        3. dspy.settings.lm (for tests that already configured global)
        """
        if self._lm_override is not None:
            return self._lm_override
        if self._default_lm is None:
            self._default_lm = _build_default_lm(self.model, self.max_tokens)
        if self._default_lm is not None:
            return self._default_lm
        lm = dspy.settings.lm
        if lm is None:
            raise RuntimeError(
                "RouterAgent: no LM available. Set DEEPSEEK_API_KEY, call init_dspy(), "
                "or pass lm= to constructor."
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
        call_kwargs: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if ROUTER_EXTRA_BODY:
            call_kwargs["extra_body"] = ROUTER_EXTRA_BODY
        outputs = lm(**call_kwargs)
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
