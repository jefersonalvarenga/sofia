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
from datetime import datetime, timezone
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

# GREETING-composition stale threshold. >24h since the last interaction
# triggers the "patient is coming back after a long gap" branch.
STALE_THRESHOLD_HOURS = 24


SYSTEM_PROMPT = """Router da Iris (recepcionista IA para clinicas de estetica). Classifique a ultima mensagem do paciente.

Intents (8):
- BUSINESS_INFO: preco, horario, endereco, convenio, lista de servicos.
- TOPIC_KNOWLEDGE: pergunta NEUTRA sobre como servico funciona, contraindicacoes, recuperacao. Paciente perguntando sem se colocar como interessado.
- REENGAGE: retomada de conversa inativa (use conversation_stage como dica).
- GREETING: "oi", "bom dia", primeira mensagem sem outra intencao OU resposta social a uma saudacao anterior do assistente ("td bem", "blz e vc?", "tudo otimo").
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

Composicao com GREETING (REGRA NOVA):
GREETING pode coexistir com outras intents em 3 casos. Quando algum desses
gatilhos for verdadeiro E houver outra intent na mensagem, emita
[GREETING, <specialist>] (informacional -> CTA) em vez de bloquear GREETING:
  (a) cold_start=true   — primeira mensagem do paciente nessa session.
  (b) Resposta social   — paciente respondeu socialmente a uma saudacao
                          anterior do assistente ("td bem", "blz e vc",
                          "tudo otimo, e voce?"). Voce detecta isso lendo
                          o history e a latest_message.
  (c) stale=true        — paciente voltou apos longo tempo (>24h). A flag
                          eh fornecida no bloco context_flags.

context_flags (block no user prompt): "cold_start" e "stale" sao booleans.
"last_interaction_hours_ago" eh int ou null. Use esses flags como verdade
absoluta — nao tente recalcular pelo conteudo do history.

Quando o paciente apenas cumprimentou sem outro pedido ("oi", "td bem"),
continue emitindo apenas [GREETING] como antes.

Exemplos:
- "voces aceitam Unimed? quero agendar" -> {"intents":[{"intent":"BUSINESS_INFO","scope_text":"voces aceitam Unimed?"},{"intent":"SCHEDULE","scope_text":"quero agendar"}],"confidence":0.88,"reasoning":"info de convenio + intencao de agendamento"}
- "posso fazer botox amamentando?" -> {"intents":[{"intent":"TOPIC_KNOWLEDGE","scope_text":"posso fazer botox amamentando?"}],"confidence":0.88,"reasoning":"pergunta neutra sobre contraindicacao"}
- "quero fazer botox" -> {"intents":[{"intent":"INTAKE","scope_text":"quero fazer botox"}],"confidence":0.90,"reasoning":"declaracao de interesse, sem verbo de agendamento"}
- "quero marcar uma limpeza de pele" -> {"intents":[{"intent":"SCHEDULE","scope_text":"quero marcar uma limpeza de pele"}],"confidence":0.92,"reasoning":"verbo 'marcar' -> SCHEDULE, mesmo com servico especifico"}
- "tenho marcas de expressao na testa, o que voces fazem?" -> {"intents":[{"intent":"INTAKE","scope_text":"tenho marcas de expressao na testa, o que voces fazem?"}],"confidence":0.85,"reasoning":"paciente expoe problema proprio buscando solucao"}
- "como funciona o preenchimento?" -> {"intents":[{"intent":"TOPIC_KNOWLEDGE","scope_text":"como funciona o preenchimento?"}],"confidence":0.90,"reasoning":"pergunta neutra"}
- "quero falar com atendente" -> {"intents":[{"intent":"HUMAN_ESCALATION","scope_text":"quero falar com atendente"}],"confidence":0.97,"reasoning":"pedido explicito de humano"}

Exemplos de composicao com GREETING (3 gatilhos):
- cold_start=true + "quero agendar botox" -> {"intents":[{"intent":"GREETING","scope_text":"quero agendar botox"},{"intent":"SCHEDULE","scope_text":"quero agendar botox"}],"confidence":0.92,"reasoning":"cold start + intencao de agendamento"}
- cold_start=true + "oi" -> {"intents":[{"intent":"GREETING","scope_text":"oi"}],"confidence":0.95,"reasoning":"so saudacao, sem outro pedido"}
- stale=true (>24h) + "quero remarcar minha consulta" -> {"intents":[{"intent":"GREETING","scope_text":"quero remarcar minha consulta"},{"intent":"SCHEDULE","scope_text":"quero remarcar minha consulta"}],"confidence":0.90,"reasoning":"retomada apos longo tempo + remarcacao"}
- assistente acabou de saudar e paciente respondeu "td bem e vc, quero marcar limpeza" -> {"intents":[{"intent":"GREETING","scope_text":"td bem e vc"},{"intent":"SCHEDULE","scope_text":"quero marcar limpeza"}],"confidence":0.90,"reasoning":"resposta social + agendamento"}
- assistente saudou e paciente so respondeu "td bem e vc" -> {"intents":[{"intent":"GREETING","scope_text":"td bem e vc"}],"confidence":0.92,"reasoning":"resposta social pura"}

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


def _compute_context_flags(
    history: List[Dict[str, str]],
    last_interaction_at: Optional[datetime],
) -> Dict[str, Any]:
    """Compute the ``context_flags`` block carried in the user prompt.

    Pure / side-effect-free. Returns a dict shaped as::

        {
            "cold_start": bool,
            "stale": bool,
            "last_interaction_hours_ago": Optional[int],
        }

    Rules:
      - ``cold_start`` is True when ``history`` is empty (first patient
        message in the session). Independent of ``last_interaction_at``.
      - ``stale`` is True when ``last_interaction_at`` is set AND older than
        :data:`STALE_THRESHOLD_HOURS`. If ``last_interaction_at`` is naive,
        it is assumed to be UTC.
      - ``last_interaction_hours_ago`` is the integer hour gap, floored. None
        when ``last_interaction_at`` is None.

    The "patient responded socially to a greeting" trigger is intentionally
    NOT detected here — the LLM makes that call semantically, reading the
    history that ``_format_history`` already carries in the prompt.
    """
    cold_start = not history

    hours_ago: Optional[int] = None
    stale = False
    if last_interaction_at is not None:
        ts = last_interaction_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        hours_ago = max(0, int(delta.total_seconds() // 3600))
        stale = hours_ago > STALE_THRESHOLD_HOURS

    return {
        "cold_start": cold_start,
        "stale": stale,
        "last_interaction_hours_ago": hours_ago,
    }


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
        context_flags: Dict[str, Any],
        patient_name: Optional[str] = None,
    ) -> str:
        # context_flags is rendered as a flat block (instead of JSON) so the
        # LLM picks up the boolean values via simple substring matching. The
        # SYSTEM_PROMPT references these exact field names.
        hours_ago = context_flags.get("last_interaction_hours_ago")
        hours_repr = "null" if hours_ago is None else str(int(hours_ago))
        flags_block = (
            "context_flags:\n"
            f"  cold_start: {str(bool(context_flags.get('cold_start'))).lower()}\n"
            f"  stale: {str(bool(context_flags.get('stale'))).lower()}\n"
            f"  last_interaction_hours_ago: {hours_repr}"
        )
        name_line = (
            f"patient_name: {patient_name}\n" if patient_name else ""
        )
        return (
            f"latest_message: {latest_message}\n"
            f"conversation_stage: {conversation_stage}\n"
            f"{name_line}"
            f"{flags_block}\n"
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
        last_interaction_at: Optional[datetime] = None,
        patient_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the router. Propagates exceptions on failure (no silent capture).

        Args:
            latest_message: The patient's most recent message.
            history: Prior turns ``[{"role": "human"|"assistant", "content": ...}]``.
            conversation_stage: Session stage hint (e.g. ``"new"``, ``"active"``).
            last_interaction_at: When the assistant last interacted with the
                patient (UTC). ``None`` for first-ever contact. Used to compute
                the ``stale`` flag (>24h) carried in the user prompt so the LLM
                can compose ``[GREETING, <specialist>]`` when retrieving a
                conversation after a long gap.
            patient_name: Patient's display name, if known. Optional — surfaced
                in the user prompt for context only; not required for routing.

        Returns a dict shaped:
            {
                "intents": [{"intent": str, "scope_text": str}, ...],
                "detected_intents": [str, ...],
                "reasoning": str,
                "confidence": float,
            }
        """
        threshold = _read_threshold()
        context_flags = _compute_context_flags(history, last_interaction_at)
        user_prompt = self._build_user_prompt(
            latest_message,
            history,
            conversation_stage,
            context_flags,
            patient_name=patient_name,
        )

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
