"""
ScheduleRouter — Iris schedule sub-router.

Invoked by the pipeline whenever the primary RouterAgent emits SCHEDULE.
Receives a session-scoped sequence (from an upstream Manager agent) and
decides which sub-agent should act NOW.

Behavior:
  - Reads ``sequence`` (list of expected sub-intents in order) and
    ``current_stage`` (last sub-agent that ran, or "new").
  - Classifies the patient's latest message into ONE ScheduleIntent.
  - Decides ``is_deviation``: True if patient breaks the sequence (cancel /
    change / fallback).
  - Carries ``session_data`` forward (immutable read; sub-router never drops
    entries).

Runs on DeepSeek V4 Flash (non-thinking) — same stack as GreetingAgent and
RouterAgent. The agent self-manages its LM via ``DEEPSEEK_API_KEY``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import dspy
from pydantic import ValidationError

from app.core.telemetry import log

from .schedule_intents import (
    ScheduleIntent,
    FALLBACK_INTENT,
    is_deviation as _is_deviation,
)
from .schedule_schemas import ScheduleRouterOutput, SessionDataEntry


SCHEDULE_ROUTER_MODEL = "deepseek/deepseek-v4-flash"
SCHEDULE_ROUTER_TEMPERATURE = 0.0
SCHEDULE_ROUTER_MAX_TOKENS = 384
SCHEDULE_ROUTER_EXTRA_BODY: Dict[str, Any] = {"thinking": {"type": "disabled"}}

DEFAULT_SCHEDULE_CONFIDENCE_THRESHOLD = 0.70


SYSTEM_PROMPT = """Sub-router de agendamento da Iris (recepcionista IA para clinicas de estetica).
Voce recebe a ultima mensagem do paciente + uma SEQUENCIA esperada de sub-fluxo + o ESTAGIO ATUAL.
Decida qual SUB-INTENT deve atuar agora.

Sub-intents disponiveis (11):
- SCHEDULE_INTAKE: faz perguntas clinicas (ex: alergias, gestacao, condicao da pele).
- SCHEDULE_CASHIER: trata valores/pagamento do agendamento (consulta, sinal, parcelamento).
- SCHEDULE_EVALUATION: agenda a avaliacao com profissional ANTES do procedimento (consulta de analise).
- SCHEDULE_SERVICE: agenda a realizacao do procedimento em si (apos avaliacao + consentimento).
- SCHEDULE_SERVICE_PROTOCOL: envia orientacoes/preparos pre-procedimento.
- SCHEDULE_CONFIRMATION: confirma presenca em agendamento existente.
- SCHEDULE_REMINDER: lembrete 2h antes do agendamento.
- SCHEDULE_COMPLETION: estado terminal de um sub-fluxo (etapa atual concluida com sucesso).
- SCHEDULE_CANCEL: cancela um agendamento existente. (DESVIO)
- SCHEDULE_CHANGE: altera/remarca um agendamento existente. (DESVIO)
- SCHEDULE_FALLBACK: intencao nao identificada; pipeline pergunta recalibracao. (DESVIO)

Como decidir (PROCESSO):
1. Olhe sequence + current_stage. O proximo da sequencia (apos current_stage) e o CANDIDATO DEFAULT.
2. Leia latest_message. Se ela claramente segue o caminho feliz daquele sub-fluxo, emita o candidato default com is_deviation=false.
3. Se a mensagem indica CANCELAR um agendamento existente -> SCHEDULE_CANCEL, is_deviation=true.
4. Se indica REMARCAR/ALTERAR -> SCHEDULE_CHANGE, is_deviation=true.
5. Se voce nao consegue decidir entre opcoes razoaveis ou a mensagem nao se encaixa -> SCHEDULE_FALLBACK, is_deviation=true.
6. Caso especial: se current_stage e o ULTIMO da sequencia OU e SCHEDULE_COMPLETION, e mensagem segue o fluxo, emita SCHEDULE_COMPLETION com is_deviation=false.

Regras gerais:
- next_intent: EXATAMENTE um dos 11 valores acima.
- is_deviation: true SE e SOMENTE SE next_intent for CANCEL, CHANGE ou FALLBACK.
- session_data: copie o array recebido. NUNCA remova entradas; voce pode adicionar uma nova entry se a mensagem traz info nova relevante.
- confidence em [0,1] agregado. Se ambiguo, use < 0.70 (caller forca FALLBACK).
- Sempre pt-BR. reasoning curto (<= 400 chars).

OUTPUT OBRIGATORIO: JSON com EXATAMENTE 5 campos no nivel raiz: "next_intent" (string), "is_deviation" (boolean), "session_data" (array), "confidence" (float 0-1), "reasoning" (string).

Exemplos:

# Caminho feliz (segue sequencia)
Input: sequence=["SCHEDULE_INTAKE","SCHEDULE_CASHIER","SCHEDULE_EVALUATION","SCHEDULE_COMPLETION"], current_stage="new", latest_message="oi, quero saber sobre avaliacao de fotona"
-> {"next_intent":"SCHEDULE_INTAKE","is_deviation":false,"session_data":[],"confidence":0.9,"reasoning":"primeiro contato no fluxo de avaliacao, comeca por INTAKE"}

Input: sequence=["SCHEDULE_INTAKE","SCHEDULE_CASHIER","SCHEDULE_EVALUATION","SCHEDULE_COMPLETION"], current_stage="SCHEDULE_INTAKE", latest_message="ja respondi tudo, nao tomo nenhum remedio"
-> {"next_intent":"SCHEDULE_CASHIER","is_deviation":false,"session_data":[],"confidence":0.9,"reasoning":"paciente respondeu intake, segue para cashier (proximo da sequencia)"}

Input: sequence=["SCHEDULE_CONFIRMATION","SCHEDULE_COMPLETION"], current_stage="SCHEDULE_CONFIRMATION", latest_message="sim, confirmo presenca"
-> {"next_intent":"SCHEDULE_COMPLETION","is_deviation":false,"session_data":[],"confidence":0.95,"reasoning":"paciente confirmou, ritual concluido"}

# Desvios
Input: sequence=["SCHEDULE_INTAKE","SCHEDULE_CASHIER","SCHEDULE_EVALUATION","SCHEDULE_COMPLETION"], current_stage="SCHEDULE_INTAKE", latest_message="esquece, quero cancelar minha consulta"
-> {"next_intent":"SCHEDULE_CANCEL","is_deviation":true,"session_data":[],"confidence":0.95,"reasoning":"paciente desistiu no meio do intake, pediu cancelar"}

Input: sequence=["SCHEDULE_REMINDER","SCHEDULE_COMPLETION"], current_stage="SCHEDULE_REMINDER", latest_message="preciso remarcar pra semana que vem"
-> {"next_intent":"SCHEDULE_CHANGE","is_deviation":true,"session_data":[],"confidence":0.92,"reasoning":"paciente pediu remarcar (desvio do reminder)"}

Input: sequence=["SCHEDULE_INTAKE","SCHEDULE_CASHIER","SCHEDULE_EVALUATION","SCHEDULE_COMPLETION"], current_stage="new", latest_message="kkkkk"
-> {"next_intent":"SCHEDULE_FALLBACK","is_deviation":true,"session_data":[],"confidence":0.4,"reasoning":"mensagem sem conteudo classificavel"}

Responda APENAS JSON valido com os 5 campos obrigatorios."""


def _build_default_lm(model: str, max_tokens: int) -> Optional[dspy.LM]:
    """Build the LM the schedule sub-router uses by default."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    try:
        return dspy.LM(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=SCHEDULE_ROUTER_TEMPERATURE,
        )
    except Exception as exc:
        log.error("schedule_router.lm_init_failed", error=str(exc))
        return None


class ScheduleRouter:
    """Iris schedule sub-router.

    Runs on ``deepseek/deepseek-v4-flash`` (non-thinking). Auto-manages LM via
    ``DEEPSEEK_API_KEY``. Drop-in usable without ``init_dspy()`` global config.

    Usage:
        router = ScheduleRouter()
        out = router.forward(
            latest_message="ja respondi tudo, sem alergias",
            history=[],
            sequence=["SCHEDULE_INTAKE", "SCHEDULE_CASHIER",
                      "SCHEDULE_EVALUATION", "SCHEDULE_COMPLETION"],
            current_stage="SCHEDULE_INTAKE",
            session_data=[{"name": "evaluation", "data": {"service": "fotona"}}],
        )
        # out["next_intent"] -> "SCHEDULE_CASHIER"
        # out["is_deviation"] -> False
        # out["session_data"] -> [{"name": "evaluation", ...}]
        # out["confidence"] -> 0.91
    """

    def __init__(
        self,
        lm: Optional[dspy.LM] = None,
        model: str = SCHEDULE_ROUTER_MODEL,
        max_tokens: int = SCHEDULE_ROUTER_MAX_TOKENS,
        temperature: float = SCHEDULE_ROUTER_TEMPERATURE,
        confidence_threshold: float = DEFAULT_SCHEDULE_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self._lm_override = lm
        self._default_lm: Optional[dspy.LM] = None

    def _get_lm(self) -> dspy.LM:
        if self._lm_override is not None:
            return self._lm_override
        if self._default_lm is None:
            self._default_lm = _build_default_lm(self.model, self.max_tokens)
        if self._default_lm is not None:
            return self._default_lm
        lm = dspy.settings.lm
        if lm is None:
            raise RuntimeError(
                "ScheduleRouter: no LM available. Set DEEPSEEK_API_KEY, call init_dspy(), "
                "or pass lm= to constructor."
            )
        return lm

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "Sem historico anterior."
        lines = []
        for turn in history[-10:]:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            prefix = "Paciente" if role == "human" else role
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def _format_session_data(self, session_data: List[Dict[str, Any]]) -> str:
        if not session_data:
            return "[]"
        try:
            return json.dumps(session_data, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(session_data)

    def _build_user_prompt(
        self,
        latest_message: str,
        history: List[Dict[str, str]],
        sequence: List[str],
        current_stage: str,
        session_data: List[Dict[str, Any]],
    ) -> str:
        return (
            f"sequence: {sequence}\n"
            f"current_stage: {current_stage}\n"
            f"session_data: {self._format_session_data(session_data)}\n"
            f"latest_message: {latest_message}\n"
            f"history:\n{self._format_history(history)}"
        )

    def _call_lm(self, user_prompt: str) -> str:
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
        if SCHEDULE_ROUTER_EXTRA_BODY:
            call_kwargs["extra_body"] = SCHEDULE_ROUTER_EXTRA_BODY
        outputs = lm(**call_kwargs)
        if not outputs:
            raise ValueError("schedule sub-router LM returned no outputs")
        return outputs[0]

    def _parse(self, raw_content: str) -> ScheduleRouterOutput:
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"schedule sub-router LM returned non-JSON content: {exc}"
            ) from exc
        try:
            return ScheduleRouterOutput.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"schedule sub-router output failed Pydantic validation: {exc}"
            ) from exc

    def forward(
        self,
        latest_message: str,
        history: List[Dict[str, str]],
        sequence: List[str],
        current_stage: str,
        session_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run the schedule sub-router. Propagates exceptions on failure.

        Returns:
            {
                "next_intent": str,           # one of ScheduleIntent values
                "is_deviation": bool,
                "session_data": list[dict],   # propagated forward (never drops)
                "confidence": float,
                "reasoning": str,
            }

        Below the confidence threshold OR on Pydantic mismatch with
        ``is_deviation`` (e.g. LLM emits CANCEL but is_deviation=false), the
        agent forces SCHEDULE_FALLBACK + is_deviation=True so the pipeline
        can ask for recalibration.
        """
        session_data = list(session_data or [])

        user_prompt = self._build_user_prompt(
            latest_message=latest_message,
            history=history,
            sequence=sequence,
            current_stage=current_stage,
            session_data=session_data,
        )

        try:
            raw_content = self._call_lm(user_prompt)
            parsed = self._parse(raw_content)
        except Exception as exc:
            log.error(
                "schedule_router.failed",
                error=str(exc),
                error_type=type(exc).__name__,
                model=self.model,
            )
            raise

        confidence = max(0.0, min(1.0, float(parsed.confidence)))
        reasoning = parsed.reasoning.strip()

        # Reconcile is_deviation: the canonical rule (intent -> deviation flag)
        # wins over what the LLM put in the JSON. Keeps callers consistent.
        canonical_deviation = _is_deviation(parsed.next_intent)
        if parsed.is_deviation != canonical_deviation:
            log.warn(
                "schedule_router.deviation_mismatch",
                emitted=parsed.is_deviation,
                canonical=canonical_deviation,
                intent=parsed.next_intent.value,
            )

        # Below threshold => force FALLBACK.
        if confidence < self.confidence_threshold:
            next_intent = FALLBACK_INTENT
            is_dev_final = True
        else:
            next_intent = parsed.next_intent
            is_dev_final = canonical_deviation

        # session_data passthrough: prefer what the LLM emitted (it can add
        # entries based on the message); fall back to the input if it dropped
        # everything.
        emitted_session = [entry.model_dump() for entry in parsed.session_data]
        if not emitted_session and session_data:
            emitted_session = session_data

        return {
            "next_intent": next_intent.value,
            "is_deviation": is_dev_final,
            "session_data": emitted_session,
            "confidence": confidence,
            "reasoning": reasoning,
        }
