"""ScheduleIntakeAgent — clinical intake sub-agent of the Iris schedule flow.

Spec: kb/07-MVP/Tech/03-Discussoes/schedule/01 - Spec SCHEDULE_INTAKE.md

Behavior summary:
  - Receives a pre-loaded ``questions`` list (clinic baseline + service
    override union, ordered by ``order``) and a ``contraindications`` list
    pulled from the service's ``sf_clinic_services.contraindications`` column.
  - Picks the first not-yet-answered question by dynamic lookup against
    ``intake_answers`` already stored in ``session_data["evaluation"]["data"]``.
  - When the patient has replied, calls the LLM once to (a) parse one or more
    answers and (b) decide whether each answer semantically matches an entry
    of ``contraindications`` (no Python regex; the LLM owns this).
  - Appends parsed answers to ``intake_answers`` (append-only).
  - If any answer matched a contraindication, sets
    ``next_hint="ESCALATE_TO_HUMAN"`` and stops asking further questions.
  - When all ``is_required=True`` questions have been answered (and no
    contraindication was matched), sets ``sub_intent_complete=True``.

Stack: DeepSeek V4 Flash, temperature 0.0, ``thinking`` disabled.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional

import dspy
from pydantic import BaseModel, Field, ValidationError

from app.core.telemetry import log

from .schemas import IntakeAnswer, IntakeData, IntakeOutput


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


INTAKE_MODEL = "deepseek/deepseek-v4-flash"
INTAKE_TEMPERATURE = 0.0
INTAKE_MAX_TOKENS = 384
# DeepSeek standard inference — no provider-specific extras.
INTAKE_EXTRA_BODY: Dict[str, Any] = {}

DEFAULT_PARSE_CONFIDENCE_THRESHOLD = 0.70

ESCALATION_HINT = "ESCALATE_TO_HUMAN"


_SYSTEM_PROMPT_TEMPLATE = """Voce e a __ASSISTANT__, recepcionista virtual da __CLINIC__.
Etapa atual: INTAKE clinico (perguntas pre-procedimento). Voce faz UMA pergunta por turno e parseia a resposta do paciente.

Voce recebe:
- A proxima pergunta a fazer (next_question) com id/categoria/texto.
- A lista de perguntas ja respondidas (answered_question_ids).
- A lista de TODAS as perguntas (questions) com id, order, category, question_text.
- A lista de contraindicacoes (contraindications) do servico de interesse.
- A ultima mensagem do paciente (latest_message) e o historico (history).

Sua tarefa:
1. Parsear ate N respostas que o paciente claramente deu nesta ultima mensagem. Aceite apenas matches com confianca >= 0.70. So pode parsear respostas para question_ids que AINDA NAO foram respondidos (i.e. NAO estao em answered_question_ids).
2. Para cada resposta parseada, avalie se o conteudo casa SEMANTICAMENTE com algum termo da lista `contraindications` do servico. Sinonimos coloquiais, nomes de medicamentos, e historico ambiguo CONTAM. Em duvida, FAVORECER falso positivo (preencher matched_contraindication). Se a lista contraindications esta vazia, matched_contraindication DEVE ser null sempre.
3. Gerar `next_question_text`: uma variante natural em pt-BR da proxima pergunta nao-respondida. Curta, empatica, sem markdown. Se nao ha proxima pergunta (todas respondidas OU escalation), envie string vazia.

Regras:
- NUNCA diagnostique nem prescreva. Voce apenas coleta.
- pt-BR, sem markdown, sem emojis.
- Se o paciente mudou de assunto (ex: perguntou de valores), NAO invente resposta clinica: deixe parsed_answers vazia e gere next_question_text re-pergunta da pergunta atual com um curto reconhecimento ("antes de continuar...").
- Se o paciente recusou ("prefiro nao dizer"), parseie a resposta literal e nao escale.

OUTPUT OBRIGATORIO: JSON com EXATAMENTE 3 campos no nivel raiz:
- "next_question_text": string (pode ser vazia)
- "parsed_answers": array de objetos com chaves "question_id" (string), "answer" (string), "matched_contraindication" (string ou null)
- "reasoning": string curta (<=400 chars)

Responda APENAS JSON valido."""


def _render_system_prompt(clinic_name: str, assistant_name: str) -> str:
    return (
        _SYSTEM_PROMPT_TEMPLATE
        .replace("__ASSISTANT__", assistant_name)
        .replace("__CLINIC__", clinic_name)
    )


# ---------------------------------------------------------------------------
# LM helpers (same pattern as ScheduleRouter / KnowledgeAgent)
# ---------------------------------------------------------------------------


def _build_default_lm(model: str, max_tokens: int) -> Optional[dspy.LM]:
    """Build the LM the intake agent uses by default."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    try:
        return dspy.LM(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=INTAKE_TEMPERATURE,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("schedule_intake.lm_init_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# LLM output schema (raw shape, distinct from IntakeOutput)
# ---------------------------------------------------------------------------


class _ParsedAnswer(BaseModel):
    question_id: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    matched_contraindication: Optional[str] = None


class _LLMOutput(BaseModel):
    next_question_text: str = ""
    parsed_answers: List[_ParsedAnswer] = Field(default_factory=list)
    reasoning: str = Field(default="", max_length=400)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


_REQUIRED_QUESTION_KEYS = ("id", "order", "question_text", "category")


def _validate_questions(questions: List[Dict[str, Any]]) -> None:
    for q in questions:
        missing = [k for k in _REQUIRED_QUESTION_KEYS if k not in q]
        if missing:
            raise ValueError(
                f"malformed question (missing keys {missing}): {q}"
            )


def _find_evaluation_entry(session_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    for entry in session_data:
        if entry.get("name") == "evaluation":
            return entry
    raise ValueError(
        'session_data missing required entry name="evaluation"'
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ScheduleIntakeAgent:
    """Clinical intake sub-agent — one question per turn, ordered by ``order``.

    Auto-manages its DeepSeek LM via ``DEEPSEEK_API_KEY``. Drop-in usable
    without ``init_dspy()`` global config.
    """

    def __init__(
        self,
        lm: Optional[dspy.LM] = None,
        model: str = INTAKE_MODEL,
        max_tokens: int = INTAKE_MAX_TOKENS,
        temperature: float = INTAKE_TEMPERATURE,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._lm_override = lm
        self._default_lm: Optional[dspy.LM] = None

    # -- LM management ------------------------------------------------------

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
                "ScheduleIntakeAgent: no LM available. Set DEEPSEEK_API_KEY, "
                "call init_dspy(), or pass lm= to constructor."
            )
        return lm

    # -- Prompt building ----------------------------------------------------

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "(sem historico)"
        lines = []
        for turn in history[-10:]:
            role = turn.get("role", "?")
            content = turn.get("content", "")
            prefix = "Paciente" if role in ("human", "patient") else role
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def _build_user_prompt(
        self,
        *,
        latest_message: str,
        history: List[Dict[str, str]],
        questions: List[Dict[str, Any]],
        next_question: Optional[Dict[str, Any]],
        answered_ids: List[str],
        contraindications: List[str],
        service: str,
    ) -> str:
        next_q_block = (
            json.dumps(next_question, ensure_ascii=False)
            if next_question
            else "null (nenhuma pergunta pendente)"
        )
        return (
            f"service: {service}\n"
            f"contraindications: {json.dumps(contraindications, ensure_ascii=False)}\n"
            f"questions: {json.dumps(questions, ensure_ascii=False)}\n"
            f"answered_question_ids: {json.dumps(answered_ids)}\n"
            f"next_question: {next_q_block}\n"
            f"latest_message: {latest_message or ''}\n"
            f"history:\n{self._format_history(history)}\n\n"
            "Responda em JSON conforme as regras."
        )

    # -- LM call + parse ----------------------------------------------------

    def _call_lm(self, system: str, user_prompt: str) -> str:
        lm = self._get_lm()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        call_kwargs: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if INTAKE_EXTRA_BODY:
            call_kwargs["extra_body"] = INTAKE_EXTRA_BODY
        outputs = lm(**call_kwargs)
        if not outputs:
            raise ValueError("schedule_intake LM returned no outputs")
        return outputs[0]

    def _parse_llm(self, raw: str) -> _LLMOutput:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"schedule_intake LM returned non-JSON content: {exc}"
            ) from exc
        try:
            return _LLMOutput.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"schedule_intake LM output failed Pydantic validation: {exc}"
            ) from exc

    # -- Core helpers -------------------------------------------------------

    @staticmethod
    def _sorted_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            questions,
            key=lambda q: (
                q["order"],
                0 if q.get("source") == "clinic" else 1,
                q["id"],
            ),
        )

    @staticmethod
    def _pick_next_question(
        questions: List[Dict[str, Any]],
        answered_ids: List[str],
    ) -> Optional[Dict[str, Any]]:
        answered = set(answered_ids)
        for q in ScheduleIntakeAgent._sorted_questions(questions):
            if q["id"] not in answered:
                return q
        return None

    @staticmethod
    def _all_required_answered(
        questions: List[Dict[str, Any]], answered_ids: List[str]
    ) -> bool:
        answered = set(answered_ids)
        return all(q["id"] in answered for q in questions if q.get("is_required", True))

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        latest_message: str,
        history: List[Dict[str, str]],
        session_data: List[Dict[str, Any]],
        clinic_id: str,
        service: str,
        questions: List[Dict[str, Any]],
        contraindications: Optional[List[str]] = None,
        clinic_name: str = "nossa clínica",
        assistant_name: str = "Iris",
        dry_run: bool = False,
        *args: Any,
    ) -> Dict[str, Any]:
        """See spec §7 for full behavior contract.

        Args:
            latest_message: latest patient message (empty on cold start).
            history: prior conversation turns.
            session_data: cross-agent state list, must contain an "evaluation" entry.
            clinic_id: clinic uuid (audit/log only).
            service: service of interest (must be present in evaluation entry).
            questions: pre-loaded intake questions (pipeline responsibility).
            contraindications: per-service contraindication terms (may be empty).
            clinic_name: clinic display name for prompt context.
            assistant_name: assistant display name for prompt context.
            dry_run: when True, skips side-effects (currently a no-op — agent
                has no I/O of its own).
            *args: ignored.

        Returns:
            Dict matching IntakeOutput schema (messages + conversation_stage
            + reasoning + data).

        Raises:
            ValueError: when session_data is malformed, questions malformed,
                or LM output fails validation.
            RuntimeError: when no LM is available.
        """
        contraindications = list(contraindications or [])

        # ---- Validate inputs -----------------------------------------------
        evaluation_entry = _find_evaluation_entry(session_data)
        eval_data: Dict[str, Any] = evaluation_entry.setdefault("data", {})
        if "service" not in eval_data:
            raise ValueError(
                'session_data["evaluation"]["data"] missing "service"'
            )
        _validate_questions(questions)

        # Deep-copy session_data to keep this function pure for the caller.
        session_data = copy.deepcopy(session_data)
        evaluation_entry = _find_evaluation_entry(session_data)
        eval_data = evaluation_entry["data"]

        intake_answers: List[Dict[str, Any]] = list(
            eval_data.get("intake_answers") or []
        )

        # ---- Empty questions => immediate completion -----------------------
        if not questions:
            return _build_envelope(
                messages_text=(
                    "Pronto! Não preciso de mais nada nesta etapa, podemos seguir."
                ),
                session_data=session_data,
                sub_intent_complete=True,
                next_hint=None,
                intake_answers=intake_answers,
                next_question_id=None,
                escalation_reason=None,
                reasoning="source=agent | questions=[] (clinica sem intake configurado)",
            )

        # ---- Call LM (single shot: parse + contraindication eval + variant)-
        answered_ids = [a["question_id"] for a in intake_answers]
        next_question = self._pick_next_question(questions, answered_ids)

        system_prompt = _render_system_prompt(clinic_name, assistant_name)
        user_prompt = self._build_user_prompt(
            latest_message=latest_message,
            history=history,
            questions=questions,
            next_question=next_question,
            answered_ids=answered_ids,
            contraindications=contraindications,
            service=service,
        )

        raw = self._call_lm(system_prompt, user_prompt)
        llm_out = self._parse_llm(raw)

        # ---- Append parsed answers (append-only, dedup by question_id) -----
        questions_by_id = {q["id"]: q for q in questions}
        answered_set = set(answered_ids)
        new_answers: List[IntakeAnswer] = []
        for pa in llm_out.parsed_answers:
            if pa.question_id in answered_set:
                # Defensive: LLM should not re-answer; skip silently.
                continue
            q = questions_by_id.get(pa.question_id)
            if q is None:
                # LLM hallucinated a question_id; skip.
                continue
            matched = pa.matched_contraindication
            # Defense-in-depth: when contraindications is empty, force null.
            if not contraindications:
                matched = None
            new_answers.append(
                IntakeAnswer(
                    question_id=pa.question_id,
                    question_text=q["question_text"],
                    category=q["category"],
                    answer=pa.answer,
                    matched_contraindication=matched,
                )
            )
            answered_set.add(pa.question_id)

        intake_answers.extend([a.model_dump() for a in new_answers])

        # ---- Detect contraindication (escalation) --------------------------
        escalated_answer = next(
            (a for a in new_answers if a.matched_contraindication), None
        )

        if escalated_answer is not None:
            escalation_reason = (
                f"Paciente apresenta indicador compatível com "
                f"'{escalated_answer.matched_contraindication}'. "
                f"Serviço '{service}' contraindicado — encaminhar para avaliação humana."
            )
            empathic_msg = (
                "Entendi. Por causa disso, prefiro conectar você com nossa equipe "
                "de especialistas pra avaliar com cuidado. Um momento, por favor."
            )
            eval_data["intake_answers"] = intake_answers
            return _build_envelope(
                messages_text=empathic_msg,
                session_data=session_data,
                sub_intent_complete=False,
                next_hint=ESCALATION_HINT,
                intake_answers=intake_answers,
                next_question_id=None,
                escalation_reason=escalation_reason,
                reasoning=(
                    f"source=llm | matched_contraindication="
                    f"{escalated_answer.matched_contraindication} em "
                    f"{escalated_answer.question_id}. Escalando."
                )[:400],
            )

        # ---- Re-compute next question -------------------------------------
        next_question = self._pick_next_question(questions, list(answered_set))

        # ---- All required answered, no escalation => complete --------------
        if next_question is None and self._all_required_answered(
            questions, list(answered_set)
        ):
            eval_data["intake_answers"] = intake_answers
            return _build_envelope(
                messages_text=(
                    "Perfeito, anotei tudo. Vamos seguir para os próximos passos."
                ),
                session_data=session_data,
                sub_intent_complete=True,
                next_hint=None,
                intake_answers=intake_answers,
                next_question_id=None,
                escalation_reason=None,
                reasoning="source=llm | todas required respondidas; intake completo",
            )

        # ---- Otherwise ask the next question -------------------------------
        eval_data["intake_answers"] = intake_answers
        next_text = (llm_out.next_question_text or "").strip()
        if not next_text and next_question is not None:
            next_text = next_question["question_text"]

        return _build_envelope(
            messages_text=next_text,
            session_data=session_data,
            sub_intent_complete=False,
            next_hint=None,
            intake_answers=intake_answers,
            next_question_id=next_question["id"] if next_question else None,
            escalation_reason=None,
            reasoning=(
                f"source=llm | answered={len(intake_answers)}; "
                f"next={next_question['id'] if next_question else 'None'}"
            )[:400],
        )


# ---------------------------------------------------------------------------
# Envelope builder (kept module-level so tests / pipeline can rely on shape)
# ---------------------------------------------------------------------------


def _build_envelope(
    *,
    messages_text: str,
    session_data: List[Dict[str, Any]],
    sub_intent_complete: bool,
    next_hint: Optional[str],
    intake_answers: List[Dict[str, Any]],
    next_question_id: Optional[str],
    escalation_reason: Optional[str],
    reasoning: str,
) -> Dict[str, Any]:
    data_payload = IntakeData(
        session_data=session_data,
        sub_intent_complete=sub_intent_complete,
        next_hint=next_hint,
        intake_answers=[IntakeAnswer(**a) for a in intake_answers],
        next_question_id=next_question_id,
        escalation_reason=escalation_reason,
    )
    output = IntakeOutput(
        messages=[{"type": "text", "content": messages_text}],
        reasoning=reasoning,
        data=data_payload.model_dump(),
    )
    return output.model_dump()
