import dspy
from enum import Enum


class SofiaIntentType(str, Enum):
    GREETING = "GREETING"
    FAQ = "FAQ"
    SCHEDULE = "SCHEDULE"
    HUMAN_ESCALATION = "HUMAN_ESCALATION"
    REENGAGE = "REENGAGE"
    UNCLASSIFIED = "UNCLASSIFIED"


class SofiaRouterSignature(dspy.Signature):
    """
    You are the Router Agent for Sofia, an AI receptionist for aesthetic/medical clinics on WhatsApp.
    Your job is to classify the patient's latest message into one or more intents and detect the patient's language.

    Intent definitions:
    - GREETING: Patient initiates conversation, says hello, or sends a first message.
    - FAQ: Patient asks about services, prices, clinic address, opening hours, insurance/convenios,
           procedures, recovery, or any general information about the clinic.
    - SCHEDULE: Patient wants to book, schedule, or confirm an appointment. Also use when patient
                is in the middle of a scheduling conversation (selecting service, choosing a slot).
    - HUMAN_ESCALATION: Patient explicitly asks to speak with a human, attendant, or receptionist.
    - REENGAGE: Patient resumes a conversation that was paused or idle.
    - UNCLASSIFIED: None of the above applies clearly.

    Multi-intent rules:
    1. A message may trigger multiple intents — detect ALL that apply.
    2. Return intents as a comma-separated list ordered from informational to most important (CTA last).
       Priority order (most important = last): HUMAN_ESCALATION > SCHEDULE > REENGAGE > FAQ > GREETING
    3. HUMAN_ESCALATION always appears last if present, as it overrides all other actions.
    4. When only one intent applies, return just that single value.
    5. Use `conversation_stage` for context — a patient mid-scheduling likely means SCHEDULE.
    6. Keep reasoning concise (max 200 chars).

    Language detection:
    - Detect the patient's language from the message text.
    - Return BCP-47 language tags (e.g. "pt-BR", "es", "en").
    - Default to "pt-BR" if the language is ambiguous or cannot be determined.

    Examples:
    - "oi" → detected_intents: "GREETING", language: "pt-BR"
    - "quanto custa limpeza?" → detected_intents: "FAQ", language: "pt-BR"
    - "quero agendar uma limpeza, vocês aceitam Unimed?" → detected_intents: "FAQ,SCHEDULE", language: "pt-BR"
    - "quero falar com atendente" → detected_intents: "HUMAN_ESCALATION", language: "pt-BR"
    """

    latest_message = dspy.InputField(desc="Latest message from the patient.")
    history_str = dspy.InputField(desc="Conversation history as formatted string. 'Sem histórico.' if empty.")
    conversation_stage = dspy.InputField(desc="Current conversation stage (e.g., 'new', 'presenting_slots', 'booked').")

    detected_intents = dspy.OutputField(desc="Comma-separated intents ordered from informational to most important (CTA last). E.g. 'FAQ,SCHEDULE' or 'GREETING'. Valid values: GREETING | FAQ | SCHEDULE | REENGAGE | HUMAN_ESCALATION | UNCLASSIFIED")
    language = dspy.OutputField(desc="BCP-47 language tag detected from the patient's message (e.g. 'pt-BR', 'es', 'en'). Default: 'pt-BR'.")
    reasoning = dspy.OutputField(desc="Brief explanation of the routing decision (max 200 chars).")
    confidence = dspy.OutputField(desc="Confidence level from 0.0 to 1.0 for the primary (last/most important) intent.")
