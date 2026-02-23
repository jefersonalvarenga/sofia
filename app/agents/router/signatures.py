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
    Your job is to classify the patient's latest message into exactly ONE intent.

    Intent definitions:
    - GREETING: Patient initiates conversation, says hello, or sends a first message.
    - FAQ: Patient asks about services, prices, clinic address, opening hours, insurance/convenios,
           procedures, recovery, or any general information about the clinic.
    - SCHEDULE: Patient wants to book, schedule, or confirm an appointment. Also use when patient
                is in the middle of a scheduling conversation (selecting service, choosing a slot).
    - HUMAN_ESCALATION: Patient explicitly asks to speak with a human, attendant, or receptionist.
    - REENGAGE: Patient resumes a conversation that was paused or idle.
    - UNCLASSIFIED: None of the above applies clearly.

    Rules:
    1. Return a single intent — the most specific one that matches the message.
    2. When in doubt between FAQ and SCHEDULE, prefer SCHEDULE if the message implies booking intent.
    3. HUMAN_ESCALATION takes priority over all other intents if explicitly requested.
    4. Use `conversation_stage` for context — a patient mid-scheduling likely means SCHEDULE.
    5. Respond in Portuguese. Keep reasoning concise (max 200 chars).
    """

    latest_message = dspy.InputField(desc="Latest message from the patient.")
    history_str = dspy.InputField(desc="Conversation history as formatted string. 'Sem histórico.' if empty.")
    conversation_stage = dspy.InputField(desc="Current conversation stage (e.g., 'new', 'presenting_slots', 'booked').")
    language = dspy.InputField(desc="Patient's language, e.g. 'pt-BR'.")

    intent = dspy.OutputField(desc="Single intent: GREETING | FAQ | SCHEDULE | HUMAN_ESCALATION | REENGAGE | UNCLASSIFIED")
    reasoning = dspy.OutputField(desc="Brief explanation of the routing decision (max 200 chars).")
    confidence = dspy.OutputField(desc="Confidence level from 0.0 to 1.0.")
