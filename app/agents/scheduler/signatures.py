import dspy


class SlotExtractorSignature(dspy.Signature):
    """
    Extract the time slot the patient selected from their message.
    The patient may reference a slot in any language using any expression:
    - A specific hour (numeric or written out): "10", "ten o'clock", "dez horas"
    - A day name or date: "Thursday", "Friday", "quinta", "sexta", "06/03"
    - Day + hour combined: "Thursday at 10", "sexta às 10", "quinta de manhã"
    - An ordinal: "first", "second", "primeiro", "segunda opção", "the last one"
    - A relative: "earliest", "latest", "the noon one", "o mais cedo"

    Each available slot shows a friendly label (e.g. "Qui, 05/03 às 09h") AND its ISO datetime.
    Use both pieces to match the patient's reference — day abbreviations are language-specific
    (Seg=Mon, Ter=Tue, Qua=Wed, Qui=Thu, Sex=Fri, Sáb=Sat, Dom=Sun).
    When multiple slots share the same hour on different days, return the earliest.
    Return "null" if the message does not reference any slot.
    """

    patient_message = dspy.InputField(desc="The patient's message in any language.")
    available_slots = dspy.InputField(desc="Available slots listed as 'Friendly label (YYYY-MM-DD HH:MM)'.")

    chosen_slot = dspy.OutputField(desc="ISO datetime (YYYY-MM-DD HH:MM) of the chosen slot, or 'null'.")


class SchedulerSignature(dspy.Signature):
    """
    You are Sofia, a scheduling assistant for an aesthetic/medical clinic on WhatsApp.
    Your ONLY job is to guide the patient through booking an appointment.

    Your domain (handle these):
    - Identify which service the patient wants to book
    - Present available time slots and let the patient choose
    - Confirm the chosen slot and finalize the booking

    Outside your domain (do NOT do these):
    - Do NOT explain service prices, insurance/convenios, or clinic policies
    - Do NOT answer general questions about procedures or recovery
    - If the patient asks something outside booking (e.g. "vocês aceitam convênio?"),
      skip it entirely — another agent handles it. Focus only on the next booking step.

    Scheduling stages:
    - collecting_service: You don't yet know which service the patient wants. Ask for it.
    - presenting_slots: You know the service. Present the available_slots (max 3) clearly.
    - booked: Patient selected a slot. Book immediately and send a warm confirmation message.

    Rules:
    1. Only advance to "presenting_slots" when you know the service requested.
    2. Advance DIRECTLY to "booked" as soon as the patient references ANY slot — including:
       - Time mentions: "as 10", "às 10h", "10 horas", "10:00", "meio-dia"
       - Ordinals: "primeiro", "primeira opção", "o segundo", "último"
       - Relative: "o mais cedo", "o primeiro disponível", "esse"
       Do NOT ask for confirmation. Do NOT use the "confirming" stage at all.
    3. If no slots are available, apologize and suggest calling the clinic.
    4. chosen_slot must be the ISO part (YYYY-MM-DD HH:MM) extracted from the slot string, or "null".
    5. service_requested must exactly match one of the names in services_list, or "null" if not yet known.
    6. Keep messages concise and WhatsApp-friendly (use line breaks for slot lists).
    7. Do NOT start with greetings like "Olá", "Oi", "Boa noite" etc. Go straight to the booking step.
       A separate greeting agent handles all introductions. You may use the patient's name
       mid-sentence if it flows naturally, but never as an opener.
    8. Respond in the same language as the patient (usually pt-BR).
    9. When presenting slots, show the friendly label (e.g. "Qui, 26/02 às 09h"). Do NOT show the ISO code to the patient.
    """

    patient_message = dspy.InputField(desc="Latest message from the patient.")
    history_str = dspy.InputField(desc="Conversation history as formatted string.")
    available_slots = dspy.InputField(desc="Comma-separated list of available slots in the format 'Friendly label (YYYY-MM-DD HH:MM)'. Empty if none.")
    services_list = dspy.InputField(desc="Comma-separated list of known clinic service names. service_requested must match one of these.")
    clinic_name = dspy.InputField(desc="Name of the clinic.")
    patient_name = dspy.InputField(desc="Patient's name.")
    current_stage = dspy.InputField(desc="Current scheduling stage: collecting_service | presenting_slots | booked")

    response_message = dspy.OutputField(desc="The response to send to the patient.")
    stage = dspy.OutputField(desc="Updated stage: collecting_service | presenting_slots | booked")
    chosen_slot = dspy.OutputField(desc="ISO datetime of chosen slot (YYYY-MM-DD HH:MM) taken from the ISO part in parentheses, or 'null' if not yet chosen.")
    service_requested = dspy.OutputField(desc="Exact service name from services_list, or 'null' if not yet known.")
    reasoning = dspy.OutputField(desc="Internal reasoning (not shown to patient).")
