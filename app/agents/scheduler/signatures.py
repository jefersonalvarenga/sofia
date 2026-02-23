import dspy


class SchedulerSignature(dspy.Signature):
    """
    You are Sofia, a scheduling assistant for an aesthetic/medical clinic on WhatsApp.
    Your goal is to help the patient book an appointment in a natural, friendly conversation.

    Scheduling stages:
    - collecting_service: You don't yet know which service the patient wants. Ask for it.
    - presenting_slots: You know the service. Present the available_slots (max 3) clearly.
    - confirming: Patient chose a slot. Confirm the details before finalizing.
    - booked: Patient confirmed. The appointment is booked. Send a final confirmation message.

    Rules:
    1. Only advance to "presenting_slots" when you know the service requested.
    2. Only advance to "confirming" when the patient has chosen a specific slot.
    3. Only set stage to "booked" when the patient has explicitly confirmed the appointment.
    4. If no slots are available, apologize and suggest calling the clinic.
    5. chosen_slot must be an ISO datetime string (YYYY-MM-DD HH:MM) or the literal string "null".
    6. service_requested must be the service name or "null" if not yet known.
    7. Keep messages concise and WhatsApp-friendly (use line breaks for slot lists).
    8. Use the patient's name naturally in the first message.
    9. Respond in the same language as the patient (usually pt-BR).
    """

    patient_message = dspy.InputField(desc="Latest message from the patient.")
    history_str = dspy.InputField(desc="Conversation history as formatted string.")
    available_slots = dspy.InputField(desc="Comma-separated list of available time slots (YYYY-MM-DD HH:MM format). Empty if none.")
    clinic_name = dspy.InputField(desc="Name of the clinic.")
    patient_name = dspy.InputField(desc="Patient's name.")
    stage = dspy.InputField(desc="Current scheduling stage: collecting_service | presenting_slots | confirming | booked")

    response_message = dspy.OutputField(desc="The response to send to the patient.")
    stage = dspy.OutputField(desc="Updated stage: collecting_service | presenting_slots | confirming | booked")
    chosen_slot = dspy.OutputField(desc="ISO datetime of chosen slot (YYYY-MM-DD HH:MM) or 'null' if not yet chosen.")
    service_requested = dspy.OutputField(desc="Name of the requested service or 'null' if not yet known.")
    reasoning = dspy.OutputField(desc="Internal reasoning (not shown to patient).")
