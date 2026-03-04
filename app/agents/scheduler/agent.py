"""
SchedulerAgent — multi-turn appointment booking agent.
"""

import re
import dspy
from typing import List, Dict, Any, Optional
from datetime import datetime

from .signatures import SchedulerSignature, SlotExtractorSignature


VALID_STAGES = {"collecting_service", "presenting_slots", "booked"}

WEEKDAYS_PT = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]


class SchedulerAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process = dspy.ChainOfThought(SchedulerSignature)
        self.slot_extractor = dspy.Predict(SlotExtractorSignature)

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "Sem histórico anterior."
        lines = []
        for turn in history[-20:]:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            prefix = "Paciente" if role == "human" else role
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def _parse_stage(self, raw: Any, current_stage: str) -> str:
        if isinstance(raw, str):
            candidate = raw.strip().lower()
            if candidate in VALID_STAGES:
                return candidate
        return current_stage

    def _parse_slot(self, raw: Any) -> Optional[str]:
        if not raw or str(raw).strip().lower() in ("null", "none", ""):
            return None
        cleaned = str(raw).strip()
        for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
            try:
                return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                continue
        match = re.search(r"(\d{4}-\d{2}-\d{2})[T\s](\d{2}:\d{2})", cleaned)
        if match:
            return f"{match.group(1)} {match.group(2)}"
        return None

    def _parse_service(self, raw: Any) -> Optional[str]:
        if not raw or str(raw).strip().lower() in ("null", "none", ""):
            return None
        return str(raw).strip()

    def _humanize_slot(self, slot: str) -> str:
        """Convert 'YYYY-MM-DD HH:MM' to 'Qui, 05/03 às 11h'."""
        try:
            dt = datetime.strptime(slot, "%Y-%m-%d %H:%M")
            dow = WEEKDAYS_PT[dt.weekday()]
            return f"{dow}, {dt.day:02d}/{dt.month:02d} às {dt.hour:02d}h"
        except ValueError:
            return slot

    def _extract_slot_with_llm(
        self, patient_message: str, slots_str: str
    ) -> Optional[str]:
        """
        Focused LLM call (dspy.Predict, no CoT) to extract which slot the patient chose.
        Handles informal references in any language: "as 9", "9 tá bom", "o primeiro",
        "the first one", "a las 9", etc.
        Returns the ISO slot string or None.
        """
        try:
            result = self.slot_extractor(
                patient_message=patient_message,
                available_slots=slots_str,
            )
            return self._parse_slot(result.chosen_slot)
        except Exception:
            return None

    def forward(
        self,
        patient_message: str,
        history: List[Dict[str, str]],
        available_slots: List[str],
        clinic_name: str,
        patient_name: str,
        stage: str,
        services_list: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        history_str = self._format_history(history)
        slots_str = (
            ", ".join(
                f"{self._humanize_slot(s)} ({s})" for s in available_slots
            )
            if available_slots else "Sem horários disponíveis"
        )
        services_str = ", ".join(services_list[:50]) if services_list else ""

        # Pre-LLM: focused slot extractor runs first when patient may be selecting a slot.
        # Uses a lightweight dspy.Predict (no CoT) — cheaper and faster than main process.
        # Handles any language or informal expression ("as 9", "9 tá bom", "the first one").
        pre_chosen: Optional[str] = None
        if stage == "presenting_slots" and available_slots:
            pre_chosen = self._extract_slot_with_llm(patient_message, slots_str)

        try:
            result = self.process(
                patient_message=patient_message,
                history_str=history_str,
                available_slots=slots_str,
                services_list=services_str,
                clinic_name=clinic_name,
                patient_name=patient_name or "Paciente",
                current_stage=stage,
            )

            new_stage = self._parse_stage(result.stage, stage)
            chosen_slot = self._parse_slot(result.chosen_slot)
            service_requested = self._parse_service(result.service_requested)

            # If Python detected a slot but LLM didn't advance, override.
            if pre_chosen and new_stage != "booked":
                chosen_slot = pre_chosen
                new_stage = "booked"

            # Guard: booked requires a chosen slot.
            if new_stage == "booked" and not chosen_slot:
                chosen_slot = pre_chosen  # last resort
            if new_stage == "booked" and not chosen_slot:
                new_stage = "presenting_slots"

            # Guard: can't present slots without available_slots
            if new_stage == "presenting_slots" and not available_slots:
                new_stage = "collecting_service"

            if new_stage == "booked":
                return {
                    "messages": [{"type": "text", "content": str(result.response_message).strip()}],
                    "conversation_stage": "booked",
                    "reasoning": str(result.reasoning).strip(),
                    "data": {
                        "type": "appointment",
                        "service": service_requested,
                        "chosen_slot": chosen_slot,
                    },
                }

            return {
                "messages": [{"type": "text", "content": str(result.response_message).strip()}],
                "conversation_stage": new_stage,
                "reasoning": str(result.reasoning).strip(),
                "data": None,
            }

        except Exception as e:
            print(f"SchedulerAgent error: {e}")
            return {
                "messages": [{"type": "text", "content": "Vou verificar os horários disponíveis para você. Um momento!"}],
                "conversation_stage": stage,
                "reasoning": f"Erro: {str(e)}",
                "data": None,
            }
