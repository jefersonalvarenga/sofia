"""
SchedulerAgent — multi-turn appointment booking agent.
"""

import re
import dspy
from typing import List, Dict, Any, Optional
from datetime import datetime

from .signatures import SchedulerSignature


VALID_STAGES = {"collecting_service", "presenting_slots", "confirming", "booked"}

WEEKDAYS_PT = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]


class SchedulerAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process = dspy.ChainOfThought(SchedulerSignature)

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
        # Try ISO formats
        for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
            try:
                return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                continue
        # Extract via regex
        match = re.search(r"(\d{4}-\d{2}-\d{2})[T\s](\d{2}:\d{2})", cleaned)
        if match:
            return f"{match.group(1)} {match.group(2)}"
        return None

    def _parse_service(self, raw: Any) -> Optional[str]:
        if not raw or str(raw).strip().lower() in ("null", "none", ""):
            return None
        return str(raw).strip()

    def _humanize_slot(self, slot: str) -> str:
        """Convert 'YYYY-MM-DD HH:MM' to 'Qui, 26/02 às 09h (YYYY-MM-DD HH:MM)'."""
        try:
            dt = datetime.strptime(slot, "%Y-%m-%d %H:%M")
            dow = WEEKDAYS_PT[dt.weekday()]
            return f"{dow}, {dt.day:02d}/{dt.month:02d} às {dt.hour:02d}h ({slot})"
        except ValueError:
            return slot

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
            ", ".join(self._humanize_slot(s) for s in available_slots)
            if available_slots else "Sem horários disponíveis"
        )
        services_str = ", ".join(services_list[:50]) if services_list else ""

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

            # Guard: can't be booked without a chosen slot.
            if new_stage == "booked" and not chosen_slot:
                # Fallback A: ISO in parentheses in history "(2026-03-04 10:00)"
                for turn in reversed(history[-10:]):
                    m = re.search(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2})\)", turn.get("content", ""))
                    if m:
                        chosen_slot = m.group(1)
                        break
                # Fallback B: "às Xh" time mention cross-referenced with available_slots
                # (Sofia shows friendly format only — ISO is never shown to the patient)
                if not chosen_slot and available_slots:
                    for turn in reversed(history[-10:]):
                        hour_match = re.search(r"às\s+(\d{1,2})h", turn.get("content", ""))
                        if hour_match:
                            target_hour = int(hour_match.group(1))
                            for slot in available_slots:
                                try:
                                    if int(slot.split(" ")[1].split(":")[0]) == target_hour:
                                        chosen_slot = slot
                                        break
                                except Exception:
                                    continue
                        if chosen_slot:
                            break
            if new_stage == "booked" and not chosen_slot:
                new_stage = "confirming"

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
