"""
SchedulerAgent — multi-turn appointment booking agent.
"""

import re
import dspy
from typing import List, Dict, Any, Optional
from datetime import datetime

from .signatures import SchedulerSignature


VALID_STAGES = {"collecting_service", "presenting_slots", "confirming", "booked"}


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

    def forward(
        self,
        patient_message: str,
        history: List[Dict[str, str]],
        available_slots: List[str],
        clinic_name: str,
        patient_name: str,
        stage: str,
    ) -> Dict[str, Any]:
        history_str = self._format_history(history)
        slots_str = ", ".join(available_slots) if available_slots else "Sem horários disponíveis"

        try:
            result = self.process(
                patient_message=patient_message,
                history_str=history_str,
                available_slots=slots_str,
                clinic_name=clinic_name,
                patient_name=patient_name or "Paciente",
                stage=stage,
            )

            new_stage = self._parse_stage(result.stage, stage)
            chosen_slot = self._parse_slot(result.chosen_slot)
            service_requested = self._parse_service(result.service_requested)

            # Guard: can't be booked without a chosen slot
            if new_stage == "booked" and not chosen_slot:
                new_stage = "confirming"

            # Guard: can't present slots without available_slots
            if new_stage == "presenting_slots" and not available_slots:
                new_stage = "collecting_service"

            return {
                "response_message": str(result.response_message).strip(),
                "conversation_stage": new_stage,
                "chosen_slot": chosen_slot,
                "service_requested": service_requested,
                "reasoning": str(result.reasoning).strip(),
                "agent_name": "Scheduler",
                "requires_human": False,
            }

        except Exception as e:
            print(f"SchedulerAgent error: {e}")
            return {
                "response_message": "Vou verificar os horários disponíveis para você. Um momento!",
                "conversation_stage": stage,
                "chosen_slot": None,
                "service_requested": None,
                "reasoning": f"Erro: {str(e)}",
                "agent_name": "Scheduler",
                "requires_human": False,
            }
