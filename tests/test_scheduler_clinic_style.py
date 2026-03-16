"""
Tests for SchedulerAgent clinic_style injection and _call_agent wiring (Task 2).
Verifies SchedulerSignature and SchedulerAgent.forward() accept clinic_tone
and personality_traits, and that sofia_graph.py wires them from state["clinic_style"].
"""

import inspect
import pathlib
import pytest


class TestSchedulerSignatureFields:
    def test_clinic_tone_in_signature_fields(self):
        from app.agents.scheduler.signatures import SchedulerSignature
        fields = list(SchedulerSignature.model_fields.keys())
        assert "clinic_tone" in fields, f"clinic_tone missing from SchedulerSignature fields: {fields}"

    def test_personality_traits_in_signature_fields(self):
        from app.agents.scheduler.signatures import SchedulerSignature
        fields = list(SchedulerSignature.model_fields.keys())
        assert "personality_traits" in fields, f"personality_traits missing from SchedulerSignature fields: {fields}"


class TestSchedulerAgentForwardSignature:
    def test_tone_param_in_forward(self):
        from app.agents.scheduler.agent import SchedulerAgent
        sig = inspect.signature(SchedulerAgent.forward)
        params = list(sig.parameters.keys())
        assert "tone" in params, f"tone missing from SchedulerAgent.forward(): {params}"

    def test_personality_traits_param_in_forward(self):
        from app.agents.scheduler.agent import SchedulerAgent
        sig = inspect.signature(SchedulerAgent.forward)
        params = list(sig.parameters.keys())
        assert "personality_traits" in params, f"personality_traits missing from SchedulerAgent.forward(): {params}"

    def test_tone_defaults_to_empty_string(self):
        from app.agents.scheduler.agent import SchedulerAgent
        sig = inspect.signature(SchedulerAgent.forward)
        assert sig.parameters["tone"].default == "", "tone should default to empty string"

    def test_personality_traits_defaults_to_none(self):
        from app.agents.scheduler.agent import SchedulerAgent
        sig = inspect.signature(SchedulerAgent.forward)
        assert sig.parameters["personality_traits"].default is None, "personality_traits should default to None"


class TestCallAgentWiring:
    def test_call_agent_passes_attendance_flow_to_faq(self):
        graph_src = pathlib.Path("app/graph/sofia_graph.py").read_text()
        assert "attendance_flow" in graph_src, "_call_agent does not pass attendance_flow to FAQ"

    def test_call_agent_passes_tone(self):
        graph_src = pathlib.Path("app/graph/sofia_graph.py").read_text()
        assert "tone=tone" in graph_src, "_call_agent does not pass tone= to agents"

    def test_call_agent_passes_personality_traits(self):
        graph_src = pathlib.Path("app/graph/sofia_graph.py").read_text()
        assert "personality_traits=personality_traits" in graph_src, \
            "_call_agent does not pass personality_traits= to agents"

    def test_call_agent_extracts_clinic_style_from_state(self):
        graph_src = pathlib.Path("app/graph/sofia_graph.py").read_text()
        assert 'clinic_style.get' in graph_src, \
            "_call_agent does not extract fields from clinic_style via .get()"
