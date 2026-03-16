"""
Tests for FAQResponder clinic_style injection (Task 1).
Verifies that FAQResponderSignature and FAQResponderAgent.forward() accept
clinic_tone, personality_traits, and attendance_flow fields.
"""

import inspect
import pytest


class TestFAQResponderSignatureFields:
    def test_clinic_tone_in_signature_fields(self):
        from app.agents.faq_responder.signatures import FAQResponderSignature
        fields = list(FAQResponderSignature.signature.fields.keys())
        assert "clinic_tone" in fields, f"clinic_tone missing from FAQResponderSignature fields: {fields}"

    def test_personality_traits_in_signature_fields(self):
        from app.agents.faq_responder.signatures import FAQResponderSignature
        fields = list(FAQResponderSignature.signature.fields.keys())
        assert "personality_traits" in fields, f"personality_traits missing from FAQResponderSignature fields: {fields}"

    def test_attendance_flow_in_signature_fields(self):
        from app.agents.faq_responder.signatures import FAQResponderSignature
        fields = list(FAQResponderSignature.signature.fields.keys())
        assert "attendance_flow" in fields, f"attendance_flow missing from FAQResponderSignature fields: {fields}"


class TestFAQResponderAgentForwardSignature:
    def test_tone_param_in_forward(self):
        from app.agents.faq_responder.agent import FAQResponderAgent
        sig = inspect.signature(FAQResponderAgent.forward)
        params = list(sig.parameters.keys())
        assert "tone" in params, f"tone missing from FAQResponderAgent.forward(): {params}"

    def test_personality_traits_param_in_forward(self):
        from app.agents.faq_responder.agent import FAQResponderAgent
        sig = inspect.signature(FAQResponderAgent.forward)
        params = list(sig.parameters.keys())
        assert "personality_traits" in params, f"personality_traits missing from FAQResponderAgent.forward(): {params}"

    def test_attendance_flow_param_in_forward(self):
        from app.agents.faq_responder.agent import FAQResponderAgent
        sig = inspect.signature(FAQResponderAgent.forward)
        params = list(sig.parameters.keys())
        assert "attendance_flow" in params, f"attendance_flow missing from FAQResponderAgent.forward(): {params}"

    def test_tone_defaults_to_empty_string(self):
        from app.agents.faq_responder.agent import FAQResponderAgent
        sig = inspect.signature(FAQResponderAgent.forward)
        assert sig.parameters["tone"].default == "", "tone should default to empty string"

    def test_personality_traits_defaults_to_none(self):
        from app.agents.faq_responder.agent import FAQResponderAgent
        sig = inspect.signature(FAQResponderAgent.forward)
        assert sig.parameters["personality_traits"].default is None, "personality_traits should default to None"

    def test_attendance_flow_defaults_to_none(self):
        from app.agents.faq_responder.agent import FAQResponderAgent
        sig = inspect.signature(FAQResponderAgent.forward)
        assert sig.parameters["attendance_flow"].default is None, "attendance_flow should default to None"
