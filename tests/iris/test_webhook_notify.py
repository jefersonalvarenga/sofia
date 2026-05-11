"""Tests for services.iris.webhook — notification stub."""

import io
import json
import logging
import os

import pytest

from services.iris.webhook import notify_receptionist


@pytest.fixture(autouse=True)
def _isolate_logger():
    """Capture log output and clean env vars per test."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("services.iris.webhook")
    old_handlers = logger.handlers[:]
    old_level = logger.level
    old_propagate = logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield stream
    logger.handlers = old_handlers
    logger.setLevel(old_level)
    logger.propagate = old_propagate


@pytest.fixture
def clean_webhook_env():
    """Remove webhook env vars for the test."""
    saved = {}
    for key in list(os.environ):
        if key.startswith("WEBHOOK_URL_"):
            saved[key] = os.environ.pop(key)
    yield
    os.environ.update(saved)


def test_notify_no_url(clean_webhook_env, _isolate_logger):
    """Logs JSON event without webhook_url when no env vars are set."""
    notify_receptionist("t1", "c1", "Maria", "explicit_request")
    output = _isolate_logger.getvalue()
    lines = [l for l in output.strip().split("\n") if l]
    assert len(lines) == 2  # debug + info
    assert "no webhook url configured" in lines[0]
    event = json.loads(lines[1])
    assert event["event"] == "escalation.notify"
    assert event["tenant_id"] == "t1"
    assert event["conversation_id"] == "c1"
    assert event["patient_name"] == "Maria"
    assert event["trigger"] == "explicit_request"
    assert "timestamp" in event
    assert "webhook_url" not in event


def test_notify_with_default_url(clean_webhook_env, _isolate_logger):
    """Includes webhook_url from WEBHOOK_URL_DEFAULT env var."""
    os.environ["WEBHOOK_URL_DEFAULT"] = "https://hooks.example.com/notify"
    notify_receptionist("t1", "c1", "João", "clinical_urgency")
    output = _isolate_logger.getvalue()
    lines = [l for l in output.strip().split("\n") if l]
    assert len(lines) == 2
    assert "webhook url resolved" in lines[0]
    event = json.loads(lines[1])
    assert event["webhook_url"] == "https://hooks.example.com/notify"
    assert event["trigger"] == "clinical_urgency"


def test_notify_tenant_specific_url(clean_webhook_env, _isolate_logger):
    """Tenant-specific URL takes priority over default."""
    os.environ["WEBHOOK_URL_DEFAULT"] = "https://default.example.com"
    os.environ["WEBHOOK_URL_clinic_abc"] = "https://clinic-abc.example.com"
    notify_receptionist("clinic_abc", "c1", "Ana", "sensitive_flag")
    output = _isolate_logger.getvalue()
    lines = [l for l in output.strip().split("\n") if l]
    event = json.loads(lines[1])
    assert event["webhook_url"] == "https://clinic-abc.example.com"


def test_notify_patient_name_none(clean_webhook_env, _isolate_logger):
    """Handles None patient_name."""
    notify_receptionist("t1", "c1", None, "complaint_tone")
    output = _isolate_logger.getvalue()
    lines = [l for l in output.strip().split("\n") if l]
    event = json.loads(lines[1])
    assert event["patient_name"] is None
    assert event["trigger"] == "complaint_tone"


def test_notify_timestamp_is_iso8601(clean_webhook_env, _isolate_logger):
    """Timestamp is a valid ISO8601 string."""
    notify_receptionist("t1", "c1", "Maria", "explicit_request")
    output = _isolate_logger.getvalue()
    lines = [l for l in output.strip().split("\n") if l]
    event = json.loads(lines[1])
    ts = event["timestamp"]
    assert ts.endswith("+00:00") or ts.endswith("Z")
