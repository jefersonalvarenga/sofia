"""
Sofia Telemetry — structured logging and token extraction for observability.
"""

import time
import dspy
import structlog
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

# ============================================================================
# Structured logger
# ============================================================================

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
)

log = structlog.get_logger()


# ============================================================================
# Token extraction from DSPy history
# ============================================================================

def extract_tokens() -> Dict[str, int]:
    """
    Extract token usage from the last DSPy LM call.
    Returns {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for deterministic agents that made no LLM call.
    """
    empty = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        lm = dspy.settings.lm
        if not lm or not hasattr(lm, "history") or not lm.history:
            return empty
        last = lm.history[-1]
        # LiteLLM response stores usage in the response object
        response = last.get("response")
        if not response:
            return empty
        usage = getattr(response, "usage", None)
        if not usage:
            return empty
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
    except Exception:
        return empty


# ============================================================================
# Agent run wrapper
# ============================================================================

def build_agent_run(
    agent_name: str,
    reason: str,
    trace_id: str,
    clinic_id: str,
    session_id: str,
    language: str,
    sofia_version: str,
    call: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Execute `call()` and wrap its result into a complete AgentRun dict
    with timing, token usage, and observability fields.

    `call` must be a zero-argument callable that returns a dict with at least:
      - messages: List[dict]
      - conversation_stage: str
      - reasoning: str
      - data: Optional[dict]
    """
    history_len_before = 0
    try:
        lm = dspy.settings.lm
        if lm and hasattr(lm, "history"):
            history_len_before = len(lm.history)
    except Exception:
        pass

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    status = "success"
    result: Dict[str, Any] = {}
    try:
        result = call()
    except Exception as e:
        status = "error"
        result = {
            "messages": [{"type": "text", "content": "Desculpe, ocorreu um erro. Tente novamente."}],
            "conversation_stage": "error",
            "reasoning": f"Unhandled exception: {str(e)}",
            "data": None,
        }
        log.error("agent.error", agent=agent_name, trace_id=trace_id,
                  clinic_id=clinic_id, error=str(e))

    duration_ms = (time.perf_counter() - t0) * 1000

    # Only read new token entries added during this call
    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        lm = dspy.settings.lm
        if lm and hasattr(lm, "history") and len(lm.history) > history_len_before:
            tokens = extract_tokens()
    except Exception:
        pass

    agent_run = {
        "agent": agent_name,
        "reason": reason,
        "status": status,
        "messages": result.get("messages", []),
        "data": result.get("data"),
        "reasoning": result.get("reasoning"),
        "conversation_stage": result.get("conversation_stage"),
        "started_at": started_at,
        "duration_ms": round(duration_ms, 2),
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
        "trace_id": trace_id,
        "clinic_id": clinic_id,
        "session_id": session_id,
        "language": language,
        "sofia_version": sofia_version,
    }

    log.info(
        "agent.run",
        agent=agent_name,
        status=status,
        duration_ms=round(duration_ms, 2),
        total_tokens=tokens["total_tokens"],
        trace_id=trace_id,
        clinic_id=clinic_id,
        reason=reason,
        reasoning=result.get("reasoning"),
        stage=result.get("conversation_stage"),
    )

    return agent_run
