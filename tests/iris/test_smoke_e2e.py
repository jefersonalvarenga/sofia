"""
Smoke E2E — Iris full conversation flow against real Anthropic.

Drives the Iris LangGraph pipeline through 6 conversation turns covering
greeting → faq → knowledge (multi-intent) → scheduling → confirmation:

  1. "oi"                              → GreetingAgent (deterministic)
  2. "qual o endereço?"                → KnowledgeSpecialist (FAQ, real Anthropic)
  3. "quanto custa botox e gravida?"   → multi-intent: FAQ + HUMAN_ESCALATION
  4. "quero agendar"                   → Scheduler collects info (real DSPy/Anthropic)
  5. "Maria, 11999998888"             → Scheduler proposes 3 slots
  6. "o de amanhã 14h"                → Scheduler confirms + persists appointment

Infrastructure (Supabase, Evolution) is mocked — CI-safe.
LLM calls are REAL — requires ANTHROPIC_API_KEY in env.

Usage:
  ANTHROPIC_API_KEY=sk-... python -m pytest tests/iris/test_smoke_e2e.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import dspy
import httpx
import pytest

from app.iris import pipeline as iris_pipeline
from app.iris.schemas import ParsedMessage

CLINIC_ID = "57952a29-e228-4cac-b5fa-3d20ba478f5d"
PATIENT_JID = "5511999990000@s.whatsapp.net"
INSTANCE_NAME = "iris-prod"
AVAILABLE_SLOTS = [
    "2026-05-15 09:00",
    "2026-05-15 10:00",
    "2026-05-15 14:00",
]
SERVICES_CTX = json.dumps({
    "services": [
        {"name": "Botox", "description": "Aplicação de toxina botulínica", "price": 1200.0},
        {"name": "Limpeza de Pele", "description": "Limpeza profunda", "price": 250.0},
        {"name": "Preenchimento Labial", "description": "Preenchimento com ácido hialurônico", "price": 800.0},
    ],
    "offers": [],
})


class _SessionManager:
    """In-memory session state across pipeline.invoke() calls."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._activations: List[Dict[str, Any]] = []
        self._outbound: List[Dict[str, Any]] = []
        self._appointments: List[Dict[str, Any]] = []

    def load(
        self,
        remote_jid: str,
        clinic_id: str,
        push_name: Optional[str] = None,
        instance_id: str = "",
        attribution_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        session_id = f"{remote_jid}:{clinic_id}"
        existing = self._sessions.get(session_id)
        if existing:
            return {
                **existing,
                "conversation_type": "returning",
            }
        return {
            "session_id": session_id,
            "customer_id": "smoke-customer-001",
            "history": [],
            "conversation_stage": "new",
            "conversation_type": "first_contact",
            "patient_name": push_name or "Maria",
            "clinic_name": "Clínica Vitória",
            "assistant_name": "Iris",
            "clinic_style": None,
            "paused": False,
            "attribution_id": attribution_id,
            "available_slots": AVAILABLE_SLOTS,
            "services_context": SERVICES_CTX,
        }

    def save(self, state: Dict[str, Any]) -> None:
        session_id = state.get("session_id", "")
        if not session_id:
            return

        runs = state.get("agent_runs", [])
        history = list(state.get("history", []))
        history.append({"role": "human", "content": state.get("message", "")})
        for run in runs:
            for msg in run.get("messages", []):
                if msg.get("type") == "text":
                    history.append({"role": run["agent"], "content": msg["content"]})
        history = history[-20:]

        stage = state.get("conversation_stage", "active")
        if runs:
            last_stage = runs[-1].get("conversation_stage")
            if last_stage:
                stage = last_stage

        paused = False
        for run in runs:
            data = run.get("data")
            if data and data.get("type") == "escalation":
                paused = True
                break

        self._sessions[session_id] = {
            "session_id": session_id,
            "customer_id": state.get("customer_id", "smoke-customer-001"),
            "history": history,
            "conversation_stage": stage,
            "patient_name": state.get("patient_name", "Maria"),
            "clinic_name": state.get("clinic_name", "Clínica Vitória"),
            "assistant_name": state.get("assistant_name", "Iris"),
            "clinic_style": state.get("clinic_style"),
            "paused": paused,
            "available_slots": AVAILABLE_SLOTS,
            "services_context": SERVICES_CTX,
        }

        for run in runs:
            self._activations.append({
                "agent_name": run.get("agent"),
                "cost_usd": run.get("cost_usd", "0"),
                "duration_ms": run.get("duration_ms", 0),
                "prompt_tokens": run.get("prompt_tokens", 0),
                "completion_tokens": run.get("completion_tokens", 0),
                "status": run.get("status", "success"),
                "conversation_stage": run.get("conversation_stage"),
            })

            data = run.get("data")
            if data and data.get("type") == "appointment":
                self._appointments.append(data)

        for run in runs:
            for msg in run.get("messages", []):
                if msg.get("type") == "text":
                    self._outbound.append({
                        "wamid": state.get("wamid", ""),
                        "content": msg["content"],
                        "direction": "outbound",
                    })

    def unpause(self, remote_jid: str, clinic_id: str) -> None:
        session_id = f"{remote_jid}:{clinic_id}"
        if session_id in self._sessions:
            self._sessions[session_id]["paused"] = False


def _ok_evolution_response(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "key": {"id": "BAE5OUTBOUND001", "remoteJid": PATIENT_JID, "fromMe": True},
            "status": "PENDING",
        },
    )


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOLUTION_API_KEY", "test-key")
    monkeypatch.setenv("EVOLUTION_API_URL", "https://evo.test")
    monkeypatch.setenv("SOFIA_VERSION", "iris-smoke")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("DSPY_PROVIDER", "anthropic")
    monkeypatch.setenv("DSPY_MODEL", "claude-haiku-4-5-20251001")
    from app.core import config as core_config
    monkeypatch.setattr(core_config, "_settings", None, raising=False)


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY required for real LLM smoke test",
)
class TestSmokeE2E:
    """Full conversation smoke — real Anthropic, mocked infra."""

    def _init_dspy(self) -> None:
        """Initialize DSPy LM so the SchedulerAgent can call it."""
        from app.core.config import get_settings
        settings = get_settings()
        try:
            lm = dspy.LM(
                model=f"{settings.dspy_provider}/{settings.dspy_model}",
                api_key=settings.anthropic_api_key or settings.get_llm_api_key(),
                temperature=settings.dspy_temperature,
                max_tokens=settings.dspy_max_tokens,
            )
            dspy.settings.configure(lm=lm)
        except Exception as e:
            import warnings
            warnings.warn(f"DSPy init failed (Scheduler will fall back): {e}")

    def _send_turn(
        self,
        session_mgr: _SessionManager,
        msg: str,
        wamid: str,
        mid: str,
        tid: str,
    ) -> Dict[str, Any]:
        parsed = ParsedMessage(
            instance_name=INSTANCE_NAME,
            remote_jid=PATIENT_JID,
            phone="5511999990000",
            wamid=wamid,
            push_name="Maria",
            message_content=msg,
            message_type="text",
        )
        return asyncio.run(
            iris_pipeline.invoke(
                clinic_id=CLINIC_ID,
                message_id=mid,
                parsed=parsed,
                trace_id=tid,
            )
        )

    def test_full_conversation_flow(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Init DSPy once so Scheduler can call it
        self._init_dspy()

        session_mgr = _SessionManager()

        monkeypatch.setattr("app.iris.pipeline.load_session", session_mgr.load)
        monkeypatch.setattr("app.iris.pipeline.save_session", session_mgr.save)

        for module in [
            "app.core.supabase_client",
            "app.session.manager",
            "app.agents.knowledge.agent",
        ]:
            monkeypatch.setattr(f"{module}.get_supabase", lambda: MagicMock())

        transport = httpx.MockTransport(_ok_evolution_response)
        original_send = iris_pipeline.send_text_message

        async def _patched_send(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            kwargs.setdefault("base_url", "https://evo.test")
            async with httpx.AsyncClient(transport=transport) as http:
                return await original_send(*args, client=http, **kwargs)

        monkeypatch.setattr(iris_pipeline, "send_text_message", _patched_send)

        # ================================================================
        # Turn 1 — "oi" → GreetingAgent
        # ================================================================
        r1 = self._send_turn(session_mgr, "oi", "smoke-wamid-001", "smoke-msg-001", "smoke-001")
        assert r1["status"] == "ok"
        assert r1["primary_intent"] == "GREETING"
        agent_names_1 = [run.get("agent") for run in r1["agent_runs"]]
        assert "GreetingAgent" in agent_names_1, f"Missing GreetingAgent in {agent_names_1}"
        assert "IrisRouterAgent" in agent_names_1, f"Missing Router in {agent_names_1}"

        # ================================================================
        # Turn 2 — "qual o endereço?" → KnowledgeSpecialist (FAQ)
        # ================================================================
        r2 = self._send_turn(session_mgr, "qual o endereço?", "smoke-wamid-002", "smoke-msg-002", "smoke-002")
        assert r2["status"] == "ok"
        assert r2["primary_intent"] == "FAQ"
        agent_names_2 = [run.get("agent") for run in r2["agent_runs"]]
        assert "KnowledgeSpecialist" in agent_names_2, f"Missing KnowledgeSpecialist in {agent_names_2}"

        # ================================================================
        # Turn 3 — "quanto custa botox e posso fazer grávida?"
        #           multi-intent: FAQ (price) + HUMAN_ESCALATION (medical)
        # ================================================================
        r3 = self._send_turn(
            session_mgr,
            "quanto custa o botox? posso fazer estando grávida?",
            "smoke-wamid-003",
            "smoke-msg-003",
            "smoke-003",
        )
        assert r3["status"] == "ok"
        intents_3 = [i["macro_state"] for i in r3["intents"]]
        assert "FAQ" in intents_3, f"FAQ missing from {intents_3}"
        assert "HUMAN_ESCALATION" in intents_3, f"HUMAN_ESCALATION missing from {intents_3}"
        specialist_runs_3 = [
            run for run in r3["agent_runs"]
            if run.get("agent") not in ("IrisRouterAgent",)
        ]
        assert len(specialist_runs_3) == 2, (
            f"Expected 2 specialist runs, got {len(specialist_runs_3)}"
        )

        session_mgr.unpause(PATIENT_JID, CLINIC_ID)

        # ================================================================
        # Turn 4 — "quero agendar" → Scheduler collects info
        # ================================================================
        r4 = self._send_turn(
            session_mgr,
            "quero agendar",
            "smoke-wamid-004",
            "smoke-msg-004",
            "smoke-004",
        )
        assert r4["status"] == "ok"
        assert r4["primary_intent"] == "SCHEDULE"
        agent_names_4 = [run.get("agent") for run in r4["agent_runs"]]
        assert "Scheduler" in agent_names_4, f"Missing Scheduler in {agent_names_4}"

        # ================================================================
        # Turn 5 — "Maria Silva, 11999998888" → Scheduler proposes 3 slots
        # ================================================================
        r5 = self._send_turn(
            session_mgr,
            "meu nome é Maria Silva, telefone 11999998888",
            "smoke-wamid-005",
            "smoke-msg-005",
            "smoke-005",
        )
        assert r5["status"] == "ok"
        assert r5["primary_intent"] == "SCHEDULE"

        # ================================================================
        # Turn 6 — "o de amanhã 14h" → Scheduler confirms + appointment
        # ================================================================
        r6 = self._send_turn(
            session_mgr,
            "o de amanhã 14h",
            "smoke-wamid-006",
            "smoke-msg-006",
            "smoke-006",
        )
        assert r6["status"] == "ok"

        # ================================================================
        # Verifications
        # ================================================================

        all_agent_runs = [
            run for r in [r1, r2, r3, r4, r5, r6]
            for run in r["agent_runs"]
        ]
        llm_runs = [run for run in all_agent_runs if run.get("total_tokens", 0) > 0]
        if llm_runs:
            avg_latency = sum(run.get("duration_ms", 0) for run in llm_runs) / len(llm_runs)
            assert avg_latency < 3000, (
                f"Avg LLM latency {avg_latency:.0f}ms exceeds 3000ms"
            )

        total_cost = Decimal("0")
        for run in all_agent_runs:
            cost = run.get("cost_usd", "0")
            try:
                total_cost += Decimal(str(cost))
            except Exception:
                pass
        assert total_cost < Decimal("0.05"), (
            f"Total cost ${total_cost} exceeds $0.05"
        )

        assert len(session_mgr._activations) > 0, "No agent activations captured"
        agents_seen = {a["agent_name"] for a in session_mgr._activations}
        print(f"\n[smoke] Agents activated: {agents_seen}")
        print(f"[smoke] Total activations: {len(session_mgr._activations)}")
        print(f"[smoke] Total cost: ${total_cost}")
        print(f"[smoke] Outbound messages: {len(session_mgr._outbound)}")
        print(f"[smoke] Appointments created: {len(session_mgr._appointments)}")
