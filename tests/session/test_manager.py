"""
Tests for ``app.session.manager`` schedule_session_data persistence.

These tests assert that ``schedule_session_data`` (cross-turn state populated by
SCHEDULE_* sub-agents) survives a round-trip through ``sf_sessions`` via
``save_session`` and ``load_session``.

Background — Migration 029 adds a JSONB column ``schedule_session_data`` to
``sf_sessions``. ``load_session`` must hydrate it onto the returned state and
``save_session`` must write it back. ``load_session`` must also degrade to
``[]`` on pre-migration deploys where the column is missing.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch


CLINIC_ID = "clinic-uuid-test"
REMOTE_JID = "5511999990000@s.whatsapp.net"
SESSION_ID = f"{REMOTE_JID}:{CLINIC_ID}"


# ---------------------------------------------------------------------------
# Supabase mock helpers
# ---------------------------------------------------------------------------


def _mk_session_row(
    *,
    schedule_session_data: Any = "__OMIT__",
    history: List[Dict[str, Any]] | None = None,
    conversation_stage: str = "new",
    paused: bool = False,
) -> Dict[str, Any]:
    """Build a fake ``sf_sessions`` row.

    Pass ``schedule_session_data="__OMIT__"`` to simulate a pre-migration row
    where the column does not exist (key absent from the dict).
    """
    row: Dict[str, Any] = {
        "session_id": SESSION_ID,
        "history": history or [],
        "conversation_stage": conversation_stage,
        "paused": paused,
        "updated_at": "2026-05-20T12:00:00+00:00",
    }
    if schedule_session_data != "__OMIT__":
        row["schedule_session_data"] = schedule_session_data
    return row


def _mk_load_supabase_mock(
    *,
    session_row: Dict[str, Any] | None,
) -> MagicMock:
    """Mock ``get_supabase()`` chain for ``load_session``.

    Captures the customer upsert, the session select/insert, the clinic profile
    select, the la_blueprints select, and the business_rules select. Returns
    ``session_row`` when ``sf_sessions`` is selected, or an empty list when
    ``session_row`` is None (forces ``load_session`` to insert a new row).
    """

    sb = MagicMock()

    def _table(name: str) -> MagicMock:
        builder = MagicMock()
        # Default chain fallthrough.
        builder.select.return_value = builder
        builder.insert.return_value = builder
        builder.upsert.return_value = builder
        builder.update.return_value = builder
        builder.eq.return_value = builder
        builder.neq.return_value = builder
        builder.in_.return_value = builder
        builder.is_.return_value = builder
        builder.order.return_value = builder
        builder.limit.return_value = builder
        builder.maybe_single.return_value = builder

        if name == "sf_customers":
            builder.execute.return_value = MagicMock(
                data=[{"id": "customer-uuid-123"}]
            )
        elif name == "sf_sessions":
            rows = [session_row] if session_row is not None else []
            builder.execute.return_value = MagicMock(data=rows)
        elif name == "sf_clinic_profiles":
            builder.execute.return_value = MagicMock(
                data={
                    "clinic_name": "Clínica Teste",
                    "assistant_name": "Iris",
                    "avg_ticket": 0,
                    "address": "",
                }
            )
        elif name == "sf_instance_clinic_map":
            builder.execute.return_value = MagicMock(
                data={"clinic_id": CLINIC_ID}
            )
        else:
            # la_blueprints, sf_clinic_business_rules → no data, default style.
            builder.execute.return_value = MagicMock(data=None)
        return builder

    sb.table.side_effect = _table
    return sb


def _mk_save_supabase_mock() -> tuple[MagicMock, MagicMock]:
    """Mock ``get_supabase()`` for ``save_session``.

    Returns ``(supabase_mock, sessions_builder)`` — ``sessions_builder`` is the
    stable mock returned every time ``sb.table("sf_sessions")`` is called, so
    tests can introspect ``sessions_builder.update.call_args``.
    """
    sb = MagicMock()
    update_chain = MagicMock()
    update_chain.eq.return_value = update_chain
    update_chain.execute.return_value = MagicMock(data=[{"session_id": SESSION_ID}])

    insert_chain = MagicMock()
    insert_chain.execute.return_value = MagicMock(data=[])

    # Memoize per-table builders so multiple sb.table("name") calls return
    # the *same* builder. supabase-py builders are method-chained and the
    # production code calls sb.table("sf_sessions") once per save; the test
    # then re-calls it to inspect — both must hit the same mock.
    builders: Dict[str, MagicMock] = {}

    def _table(name: str) -> MagicMock:
        if name in builders:
            return builders[name]
        builder = MagicMock()
        if name == "sf_sessions":
            builder.update.return_value = update_chain
            builder.insert.return_value = insert_chain
        elif name == "sf_agent_activations":
            builder.insert.return_value = insert_chain
        else:
            builder.update.return_value = update_chain
            builder.insert.return_value = insert_chain
        builders[name] = builder
        return builder

    sb.table.side_effect = _table
    sessions_builder = _table("sf_sessions")  # pre-register for test introspection
    return sb, sessions_builder


def _base_state_for_save(
    *,
    schedule_session_data: Any = None,
) -> Dict[str, Any]:
    """SofiaState-shaped dict for ``save_session`` tests."""
    state: Dict[str, Any] = {
        "instance_id": "test-instance",
        "clinic_id": CLINIC_ID,
        "remote_jid": REMOTE_JID,
        "session_id": SESSION_ID,
        "customer_id": "customer-uuid-123",
        "push_name": "Maria",
        "patient_name": "Maria",
        "message": "olá",
        "message_type": "text",
        "wamid": "test-wamid",
        "available_slots": [],
        "history": [],
        "conversation_stage": "new",
        "trace_id": "trace-1",
        "language": "pt-BR",
        "agent_runs": [],
        "detected_intents": [],
    }
    if schedule_session_data is not None:
        state["schedule_session_data"] = schedule_session_data
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveSessionPersistsScheduleSessionData:
    """save_session must include schedule_session_data in sf_sessions UPDATE."""

    def test_save_session_includes_schedule_session_data(self) -> None:
        from app.session import manager

        schedule_data = [
            {
                "name": "evaluation",
                "data": {
                    "service": "botox",
                    "intake_answers": [
                        {
                            "question_id": "q-1",
                            "question_text": "Toma algum medicamento?",
                            "answer": "não",
                        }
                    ],
                },
            }
        ]
        state = _base_state_for_save(schedule_session_data=schedule_data)
        sb, sessions_builder = _mk_save_supabase_mock()

        with patch.object(manager, "get_supabase", return_value=sb):
            manager.save_session(state)  # type: ignore[arg-type]

        # save_session must hit sf_sessions table and call .update(payload).
        table_calls = [c for c in sb.table.call_args_list if c.args and c.args[0] == "sf_sessions"]
        assert table_calls, "expected at least one sf_sessions table call"
        assert sessions_builder.update.called, "save_session must call sf_sessions.update(...)"

        payload = sessions_builder.update.call_args.args[0]
        assert "schedule_session_data" in payload, (
            "save_session must include schedule_session_data in the UPDATE payload "
            f"(payload keys: {list(payload.keys())})"
        )
        assert payload["schedule_session_data"] == schedule_data

    def test_save_session_empty_schedule_session_data(self) -> None:
        """No schedule_session_data on state → persist [] (never null)."""
        from app.session import manager

        state = _base_state_for_save(schedule_session_data=None)
        sb, _ = _mk_save_supabase_mock()

        with patch.object(manager, "get_supabase", return_value=sb):
            manager.save_session(state)  # type: ignore[arg-type]

        sf_builder = sb.table("sf_sessions")
        payload = sf_builder.update.call_args.args[0]
        assert payload.get("schedule_session_data") == [], (
            "missing schedule_session_data on state must serialize as [] "
            f"(got {payload.get('schedule_session_data')!r})"
        )


class TestLoadSessionHydratesScheduleSessionData:
    """load_session must expose schedule_session_data on the returned dict."""

    def test_load_session_returns_schedule_session_data(self) -> None:
        from app.session import manager

        schedule_data = [
            {
                "name": "evaluation",
                "data": {
                    "service": "botox",
                    "intake_answers": [
                        {
                            "question_id": "q-1",
                            "question_text": "Toma algum medicamento?",
                            "answer": "não",
                        }
                    ],
                },
            }
        ]
        row = _mk_session_row(schedule_session_data=schedule_data)
        sb = _mk_load_supabase_mock(session_row=row)

        with patch.object(manager, "get_supabase", return_value=sb):
            ctx = manager.load_session(
                remote_jid=REMOTE_JID,
                clinic_id=CLINIC_ID,
                push_name="Maria",
                instance_id="test-instance",
            )

        assert "schedule_session_data" in ctx, (
            "load_session must include schedule_session_data in the returned context "
            f"(ctx keys: {sorted(ctx.keys())})"
        )
        assert ctx["schedule_session_data"] == schedule_data

    def test_load_session_missing_column_degrades_gracefully(self) -> None:
        """Pre-migration deploy: column absent from row → ctx["schedule_session_data"] = []."""
        from app.session import manager

        row = _mk_session_row(schedule_session_data="__OMIT__")
        sb = _mk_load_supabase_mock(session_row=row)

        with patch.object(manager, "get_supabase", return_value=sb):
            ctx = manager.load_session(
                remote_jid=REMOTE_JID,
                clinic_id=CLINIC_ID,
                push_name="Maria",
                instance_id="test-instance",
            )

        assert ctx.get("schedule_session_data") == [], (
            "missing schedule_session_data column must degrade to [] "
            f"(got {ctx.get('schedule_session_data')!r})"
        )
