"""Repository tests for ``load_intake_questions``.

Spec: kb/07-MVP/Tech/03-Discussoes/schedule/01 - Spec SCHEDULE_INTAKE.md (§4.2).

Loads union (clinic baseline + service-specific) from ``sf_intake_questions``,
ordered by ``order``. Uses a fake Supabase client to avoid network calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Fake Supabase client
# ---------------------------------------------------------------------------


class _FakeQuery:
    """Mimics the subset of Supabase query builder methods we exercise."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = list(rows)
        self._filters: List[Any] = []

    def select(self, _cols: str) -> "_FakeQuery":
        return self

    def eq(self, col: str, value: Any) -> "_FakeQuery":
        self._rows = [r for r in self._rows if r.get(col) == value]
        return self

    def or_(self, expr: str) -> "_FakeQuery":
        # We do not exercise this in load_intake_questions; left as no-op.
        return self

    def is_(self, col: str, value: Any) -> "_FakeQuery":
        # supabase-py uses .is_("col", "null") for NULL checks
        if value == "null" or value is None:
            self._rows = [r for r in self._rows if r.get(col) is None]
        return self

    def in_(self, col: str, values: List[Any]) -> "_FakeQuery":
        self._rows = [r for r in self._rows if r.get(col) in values]
        return self

    def order(self, col: str, **_kwargs: Any) -> "_FakeQuery":
        self._rows.sort(key=lambda r: r.get(col, 0))
        return self

    def execute(self) -> Any:
        class _Result:
            def __init__(self, data):
                self.data = data

        return _Result(list(self._rows))


class FakeSupabase:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def table(self, name: str) -> _FakeQuery:
        assert name == "sf_intake_questions"
        return _FakeQuery(self._rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


CLINIC = "clinic-uuid-1"
SERVICE = "service-uuid-1"
OTHER_SERVICE = "service-uuid-OTHER"


def _row(
    qid: str,
    order: int,
    service_id: Optional[str] = None,
    *,
    clinic_id: str = CLINIC,
    question_text: str = "?",
    category: str = "custom",
    is_required: bool = True,
) -> Dict[str, Any]:
    return {
        "id": qid,
        "clinic_id": clinic_id,
        "service_id": service_id,
        "order": order,
        "question_text": question_text,
        "category": category,
        "is_required": is_required,
    }


@pytest.fixture
def baseline_rows() -> List[Dict[str, Any]]:
    """5 baseline questions for the clinic (service_id=NULL)."""
    return [
        _row("q-med-001", 1, None, question_text="Medicamentos?", category="medicamentos"),
        _row("q-alg-002", 2, None, question_text="Alergias?", category="alergias"),
        _row("q-gest-003", 3, None, question_text="Gravida?", category="gestacao"),
        _row("q-cron-004", 4, None, question_text="Cronicas?", category="cronicas"),
        _row("q-pele-005", 5, None, question_text="Pele?", category="pele"),
    ]


@pytest.fixture
def service_rows() -> List[Dict[str, Any]]:
    """2 service-specific questions ordered after baseline."""
    return [
        _row("q-svc-006", 6, SERVICE, question_text="Já fez botox antes?", category="historico"),
        _row("q-svc-007", 7, SERVICE, question_text="Última aplicação quando?", category="historico"),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadIntakeQuestions:
    def test_returns_empty_list_when_no_rows(self):
        from app.repositories.intake_questions import load_intake_questions

        client = FakeSupabase(rows=[])
        result = load_intake_questions(CLINIC, SERVICE, client=client)
        assert result == []

    def test_returns_baseline_only_when_service_id_none(self, baseline_rows):
        from app.repositories.intake_questions import load_intake_questions

        client = FakeSupabase(rows=baseline_rows)
        result = load_intake_questions(CLINIC, None, client=client)
        assert len(result) == 5
        assert [q["id"] for q in result] == [
            "q-med-001",
            "q-alg-002",
            "q-gest-003",
            "q-cron-004",
            "q-pele-005",
        ]
        # All baseline rows tagged as "clinic" source.
        assert all(q["source"] == "clinic" for q in result)

    def test_returns_union_baseline_and_service_specific(self, baseline_rows, service_rows):
        from app.repositories.intake_questions import load_intake_questions

        client = FakeSupabase(rows=baseline_rows + service_rows)
        result = load_intake_questions(CLINIC, SERVICE, client=client)
        assert len(result) == 7
        # Ordered by `order` ascending.
        assert [q["order"] for q in result] == [1, 2, 3, 4, 5, 6, 7]
        sources = {q["id"]: q["source"] for q in result}
        assert sources["q-med-001"] == "clinic"
        assert sources["q-svc-006"] == "service"

    def test_excludes_other_services_rows(self, baseline_rows):
        from app.repositories.intake_questions import load_intake_questions

        other_service_row = _row(
            "q-other-099",
            99,
            OTHER_SERVICE,
            question_text="other?",
            category="custom",
        )
        client = FakeSupabase(rows=baseline_rows + [other_service_row])
        result = load_intake_questions(CLINIC, SERVICE, client=client)
        ids = {q["id"] for q in result}
        assert "q-other-099" not in ids
        assert len(result) == 5  # only baseline

    def test_excludes_other_clinics_rows(self, baseline_rows):
        from app.repositories.intake_questions import load_intake_questions

        other_clinic_row = _row(
            "q-foreign-100",
            1,
            None,
            clinic_id="clinic-uuid-OTHER",
        )
        client = FakeSupabase(rows=baseline_rows + [other_clinic_row])
        result = load_intake_questions(CLINIC, None, client=client)
        ids = {q["id"] for q in result}
        assert "q-foreign-100" not in ids

    def test_returned_rows_have_required_keys(self, baseline_rows):
        from app.repositories.intake_questions import load_intake_questions

        client = FakeSupabase(rows=baseline_rows)
        result = load_intake_questions(CLINIC, None, client=client)
        for q in result:
            assert {"id", "order", "question_text", "category", "is_required", "source"} <= set(q)

    def test_baseline_visible_when_service_id_given(self, baseline_rows, service_rows):
        """Even with service_id passed, clinic baseline (service_id NULL) is included."""
        from app.repositories.intake_questions import load_intake_questions

        client = FakeSupabase(rows=baseline_rows + service_rows)
        result = load_intake_questions(CLINIC, SERVICE, client=client)
        clinic_ids = [q["id"] for q in result if q["source"] == "clinic"]
        assert set(clinic_ids) == {
            "q-med-001",
            "q-alg-002",
            "q-gest-003",
            "q-cron-004",
            "q-pele-005",
        }
