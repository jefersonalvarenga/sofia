"""Repository for ``sf_intake_questions``.

Spec: kb/07-MVP/Tech/03-Discussoes/schedule/01 - Spec SCHEDULE_INTAKE.md (§4).

Loads the union of clinic-baseline (service_id IS NULL) + service-specific
(service_id = X) intake questions for a given clinic, ordered by ``order``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def load_intake_questions(
    clinic_id: str,
    service_id: Optional[str],
    client: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Load intake questions (union baseline + service-specific) ordered by ``order``.

    Args:
        clinic_id: clinic uuid.
        service_id: service uuid; when None, returns only clinic baseline rows.
        client: optional Supabase client (testing seam). Falls back to
            :func:`app.core.supabase_client.get_supabase` when not provided.

    Returns:
        List of dicts with keys: ``id``, ``order``, ``question_text``,
        ``category``, ``is_required``, ``source`` (``"clinic"`` or ``"service"``).
        Empty list when the clinic has no questions configured (opt-out).
    """
    sb = client if client is not None else _get_default_client()

    # Pull clinic baseline (service_id IS NULL) AND, if given, service-specific.
    baseline_rows = (
        sb.table("sf_intake_questions")
        .select("id, \"order\", question_text, category, is_required, service_id")
        .eq("clinic_id", clinic_id)
        .is_("service_id", "null")
        .execute()
        .data
        or []
    )

    service_rows: List[Dict[str, Any]] = []
    if service_id is not None:
        service_rows = (
            sb.table("sf_intake_questions")
            .select("id, \"order\", question_text, category, is_required, service_id")
            .eq("clinic_id", clinic_id)
            .eq("service_id", service_id)
            .execute()
            .data
            or []
        )

    rows = list(baseline_rows) + list(service_rows)

    result: List[Dict[str, Any]] = []
    for row in rows:
        source = "clinic" if row.get("service_id") is None else "service"
        result.append(
            {
                "id": row["id"],
                "order": row["order"],
                "question_text": row["question_text"],
                "category": row["category"],
                "is_required": row["is_required"],
                "source": source,
            }
        )

    # Stable ordering: primary by `order`, then `clinic` before `service`,
    # then by `id` as ultimate tiebreaker.
    result.sort(key=lambda q: (q["order"], 0 if q["source"] == "clinic" else 1, q["id"]))
    return result


def _get_default_client() -> Any:
    """Lazy import to keep this module testable without Supabase config."""
    from app.core.supabase_client import get_supabase

    return get_supabase()
