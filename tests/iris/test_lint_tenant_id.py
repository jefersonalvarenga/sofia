"""
Iris C10 — clinic_id tenant-isolation lint.

AST scanner that walks every ``supabase.table(...).select(...).execute()``
chain in the Iris greeting pipeline and asserts that ``.eq("clinic_id", X)``
is part of the chain. The single exemption mechanism is an inline comment
on a line of the chain:

    # tenant-lint: exempt — <reason>

The comment must include the literal string ``tenant-lint: exempt`` (or
``tenant-lint: exempt:``) somewhere in the line, plus a free-text reason
after a dash or colon. The lint is intentionally strict: composite-key
joins and bootstrap maps must opt out explicitly per-line.

Why a custom AST lint instead of a runtime check:
  - clinic_id leaks are silent at runtime — the wrong row is just returned.
  - The number of Supabase queries grows monotonically; we want every new
    SELECT chain to be reviewed for tenant scoping at PR time.
  - mypy / ruff cannot express ``every chain on Supabase MUST include eq.``

See [ADR 0001](../../docs/adr/0001-iris-tenant-isolation.md) and
[EASAA-31](../../EASAA/issues/EASAA-31).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Optional, Set


REPO_ROOT = Path(__file__).resolve().parents[2]
EXEMPT_RE = re.compile(r"#\s*tenant-lint:\s*exempt\s*[:\-—]\s*\S", re.IGNORECASE)


def _iris_python_files() -> List[Path]:
    """Files in scope for the lint."""
    iris_dir = REPO_ROOT / "app" / "iris"
    files = sorted(p for p in iris_dir.glob("*.py") if not p.name.startswith("_"))
    files.append(REPO_ROOT / "app" / "session" / "manager.py")
    return files


# ----------------------------------------------------------------------------
# AST walker
# ----------------------------------------------------------------------------


class _ChainInfo:
    """Aggregated info about one supabase chain ending in ``.execute()``."""

    __slots__ = ("table_name", "has_select", "eq_fields", "lineno", "end_lineno")

    def __init__(self) -> None:
        self.table_name: Optional[str] = None
        self.has_select: bool = False
        self.eq_fields: List[str] = []
        self.lineno: int = 0
        self.end_lineno: int = 0


def _walk_chain(execute_call: ast.Call) -> _ChainInfo:
    """
    Walk back from ``.execute()`` along the call chain via ``.func.value``,
    collecting table name, eq fields, and whether ``.select(...)`` appears.
    """
    info = _ChainInfo()
    info.lineno = execute_call.lineno
    info.end_lineno = getattr(execute_call, "end_lineno", execute_call.lineno)

    node: Optional[ast.AST] = execute_call.func  # ast.Attribute(.execute)
    while isinstance(node, ast.Attribute):
        outer = node
        inner = outer.value

        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
            method = inner.func.attr
            args = inner.args

            if method == "table" and args:
                first = args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    info.table_name = first.value
            elif method == "select":
                info.has_select = True
            elif method == "eq" and args:
                first = args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    info.eq_fields.append(first.value)

            info.lineno = min(info.lineno, inner.lineno)
            node = inner.func
        else:
            break

    return info


def _find_select_chains(tree: ast.AST) -> List[_ChainInfo]:
    """All chains in ``tree`` that end with ``.execute()`` AND include ``.select()``."""
    results: List[_ChainInfo] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "execute"):
            continue
        info = _walk_chain(node)
        if info.has_select and info.table_name:
            results.append(info)
    return results


def _exempt_lines(source: str) -> Set[int]:
    """Line numbers (1-based) carrying a ``# tenant-lint: exempt — ...`` comment."""
    exempt: Set[int] = set()
    for idx, line in enumerate(source.splitlines(), start=1):
        if EXEMPT_RE.search(line):
            exempt.add(idx)
    return exempt


# ----------------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------------


def _format_violation(path: Path, info: _ChainInfo) -> str:
    rel = path.relative_to(REPO_ROOT)
    fields = ", ".join(info.eq_fields) or "<none>"
    return (
        f"  {rel}:{info.lineno}-{info.end_lineno} — "
        f"table={info.table_name!r} eq fields=[{fields}]"
    )


def test_iris_select_chains_filter_by_clinic_id() -> None:
    """Every SELECT chain in scope must filter by clinic_id (or be exempt)."""
    violations: List[str] = []

    for path in _iris_python_files():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        exempt = _exempt_lines(source)

        for info in _find_select_chains(tree):
            if "clinic_id" in info.eq_fields:
                continue

            chain_lines = set(range(info.lineno, info.end_lineno + 1))
            if chain_lines & exempt:
                continue

            violations.append(_format_violation(path, info))

    assert not violations, (
        "Tenant isolation violation — Supabase SELECT chains missing "
        '`.eq("clinic_id", ...)` and not marked exempt:\n'
        + "\n".join(violations)
        + "\n\nFix by adding `.eq(\"clinic_id\", clinic_id)` to the chain, "
          "or annotate one line of the chain with "
          "`# tenant-lint: exempt — <reason>`. "
          "See ADR 0001."
    )


def test_lint_finds_at_least_three_chains() -> None:
    """Sanity guard — if the AST walker breaks, this fires before false-passes."""
    chains = []
    for path in _iris_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        chains.extend(_find_select_chains(tree))

    assert len(chains) >= 3, (
        "Expected the AST walker to find at least 3 SELECT chains across "
        f"app/iris and app/session/manager.py — got {len(chains)}. "
        "The walker is likely broken."
    )
