"""
Iris C10 — shared fixtures + import-time guards.

The Iris pipeline imports ``app.core.telemetry``, which imports ``dspy``.
On import, dspy/litellm initialize a ``diskcache.Cache`` under
``$DSPY_CACHEDIR`` (default: ``$HOME/.dspy_cache``). On read-only or
otherwise locked HOME directories the cache write fails with
``sqlite3.OperationalError: attempt to write a readonly database`` and
collection blows up before a single test runs.

We set ``DSPY_CACHEDIR`` to a writable temp directory at import time so
both local pytest invocations and CI containers behave identically.
"""

from __future__ import annotations

import os
import tempfile

# Must run BEFORE any `app.*` import in this directory, so it lives at the
# very top of conftest.py. Tests import dspy transitively via
# `app.core.telemetry` → tests in this dir then succeed in any HOME shape.
os.environ.setdefault(
    "DSPY_CACHEDIR",
    os.path.join(tempfile.gettempdir(), "iris_dspy_cache"),
)
