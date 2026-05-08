# Iris tests

Suite for the Iris greeting-smoke pipeline ([EASAA-20](../../EASAA/issues/EASAA-20)).
All tests run with **no network**, **no Anthropic / OpenAI keys**, and **no Supabase** —
external services are stubbed via in-memory fakes (`MagicMock`, `httpx.MockTransport`).

## Running locally

From the repo root, with the project venv active:

```bash
pytest tests/iris/ -v
```

Run a single file:

```bash
pytest tests/iris/test_pipeline_e2e.py -v
```

If pytest blows up at collection with
`sqlite3.OperationalError: attempt to write a readonly database`, your
`$HOME` is not writable and the dspy/litellm disk cache cannot initialize.
Set a writable cache directory:

```bash
DSPY_CACHEDIR=/tmp/dspy_cache pytest tests/iris/
```

`tests/iris/conftest.py` sets `DSPY_CACHEDIR` to `<tempdir>/iris_dspy_cache`
when the env var is unset, so this is only necessary if you override it.

## What each file covers

| File | Scope |
| ---- | ----- |
| `test_idempotency.py` | Webhook UPSERT-on-conflict guarantee. Two POSTs with the same `wamid` → second short-circuits, no extra `sf_messages` or `sf_agent_activations` rows, no second LLM call. |
| `test_pipeline_e2e.py` | Drives the C8 LangGraph subgraph end-to-end with mocked Anthropic + mocked Supabase + `httpx.MockTransport` for Evolution. Asserts `GreetingAgent` runs, `conversation_stage = "greeting"`, outbound `wamid` is delivered. |
| `test_router_iris.py` | Unit tests for `IrisRouterAgent` (Anthropic SDK + tool use). Covers tool schema, history formatting, intent normalization, and the C10 deterministic intent table (`oi → GREETING`, `agendar → SCHEDULE`, `blá blá → UNCLASSIFIED`). |
| `test_evolution_client.py` | Unit tests for the Evolution send/persist client (C9). |
| `test_lint_tenant_id.py` | AST scanner — every `supabase.table(...).select(...).execute()` chain in `app/iris/` and `app/session/manager.py` must filter by `clinic_id`. Exemptions require an inline `# tenant-lint: exempt — <reason>` comment. See [ADR 0001](../../docs/adr/0001-iris-tenant-isolation.md). |

## Adding a new tenant-scoped query

If the lint flags your PR, the fix is one of:

1. Add `.eq("clinic_id", clinic_id)` to the chain (the default — almost always right).
2. If the table is genuinely clinic-agnostic (bootstrap maps, composite-key
   joins where the key already encodes clinic_id), annotate one line of the
   chain with `# tenant-lint: exempt — <reason>`. Justify the exemption in
   the PR description.

## CI

`pytest tests/iris/` runs on every PR targeting `main` (job `iris-tests`
in [`ci.yml`](../../.github/workflows/ci.yml)). The job has no secrets,
boots in seconds, and is independent from the heavier integration job that
hits real Supabase + OpenAI.
