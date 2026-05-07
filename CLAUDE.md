# CLAUDE.md — Sofia/Iris repo guide for Claude/agents

Use this file as the entry point when an agent (CTO, specialist, reviewer)
opens this repo. It points at the ADRs, the conventions that are not
obvious from code alone, and the gates that block PRs.

## Repository identity

This repo is `sofia` (also evolving as Iris). Backend FastAPI + LangGraph
that powers EasyScale's WhatsApp assistant for aesthetic clinics in Brazil.
One codebase serves N clinics (multi-tenant). Production tenants today:
Lumina, Sgen. Iris greeting smoke fixture: Clínica Vitória.

## Architectural Decision Records

Read the relevant ADR before changing anything in its scope. ADRs live in
[`docs/adr/`](docs/adr/).

| ADR  | Topic                                                       | Owner        |
| ---- | ----------------------------------------------------------- | ------------ |
| 0001 | [Iris tenant isolation](docs/adr/0001-iris-tenant-isolation.md) — backend é defesa real, RLS é cinto+suspensório | CTO (2026-05-07) |
| 0002 | [Canonical clinic DNA](docs/adr/0002-dna-canonical.md) — `sf_assistant_profile` tier 1, `la_blueprints` tier 2 | CTO (2026-05-04) |

## Tenant isolation (canonical reference: [ADR 0001](docs/adr/0001-iris-tenant-isolation.md))

If you are touching code that talks to Supabase, internalize this before
the first edit:

- **`clinic_id` is the tenant key.** Equivalent to `tenant_id` in the Iris
  spec. Every tenant-scoped query, insert, update, delete carries
  `clinic_id` explicitly.
- **Backend is the real defense.** `service_role` bypasses RLS by design;
  the safety net is `.eq("clinic_id", clinic_id)` injected in every query
  and `clinic_id` in every insert payload. Code review blocks PRs without
  it.
- **RLS is enabled but inert today** — `current_setting('app.current_tenant', true)`
  + `SET LOCAL` doesn't work via `supabase-py` REST. Policies are future
  protection (see ADR 0001 §2). Don't be fooled into thinking RLS is what
  blocks cross-tenant reads on the hot path.
- **Idempotency is `UNIQUE (clinic_id, wamid)` + `ON CONFLICT DO NOTHING`**
  in `sf_messages`. No advisory lock. See migration 017.
- **No `clinic_id`? Reject the PR.** Exception: tables that resolve
  `clinic_id` (e.g. `sf_instance_clinic_map`) — mark the line with
  `# tenant-lint: exempt — <reason>`.

## Conventions agents should know without re-reading the codebase

- Tabelas com prefixo `sf_` são as core multi-tenant. `la_blueprints` é
  cross-clinic (também RLS-protegida via `clinic_id`).
- Migrations em [`supabase/migrations/`](supabase/migrations/) são
  numeradas sequencialmente (`NNN_descricao.sql`). Sempre idempotente
  (`CREATE TABLE IF NOT EXISTS`, `DROP POLICY IF EXISTS … CREATE POLICY`).
- DNA da clínica é lida via `app/session/manager.py:load_style(clinic_id)`.
  Precedence chain definida em [ADR 0002](docs/adr/0002-dna-canonical.md).
- GreetingAgent (`app/agents/greeting/agent.py`) é determinístico, zero
  LLM. Decisão registrada em ADR 0001 §5.
- Communication entre humanos (issues, comments, docs de processo) em
  português. Código (variáveis, funções, comentários in-file, commits,
  branches, PRs) em inglês.

## Gates antes de merge

- Build/lint passa local (`pytest`, mypy se aplicável).
- Toda nova query Supabase tem `clinic_id` no escopo. Code review verifica
  manualmente até o lint AST de `tests/iris/test_lint_tenant_id.py`
  existir.
- Migrations testadas via Supabase MCP (`apply_migration` em branch
  Supabase isolada) antes de PR review.
- PR aponta para `main` (gitflow Iris atual: `feature/iris-* → main` direto
  com aprovação CEO por PR). Não usar `develop`.
