# Contributing to Sofia

This document covers the **non-negotiable process invariants** for working in
this repo. Code style and architecture decisions live in their respective ADR
files under `docs/adr/`.

---

## Schema migrations: PR-only

> **The repository is the source of truth for the production schema.
> Supabase Studio is a debugging tool, not a deploy tool.**

Concretely, this means:

1. **Every schema change ships as a SQL file in `supabase/migrations/` and
   lands in `main` via a reviewed pull request.** No exceptions.

2. **Do not apply DDL via Supabase Studio against the production project
   `brasilnatech`.** If you do (during a debugging session, an outage, or
   exploratory work), you must:
   - Open a PR with the matching migration before the end of the session.
   - Or revert the change in Studio and add the migration through the
     normal flow.

3. **Schema changes from other apps** (Legacy Analyzer, Onboarding/Dashboard,
   Guardião, Sourcing) must land via *their own repos' migrations*, applied
   through *their own deploy pipelines*. Do not piggyback another app's
   schema change on a Sofia PR, and vice versa.

4. **Migration filenames** follow `NNN_short_slug.sql` where `NNN` is a zero-
   padded sequence number. Increment monotonically; do not reuse a number
   even if a previous file was deleted.

5. **Every migration must be idempotent.** Use `CREATE TABLE IF NOT EXISTS`,
   `ALTER TABLE … ADD COLUMN IF NOT EXISTS`, `INSERT … ON CONFLICT …`,
   and so on. Production has been hand-edited via Studio for months; the
   migration must apply cleanly against both that drift state and a fresh
   dev DB.

6. **Seeds belong in their own migration**, not bundled with DDL. Name them
   `NNN_seed_<entity>.sql`.

### Why

Until 2026-05-04, ~80 schema objects existed in the production
`brasilnatech` DB that were not declared anywhere in any migration. Migrations
had been applied directly via Studio (or via sibling apps' deploy paths)
and never committed back. The drift broke:

- **Reproducibility** — a fresh dev clone could not re-build the prod schema.
- **Audit** — no record of who changed what, when, or why.
- **Rollback** — no down-migrations because there was no up-migration.
- **New deploys** — no way to bring up a second instance.

The drift audit and remediation lives in
[`docs/migrations/drift-audit-2026-05-04.md`](docs/migrations/drift-audit-2026-05-04.md)
and [`supabase/migrations/015_drift_snapshot_2026_05_04.sql`](supabase/migrations/015_drift_snapshot_2026_05_04.sql).
This rule exists so that we never re-create that situation.

### Enforcement

- PR review: any PR that mentions a schema change in its description but does
  not add a `supabase/migrations/NNN_*.sql` file is blocked at review.
- Whenever you suspect drift, query
  `supabase_migrations.schema_migrations` and diff against
  `supabase/migrations/` listing. If you find drift, open an issue tagged
  `drift` and follow the same process: audit → snapshot migration → ADR if
  there is a runtime decision attached.

---

## Commits

Every commit must end with the trailer:

```
Co-Authored-By: Paperclip <noreply@paperclip.ing>
```

(Required by the agent runtime — do not strip.)

---

## Architecture decisions

When a change has cross-cutting consequences (canonical sources of data,
runtime contracts between services, irreversible API shape, etc.), record
the decision as an ADR in `docs/adr/NNNN-slug.md`. ADRs are named
`NNNN-` (4-digit sequence). The latest is
[`0002-dna-canonical.md`](docs/adr/0002-dna-canonical.md).

Use the lens **two-way door vs one-way door**: if the decision is reversible
(the cost of going back later is low), pick fast and document briefly. If
it is one-way (e.g., a write contract another team builds against, an
external API shape, anything that crosses a deploy boundary), spend the time
and write the ADR with full context, alternatives, and reversibility notes.
