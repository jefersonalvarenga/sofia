# ADR 0002 — Canonical source of clinic DNA for Sofia

- **Status:** Accepted
- **Date:** 2026-05-04
- **Deciders:** CTO (this ADR), CEO (sign-off via [EASAA-36](../../../EASAA/issues/EASAA-36))
- **Related:** [EASAA-23](../../../EASAA/issues/EASAA-23) (C2 — Iris greeting smoke),
  [EASAA-28](../../../EASAA/issues/EASAA-28) (C7 — webhook),
  [EASAA-29](../../../EASAA/issues/EASAA-29) (C8 — pipeline)

## Context

Sofia, the WhatsApp assistant, must respond with the **tone, greeting style,
pricing policy, qualification flow, etc.** configured per clinic. We call
that bundle **clinic DNA**. Today there are four candidate sources of DNA in
the production database `brasilnatech`:

1. `sf_assistant_profile` — 27-column flat table with per-clinic behavioral
   identity. **13 rows in production**, written by the Onboarding/Dashboard
   "setup review" flow. Last touched 2026-04-30. Cardinality 1:1 with
   `sf_clinics`. **Not declared in this repo** until migration 015.
2. `la_blueprints.blueprint_json` — JSONB output of the Legacy Analyzer,
   schema `g1_identidade / g2_tom_voz / g3_venda / g4_fluxo /
   g5_conhecimento / g6_inteligencia_comercial`. 27 rows across 21 clinics.
   `g2_tom_voz` is a *near-superset* of the columns in `sf_assistant_profile`
   (see drift audit, §`la_blueprints`).
3. `sf_agent_profiles` — 17-column projection into the dict shape that
   `load_style()` returns. 6 rows. RLS enabled.
4. `sf_clinic_business_rules` style-keyed rows
   (`tom_voz`, `personalidade`, `saudacao_exemplo`, `fechamento`,
   `estilo_resposta`). Only 2 clinics have these populated. Legacy.

`app/session/manager.py:load_style()` in `a3b352e` reads
`la_blueprints` first (dead code: schema mismatch — `bp.get("shadow_dna_profile")`
returns `None` against the LA-2.0 output), then `sf_clinic_business_rules`,
then defaults. **`sf_assistant_profile` and `sf_agent_profiles` are not read
by Sofia today.** Yet `sf_assistant_profile` is the one actually populated by
the dashboard for new clinics. Result: in production, ~67 of 69 clinics get
generic-default DNA in Sofia.

This is also a **blocker for Iris** ([EASAA-28](../../../EASAA/issues/EASAA-28),
[EASAA-29](../../../EASAA/issues/EASAA-29)): if C7/C8 implement reads against
`la_blueprints`, the Vitória greeting smoke responds in a generic tone instead
of the configured one. If they implement against `sf_assistant_profile` we
match what the dashboard already writes.

## Options considered

### Option A — `sf_assistant_profile` is the only canonical source

Migrate `load_style()` to read exclusively from `sf_assistant_profile`.
`la_blueprints` becomes "LA history" — kept for analyst review, not consumed
by runtime.

- ✅ Single source of truth. Zero precedence ambiguity.
- ✅ Matches what the dashboard already writes.
- ❌ Loses `la_blueprints.g1_identidade` (services catalog, pricing,
  contraindications) and `g3_venda / g4_fluxo / g5_conhecimento` payloads
  that are richer than what `sf_assistant_profile` carries. Sofia would lose
  context that LA already extracted.
- ❌ Brand-new clinics with a fresh LA run but no human review yet would have
  no DNA at all (they have a blueprint but no `sf_assistant_profile` row).

### Option B — `la_blueprints` is canonical

Treat `sf_assistant_profile` as deprecated. Migrate everything to read from
`la_blueprints` with the real `g1…g6` schema.

- ✅ Single source.
- ✅ Always populated for any clinic that has run LA.
- ❌ Manual edits via the dashboard (`sf_assistant_profile`) get **silently
  lost** at runtime. The "Edit DNA" UX becomes a lie.
- ❌ Any clinic without a Legacy Analyzer run yet (e.g. Vitória, the Iris
  smoke fixture) has no DNA.
- ❌ The dashboard team is already shipping against
  `sf_assistant_profile`. Forcing them to migrate to `la_blueprints` writes
  is a one-way door we cannot afford pre-validation.

### Option C — Hybrid with explicit priority (chosen)

Read order:

1. **`sf_assistant_profile`** — if a row exists for `clinic_id`, use it.
   This is the "human curation wins" tier.
2. **`la_blueprints.blueprint_json` (g1/g2/g4 keys)** — fallback when no
   `sf_assistant_profile` row exists. This is the "machine-extracted
   baseline" tier.
3. **`sf_clinic_business_rules`** style keys — fallback for clinics still
   on the legacy seed shape (Sgen). Marked for retirement once Sgen is
   re-seeded into `sf_assistant_profile`.
4. **Generic defaults** — last resort.

`sf_agent_profiles` is **explicitly excluded** from the read chain. It is a
dashboard cache and adding it as a fourth tier just creates write-amp risk.

## Decision

**Option C.** `sf_assistant_profile` is canonical; `la_blueprints` is the
machine-extracted fallback; `sf_clinic_business_rules` is legacy and will be
deprecated; generic defaults catch the remaining holes.

## Why C and not A

A is cleaner, but right now it deletes context the LA produces (g3/g4/g5
fields that the dashboard does not yet surface in `sf_assistant_profile`). The
two-way door is C: we can collapse to A later when the dashboard's DNA editor
covers all the LA-extracted fields. Going to A today would lose information
in production, which is a one-way door.

## Why C and not B

B would work technically — the LA blueprint is a superset of what we need.
But the dashboard already writes to `sf_assistant_profile` and is the
human-facing edit point. Honoring those edits is a product invariant, not a
technical one. Forcing all DNA edits through `la_blueprints` writes would
require a parallel write path on the dashboard side and break the existing
contract.

## Consequences

### Schema (already aligned with this ADR)

- `sf_assistant_profile` is added to this repo via migration 015
  (`015_drift_snapshot_2026_05_04.sql`). FK to `sf_clinics(id)` ON DELETE
  CASCADE, UNIQUE on `clinic_id`, idempotent.
- `la_blueprints.blueprint_json` is read using the **production**
  `g1_identidade / g2_tom_voz / g4_fluxo` keys, not the legacy
  `shadow_dna_profile / agent_identity / conversational_flow` keys baked
  into the Vitória seed. The Vitória seed (migration 014) is updated to
  also insert into `sf_assistant_profile` so Iris has the canonical row;
  the embedded `la_blueprint` row remains as a legacy convenience and will
  be regenerated by the LA on first real run.

### Runtime (load_style refactor — out of scope here, owned by C7/C8)

The new `load_style()` priority must be:

```python
def load_style(clinic_id):
    if row := query_sf_assistant_profile(clinic_id):
        return _project_assistant_profile(row)        # tier 1
    if bp := query_la_blueprint_g_schema(clinic_id):
        return _project_blueprint_g_schema(bp)         # tier 2
    if rules := query_business_rules_style_keys(clinic_id):
        return _project_business_rules(rules)          # tier 3
    return _DEFAULT_STYLE                              # tier 4
```

Field-by-field projection from each tier into the dict that
`load_style()` already returns is documented in
[`docs/adr/0002-dna-canonical-projection.md`](#) — to be filled in when
[EASAA-28](../../../EASAA/issues/EASAA-28) and
[EASAA-29](../../../EASAA/issues/EASAA-29) implement the change.

### Tier-1 → tier-1 mapping (`sf_assistant_profile` → `load_style` dict)

| `load_style()` key      | `sf_assistant_profile` source                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| `tone`                  | `tom_voz`                                                                                              |
| `personality_traits`    | derived: `[tom_voz, nivel_formalidade, comprimento_msg_tipico]` (or kept empty until dashboard supports) |
| `greeting_example`      | first element of `saudacao_inicial[]` if non-empty, else `""`                                          |
| `closing_example`       | first element of `despedida_padrao[]` if non-empty, else `""`                                          |
| `attendance_flow`       | `fluxo_padrao_atendimento[]`                                                                           |
| `avg_response_tokens`   | constant `100` (no column yet — add later)                                                             |
| `forbidden_terms`       | derived from `contraindicacao_policy.terms` if present, else `[]`                                      |
| `common_objections`     | `objecoes_recorrentes[]` (already a list of strings)                                                   |
| `source`                | `"assistant_profile"`                                                                                  |

The **richer** fields in `sf_assistant_profile`
(`politica_preco`, `momento_revela_preco`, `politica_sinal`,
`como_confirma_agendamento`, `follow_up_apos_silencio`, `faq_extraido`,
`procedimentos_explicados`, `casos_de_escalation`) are **kept on the returned
dict under their original keys**. C7/C8 prompt agents can opt into them
without us having to gate on a second pass.

### Tier-2 mapping (`la_blueprints.g…` → `load_style` dict)

Only the fields needed today (Iris greeting smoke + base Sofia):

| `load_style()` key      | `la_blueprints.blueprint_json` path                          |
| ----------------------- | ------------------------------------------------------------ |
| `tone`                  | `$.g2_tom_voz.tom_voz`                                       |
| `personality_traits`    | derived from `g2_tom_voz` flags                              |
| `greeting_example`      | `$.g2_tom_voz.saudacao_inicial[0]`                           |
| `closing_example`       | `$.g2_tom_voz.despedida_padrao[0]`                           |
| `attendance_flow`       | `$.g4_fluxo.fluxo_padrao_atendimento[]`                      |
| `avg_response_tokens`   | constant `100`                                               |
| `forbidden_terms`       | `[]` (LA does not extract this directly today)               |
| `common_objections`     | `$.g3_venda.objecoes_recorrentes[]` if present, else `[]`    |
| `source`                | `"blueprint"`                                                |

### Process

- Both `EASAA-28` (webhook) and `EASAA-29` (pipeline) read from
  `sf_assistant_profile` first. Their tickets are unblocked as of merging
  this ADR + migration 015.
- The Vitória seed (migration 014) inserts into `sf_assistant_profile` so the
  Iris greeting smoke exercises tier 1, not tier 2.
- A future ticket (not opened here) will retire
  `sf_clinic_business_rules` style keys after Sgen is re-seeded.

### What this ADR explicitly does NOT do

- Does not change `app/session/manager.py:load_style()`. That is the
  implementation work in C7/C8. This ADR establishes the contract; the next
  PRs implement it.
- Does not promote `sf_agent_profiles` to a runtime read tier.
- Does not retire `sf_clinic_business_rules` yet (would break Sgen).
- Does not deduplicate the LA blueprint schemas (legacy seed-shape vs.
  g1/g6 production-shape). The seed-shape only exists in the Vitória
  fixture and will be replaced by a real LA run later.

## Reversibility

Two-way door for Options A and C: switching from C to A later is a
read-side change in `load_style()` plus a one-shot ETL from
`la_blueprints` into `sf_assistant_profile`. Easy to do once the dashboard
DNA editor is feature-complete. Going to B from C requires the dashboard
team to switch its writes — that is the actual one-way door, which is why
B is rejected.
