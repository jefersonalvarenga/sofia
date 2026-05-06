# Schema drift audit — `brasilnatech` × `sofia` repo (2026-05-04)

**Scope:** identify schema objects that exist in the production Supabase project
`brasilnatech` (`sqbidkenikaqlvdlizot`, region `sa-east-1`) but are missing from
`supabase/migrations/` in this repo (`sofia` @ `ff512d4`,
branch `feature/iris-greeting-smoke`).

**Why this audit exists:** during [EASAA-23](../../../EASAA/issues/EASAA-23) (C2 — Iris
greeting smoke) we discovered tables (notably `sf_assistant_profile`) populated
in production with a behavioral profile per clinic, but absent from any
migration in this repo. The hypothesis was confirmed: migrations have been
applied via Supabase Studio (or via sibling apps' deploy paths) and never
committed back here. This audit lists the drift, classifies ownership, and
locks the canonical decision so [EASAA-28](../../../EASAA/issues/EASAA-28) and
[EASAA-29](../../../EASAA/issues/EASAA-29) can ship against the right source.

The DDL-level remediation lives in
[`supabase/migrations/015_drift_snapshot_2026_05_04.sql`](../../supabase/migrations/015_drift_snapshot_2026_05_04.sql).
The runtime decision lives in [`docs/adr/0002-dna-canonical.md`](../adr/0002-dna-canonical.md).

---

## TL;DR

- DB has **79 applied migrations**. Repo has **14**. Most of the gap belongs
  to **other apps** (Legacy Analyzer, Onboarding/Dashboard, Closer, Sourcing)
  whose migrations never lived here. That is *expected* drift — those repos
  own those tables.
- The **drift that matters for Sofia** is narrow:
  - **`sf_assistant_profile`** (27 cols, 13 rows) — primary DNA table written
    by the Onboarding flow; **never declared in this repo**, **never read by
    `app/session/manager.py`** in `a3b352e`.
  - **`sf_agent_profiles`** (17 cols, 6 rows) — secondary DNA-shaped table.
  - Columns added to `sf_clinics`, `sf_clinic_services`, `sf_clinic_profiles`
    by Onboarding/Dashboard.
  - `la_blueprints.blueprint_json` actually uses the
    `g1_identidade / g2_tom_voz / g3_venda / g4_fluxo / g5_conhecimento /
    g6_inteligencia_comercial` schema produced by Legacy Analyzer 2.0 — **not**
    the `shadow_dna_profile / agent_identity / conversational_flow` schema
    `load_style()` reads. Today, `load_style()` fails through to the
    `sf_clinic_business_rules` fallback or to defaults for **every clinic**
    with a real LA blueprint.
- **Canonical DNA decision: Option C (hybrid with fixed priority).**
  See ADR 0002. Priority chain: `sf_assistant_profile` →
  `la_blueprints` (g2_tom_voz subset) → `sf_clinic_business_rules` →
  defaults.

---

## Repo state at `ff512d4`

```
supabase/migrations/
├── 001_sofia_agent_name.sql
├── 002_sofia_agent_activations.sql
├── 004_seed_clinica_sgen.sql
├── 005_instance_clinic_map.sql
├── 007_create_sf_appointments.sql
├── 008_rename_tables_sf_prefix.sql
├── 009_sofia_v2_observability.sql
├── 010_marketing_attribution.sql
├── 012_seed_lumina_estetica.sql
├── 013_la_blueprints_clinic_id.sql
└── 014_seed_clinica_vitoria.sql
```

(Numbers 003, 006, 011 are skipped; that gap is repo history, not drift.)

## DB state — applied migrations (79)

`SELECT name FROM supabase_migrations.schema_migrations ORDER BY version;`
returns 79 entries. Reproduced verbatim in
[appendix A](#appendix-a--full-migration-list-from-db) below.

---

## Public tables in DB, classified by ownership

| Owner            | Tables                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sofia (this repo)** | `sf_active_chats`, `sf_agent_activations`, `sf_appointments`, `sf_appointments_legacy`, `sf_clinic_business_rules`, `sf_clinic_offers`, `sf_clinic_profiles`, `sf_clinic_services`, `sf_clinics`, `sf_conversations`, `sf_customers`, `sf_instance_clinic_map`, `sf_sessions`                                                                                                                  |
| **Sofia — drift (this audit)** | `sf_assistant_profile`, `sf_agent_profiles`, `sf_resources`, `sf_resource_schedules`, `sf_resource_exceptions`, `sf_resource_services`, `sf_resource_health_insurances`, `sf_service_health_insurances`, `sf_specialties`, `sf_health_insurances`, `sf_clinic_payment_config`, `sf_clinic_ad_accounts`, `sf_payment_transactions`, `sf_onboarding_events`, `sf_ad_clicks`, `sf_ad_spend`, `sf_campaign_proposals`, `sf_campaign_audit_log` |
| **Legacy Analyzer** (separate repo) | `la_analysis_jobs`, `la_analysis_reports`, `la_blueprints`, `la_chat_analyses`, `la_clients`, `la_clinic_features`, `la_conversations`, `la_features`, `la_messages`, `la_training_exports`                                                                                                                                                                                              |
| **Guardião / GK** (separate repo)   | `gk_conversations`, `gk_discovered_cases`, `gk_events`, `gk_leads`, `gk_logs`, `gk_messages`                                                                                                                                                                                                                                                                                                |
| **Onboarding / Dashboard** (separate repo) | `onboarding_sessions`, `organizations`, `clinic_users`, `audit_logs`, `nps_surveys`, `system_logs`                                                                                                                                                                                                                                                                                |
| **SDR / Sourcing** (separate repo) | `cities_market_data`, `cities_market_tiers` (view), `clinic_ad_creatives`, `clinic_decisors`, `clinic_sales`, `closer_conversations`, `closer_messages`, `dev_synthetic_fixtures`, `doctor_profiles`, `epsilon_state`, `facebook_page_signals`, `google_ads_signals`, `google_maps_signals`, `lead_media`, `leads`, `meta_ads_signals`, `outbound_messages_history`, `outbound_queue`, `patient_profiles`, `pitch_performance` (view), `platform_job_logs`, `platform_jobs`, `sdr_config`, `sdr_debounce`, `search_run_chain`, `search_runs`, `search_terms` |
| **Evolution API** (self-hosted, separate Postgres user)   | All `Pascal-case` tables: `Chat`, `Chatwoot`, `Contact`, …, `Webhook`, `Websocket`, plus `_prisma_migrations`. **Out of scope for any sofia migration; will be split to a dedicated Postgres per [feedback memory](#).** |
| **Generic / shared** | `appointments` (legacy, pre-`sf_` rename), `chat_*_view`, `chat_heat_history`, `chat_offers`, `conversations`, `messages`, `messages_status`, `sessions`, `view_context_hydration`, `vw_attribution_funnel`, `vw_campaign_performance` |

Only the **"Sofia — drift"** row is in scope for this audit. The "other-app"
rows are listed for completeness and explicitly **not** to be backfilled into
this repo's `supabase/migrations/`.

---

## Sofia drift details

### `sf_assistant_profile` — primary DNA (canonical, per ADR 0002)

- **Cardinality:** 13 rows (one per clinic that has been onboarded), unique on
  `clinic_id`.
- **FK:** `clinic_id → sf_clinics(id) ON DELETE CASCADE`.
- **Comment in DB:** *"Identidade comportamental da Sofia por clínica.
  Editável manualmente, independente do Blueprint gerado pelo Legacy Analyzer.
  Prioridade máxima em load_style()."*
- **Last touched:** 2026-04-30 (3 distinct days of writes in last week).
- **Created in DB by migration:** `setup_review_and_assistant_profile`
  (2026-04-28) — applied via Studio, not committed here.
- **Consumers:** **none in this repo.** Confirmed by
  `grep -r sf_assistant_profile`. Written by the Onboarding/Dashboard repo via
  the "setup review" flow.
- **Schema (27 cols):** see snapshot migration.
- **Risk if ignored:** Iris greeting reads generic defaults instead of the
  marketeiro_alegre / cordial_amigavel tone configured per clinic.

### `sf_agent_profiles` — secondary DNA

- **Cardinality:** 6 rows. Unique on `clinic_id`. RLS enabled.
- **Cols:** `id, clinic_id, persona_name, tone, personality_traits[],
  attendance_flow[], greeting_example, closing_example, avg_response_tokens,
  forbidden_terms[], common_objections[], organization_id, type, related_id,
  profile jsonb, created_at, updated_at`.
- **Shape note:** the columns are an exact projection of the dict
  `load_style()` returns. Likely a precomputed cache fed by the dashboard.
- **Created by migration:** `015_sf_agent_profiles` (2026-03-16) — applied via
  Studio, not committed here.
- **Consumers in this repo:** **none.**
- **Decision (ADR 0002):** kept in the schema snapshot for reproducibility,
  but **not** elevated to canonical. Treat as a derived/cache table; do not
  add a fourth lookup tier in `load_style()` for it.

### `la_blueprints` — Legacy Analyzer canonical output (cross-app)

- Owned by the Legacy Analyzer repo. This repo only ever needed
  `clinic_id` (added in `013_la_blueprints_clinic_id.sql`).
- **Cardinality:** 27 rows across 21 clinics. All have `clinic_id` populated.
- **Real production schema** (per row sampled from clinic
  `5e1328ff-…-0efa506be6fb`):

  ```
  blueprint_json keys:
    g1_identidade
    g2_tom_voz                 ← direct mirror of sf_assistant_profile fields
    g3_venda
    g4_fluxo                   ← attendance flow + confirmation copy
    g5_conhecimento
    g6_inteligencia_comercial
    metadata                   ← analyzer_version, llm_model, message_count, ...
  ```

- **Mismatch with repo `load_style()`:** `app/session/manager.py:204` reads
  `bp.get("shadow_dna_profile", {}) / agent_identity / conversational_flow`.
  None of those keys exist in the LA-2.0 output. Result: **the la_blueprints
  branch is dead code in production** — it returns nothing useful and
  silently falls through to `sf_clinic_business_rules` (only 2 clinics) or
  to defaults (all the rest).
- **Decision (ADR 0002):** when consumed by Sofia, the LA blueprint is read
  from `g1_identidade / g2_tom_voz / g4_fluxo` keys. The
  `shadow_dna_profile / agent_identity / conversational_flow` projection used
  by migration 014 is a manual-seed convenience for Vitória and is being
  superseded by sf_assistant_profile.

### Other Sofia-prefix tables not in repo

`sf_resources`, `sf_resource_schedules`, `sf_resource_exceptions`,
`sf_resource_services`, `sf_resource_health_insurances`,
`sf_service_health_insurances`, `sf_specialties`, `sf_health_insurances`,
`sf_clinic_payment_config`, `sf_clinic_ad_accounts`, `sf_payment_transactions`,
`sf_onboarding_events`, `sf_ad_clicks`, `sf_ad_spend`, `sf_campaign_proposals`,
`sf_campaign_audit_log` — all created by the Dashboard/Onboarding flow.

These are **not** read by `app/session/manager.py` in this repo today; they
power the dashboard UI. They are documented here for inventory but not
backfilled into this repo's migrations until Sofia actually consumes them.
Unblocking C7/C8 ([EASAA-28](../../../EASAA/issues/EASAA-28),
[EASAA-29](../../../EASAA/issues/EASAA-29)) does not require them.

### Columns added to existing Sofia tables (not in repo)

Drift on tables this repo *does* own:

- `sf_clinics`: `whatsapp` nullable, `auth_user_id` (+ unique), `name`,
  `utm_params`, `website_url`, `onboarding_step`, `onboarding_status`,
  `plan`, `structured_address`, `realtime` enabled. Migrations:
  `add_easyscale_onboarding_columns`, `add_onboarding_step_and_status_to_sf_clinics`,
  `sf_clinics_replica_identity_full`, `sf_clinics_whatsapp_nullable`,
  `sf_clinics_dedup_auth_user_id`, `sf_clinics_auth_user_id_unique`,
  `sf_clinics_add_name`, `sf_clinics_utm_params`,
  `add_profile_fields_to_sf_clinics`, `sf_clinics_realtime`,
  `sf_clinics_onboarding_status_lifecycle`, `onboarding_status_simplify`,
  `onboarding_step`, `add_website_url_to_sf_clinics`,
  `plan_and_structured_address`.
- `sf_clinic_services`: `add_wizard_fields_to_sf_clinic_services`,
  `add_unique_clinic_service_name`, `sf_clinic_services_pricing`,
  `add_price_to_sf_clinic_services`,
  `payment_instructions_services` (column).
- `sf_appointments`: `20260414_sf_appointments_recreate` (full recreate),
  `20260414_seed_appointments`.
- `sf_customers`: `20260414_sf_customers` (table introduced post-rename;
  this repo references `sf_customers` from `manager.py` but does not declare
  it).
- `gk_*`: `add_is_homolog_to_gk_logs`, `add_waiting_to_detected_persona_check`,
  `expand_gk_messages_message_type_check`,
  `add_menu_bot_and_denied_status_to_gk_conversations`,
  `gk_add_detected_persona`, `create_gk_logs`. (Owned by Guardião repo —
  *not* in our scope.)

The drift on `sf_clinics` columns is **the riskiest** because Sofia reads
`sf_clinics` directly (via `sf_clinic_profiles` / `sf_instance_clinic_map`).
Backfilling the column list is included in the snapshot migration, gated
with `IF NOT EXISTS` so it is a no-op in prod.

---

## Snapshot migration scope

`supabase/migrations/015_drift_snapshot_2026_05_04.sql` declares — idempotently
— **only** the Sofia-owned drift that affects what `manager.py` (current or
near-future) reads or writes:

1. `CREATE TABLE IF NOT EXISTS sf_assistant_profile` + comment + indexes + FK.
2. `CREATE TABLE IF NOT EXISTS sf_agent_profiles` + RLS-on + indexes + FK.
3. `ALTER TABLE sf_clinics ADD COLUMN IF NOT EXISTS …` for every column added
   in prod after `008_rename_tables_sf_prefix.sql`.
4. `ALTER TABLE sf_clinic_services ADD COLUMN IF NOT EXISTS …` for the
   wizard/pricing columns.
5. `CREATE TABLE IF NOT EXISTS sf_customers` + indexes (used by
   `manager.py:54` for upserts).

Every other drift entry (LA, GK, Onboarding, SDR) is **excluded by design**.
Those tables belong to other repos and will be backfilled there. Putting them
here would create false ownership and break the "this repo's
`supabase/migrations/` is the source of truth for Sofia" invariant we are
re-establishing.

---

## Process remediation

See `CONTRIBUTING.md` (newly added). Summary:

- Migrations enter prod **only** through merged PRs. Studio is a debugging
  tool, not a deploy tool.
- Any out-of-band schema change must be reverted to a migration file *or*
  promoted to a migration file before the next deploy.
- The CI of the dashboard/onboarding repo must apply its own migrations
  through its own PRs, not through Studio in `brasilnatech`.

---

## Appendix A — Full migration list from DB

```
20260224190153  004_seed_clinica_sgen
20260224194733  005_instance_clinic_map
20260224215842  006_la_blueprints
20260225102149  007_fix_sgen_business_rules
20260225170005  008_fix_sgen_business_rules_v2
20260225171536  009_reseed_sgen_services
20260225174355  010_add_conversation_stage_to_sessions
20260225192834  005_create_sf_sessions
20260225195031  006_fix_sf_agent_activations_fk
20260226110512  create_sf_appointments
20260226112540  008_rename_tables_sf_prefix
20260303002353  sofia_v2_observability
20260303103205  010_marketing_attribution
20260304131831  012_seed_lumina_estetica
20260306150121  013_la_blueprints_clinic_id
20260306152938  gk_add_detected_persona
20260306222415  create_gk_logs
20260307120812  add_is_homolog_to_gk_logs
20260309125704  add_waiting_to_detected_persona_check
20260310200710  expand_gk_messages_message_type_check
20260310202236  add_menu_bot_and_denied_status_to_gk_conversations
20260312181936  add_easyscale_onboarding_columns
20260312182120  make_sf_clinics_name_nullable
20260313111258  add_onboarding_step_and_status_to_sf_clinics
20260313120634  sf_clinics_replica_identity_full
20260316133121  sf_clinics_whatsapp_nullable
20260316170209  014_sf_resources
20260316183141  015_sf_agent_profiles
20260317020830  sf_resource_schedules_exceptions
20260317022443  016_cleanup_resource_schedules
20260318172321  phase9_blueprints_clinic_id_index_and_client_id_nullable
20260318174019  017_seed_lumina_schedules
20260320194115  onboarding_sessions
20260320194118  sf_resources_sofia_enabled
20260320195342  sf_clinics_dedup_auth_user_id
20260320195350  sf_clinics_auth_user_id_unique
20260321205907  20260321_blueprint_chat
20260322130523  sf_clinics_add_name
20260322150337  create_organizations
20260324100615  sf_clinics_utm_params
20260324120035  add_wizard_fields_to_sf_clinic_services
20260324120050  add_unique_clinic_service_name
20260324121051  add_profile_fields_to_sf_clinics
20260324121910  remove_generic_from_sf_resources_type
20260324165854  sf_clinic_services_pricing
20260324173650  sf_resource_services
20260325113959  sf_health_insurances
20260330114326  sf_specialties
20260330200443  sf_campaign_proposals
20260330200521  sf_campaign_proposals_seed
20260331113841  add_website_url_to_sf_clinics
20260331150220  payment_instructions_clinics
20260331150225  payment_instructions_services
20260331150231  payment_instructions_resources
20260413204112  add_price_to_sf_clinic_services
20260413223831  create_la_features
20260413223838  create_la_clinic_features
20260413223847  seed_la_features_catalog
20260414120802  20260414_sf_customers
20260414122338  20260414_sf_appointments_recreate
20260414122814  20260414_seed_appointments
20260424123446  plan_and_structured_address
20260424172958  dev_synthetic_fixtures
20260424175708  upsert_dev_synthetic_fixture_rpc
20260424194212  dev_synthetic_fixtures_progress
20260427183816  sf_clinics_realtime
20260427183954  dev_migrate_fixture_to_message
20260427204250  la_analysis_jobs_realtime
20260427204930  la_analysis_jobs_add_clinic_id
20260427205257  la_job_status_add_pending
20260427205920  la_analysis_jobs_client_id_nullable
20260428102726  sf_clinics_onboarding_status_lifecycle
20260428122021  setup_review_and_assistant_profile
20260428195954  la_analysis_jobs_chunk_progress
20260429001859  la_analysis_jobs_eta_finished_at
20260429125807  onboarding_status_simplify
20260429183954  onboarding_review
20260429203532  onboarding_review_add_test_drive
20260429212226  onboarding_step
20260504225034  014_seed_clinica_vitoria
20260504225154  014_seed_clinica_vitoria
```

(The two `014_seed_clinica_vitoria` entries are an idempotency artifact from
running migration 014 twice during EASAA-23; the second was a no-op. Cleanup
out of scope for this audit.)
