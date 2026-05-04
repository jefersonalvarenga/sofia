-- Migration 015 — drift snapshot 2026-05-04
--
-- Purpose: declare in this repo the schema objects that exist in production
-- (`brasilnatech`) but were never committed via a migration here AND that
-- Sofia (this repo) reads or will read in the canonical-DNA path defined
-- by ADR 0002.
--
-- Idempotent. Every statement guards on IF NOT EXISTS so this file is a
-- no-op against the live DB and a build step for fresh dev clones.
--
-- Scope is intentionally narrow. Tables owned by other repos (Legacy
-- Analyzer, Onboarding/Dashboard, GK, SDR/Sourcing) are NOT declared here:
-- those repos own their own migrations. See
-- `docs/migrations/drift-audit-2026-05-04.md` for the full classification.
--
-- Related: docs/adr/0002-dna-canonical.md, EASAA-36, EASAA-28, EASAA-29.

BEGIN;

-- ============================================================================
-- 1. sf_assistant_profile — primary canonical DNA (ADR 0002 tier 1)
--
-- Created in prod by `setup_review_and_assistant_profile` (2026-04-28) via
-- the Onboarding/Dashboard flow. Never committed here. C7/C8 will read it.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.sf_assistant_profile (
  id                          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  clinic_id                   UUID        NOT NULL UNIQUE
                                          REFERENCES public.sf_clinics(id) ON DELETE CASCADE,

  -- Voice
  tom_voz                     TEXT,
  nivel_formalidade           TEXT,
  uso_emoji_frequencia        TEXT,
  uso_emoji_tipos             JSONB       DEFAULT '[]'::jsonb,
  comprimento_msg_tipico      TEXT,
  quebra_de_msg               TEXT,
  saudacao_inicial            JSONB       DEFAULT '[]'::jsonb,
  despedida_padrao            JSONB       DEFAULT '[]'::jsonb,

  -- Pricing & sales policy
  politica_preco              TEXT,
  momento_revela_preco        TEXT,
  educacao_tecnica            TEXT,
  qualificacao_tipica         JSONB       DEFAULT '[]'::jsonb,
  prova_social_uso            TEXT,
  mencao_profissional         TEXT,
  politica_sinal              JSONB       DEFAULT '{}'::jsonb,
  objecoes_recorrentes        JSONB       DEFAULT '[]'::jsonb,
  contraindicacao_policy      JSONB       DEFAULT '{}'::jsonb,

  -- Flow
  fluxo_padrao_atendimento    JSONB       DEFAULT '[]'::jsonb,
  como_confirma_agendamento   TEXT,
  follow_up_apos_silencio     JSONB       DEFAULT '{}'::jsonb,

  -- Knowledge / escalation
  faq_extraido                JSONB       DEFAULT '[]'::jsonb,
  procedimentos_explicados    JSONB       DEFAULT '[]'::jsonb,
  casos_de_escalation         JSONB       DEFAULT '[]'::jsonb,

  created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sf_assistant_profile_clinic
  ON public.sf_assistant_profile (clinic_id);

COMMENT ON TABLE public.sf_assistant_profile IS
  'Identidade comportamental da Sofia por clínica. Editável manualmente via Dashboard, '
  'independente do Blueprint gerado pelo Legacy Analyzer. Tier 1 (canonical) de '
  'load_style() conforme ADR 0002.';

-- ============================================================================
-- 2. sf_customers — referenced by app/session/manager.py:load_session
--
-- Created in prod by `20260414_sf_customers` (2026-04-14). The repo
-- references this table from `manager.py:54` (upsert) but never declared it.
-- Declared here so fresh dev clones can run end-to-end.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.sf_customers (
  id                      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  clinic_id               UUID        NOT NULL REFERENCES public.sf_clinics(id) ON DELETE CASCADE,
  phone                   TEXT        NOT NULL,
  full_name               TEXT,
  first_attribution_id    UUID,
  created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT sf_customers_clinic_phone_unique UNIQUE (clinic_id, phone)
);

CREATE INDEX IF NOT EXISTS idx_sf_customers_clinic_phone
  ON public.sf_customers (clinic_id, phone);

COMMENT ON TABLE public.sf_customers IS
  'Pacientes/leads da clínica indexados por (clinic_id, phone). Upsert em '
  'app/session/manager.py:load_session.';

-- ============================================================================
-- NOT INCLUDED IN THIS SNAPSHOT (and why):
--
-- - sf_agent_profiles
--     Per ADR 0002 it is a dashboard cache, not a Sofia runtime read tier.
--     Adding it here would create false ownership; the dashboard repo
--     owns it.
--
-- - la_blueprints schema beyond `clinic_id`
--     Owned by the Legacy Analyzer repo. The production blueprint_json
--     uses g1/g2/g3/g4/g5/g6 keys, not the legacy
--     shadow_dna_profile/agent_identity/conversational_flow keys still
--     used in the seed payload of migration 014. The reader projection
--     lives in ADR 0002 and the implementation lands in C7/C8.
--
-- - sf_resources, sf_resource_*, sf_specialties, sf_health_insurances,
--   sf_clinic_payment_config, sf_clinic_ad_accounts, sf_payment_transactions,
--   sf_onboarding_events, sf_ad_clicks, sf_ad_spend, sf_campaign_proposals,
--   sf_campaign_audit_log
--     All created and read by the Dashboard/Onboarding repo. Sofia does
--     not read them today. They will be backfilled in their owning repo.
--
-- - sf_clinics drift columns (auth_user_id, name, website_url, utm_params,
--   onboarding_step, onboarding_status, onboarding_review, plan,
--   structured_address, …)
--     Owned by the Onboarding/Dashboard repo. Sofia reads `sf_clinics` only
--     for the columns originally declared here. Adding the dashboard
--     columns would split ownership.
--
-- - sf_clinic_services pricing/wizard columns
--     Owned by the Dashboard repo.
--
-- - GK, SDR/Sourcing, LA, Onboarding tables
--     Owned by their respective repos.
-- ============================================================================

COMMIT;
