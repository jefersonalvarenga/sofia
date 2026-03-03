-- Migration 009: Sofia 2.0 — observability columns + conversation tracking
-- Adds token usage, trace_id, language, clinic_id to sf_agent_activations
-- Adds sf_conversations table for billable conversation tracking

-- ============================================================
-- 1. sf_agent_activations — observability extensions
-- ============================================================

ALTER TABLE sf_agent_activations
  ADD COLUMN IF NOT EXISTS prompt_tokens     INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS completion_tokens INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS total_tokens      INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS trace_id          TEXT,
  ADD COLUMN IF NOT EXISTS language          TEXT NOT NULL DEFAULT 'pt-BR',
  ADD COLUMN IF NOT EXISTS clinic_id         UUID,
  ADD COLUMN IF NOT EXISTS messages          JSONB,
  ADD COLUMN IF NOT EXISTS data              JSONB,
  ADD COLUMN IF NOT EXISTS started_at        TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS duration_ms       NUMERIC(10, 2);

-- Index for dashboard queries: cost per clinic per month
CREATE INDEX IF NOT EXISTS idx_agent_activations_clinic_id
  ON sf_agent_activations (clinic_id);

-- Index for trace correlation
CREATE INDEX IF NOT EXISTS idx_agent_activations_trace_id
  ON sf_agent_activations (trace_id);

-- ============================================================
-- 2. sf_conversations — billable conversation tracking
-- ============================================================

CREATE TABLE IF NOT EXISTS sf_conversations (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  clinic_id         UUID NOT NULL,
  session_id        TEXT NOT NULL,
  trace_id          TEXT,
  conversation_type TEXT NOT NULL DEFAULT 'first_contact',
  -- conversation_type: first_contact | confirmation | reminder | reengage | upsell
  outcome           TEXT,
  -- outcome: booked | dropped | confirmed | ignored | reactivated | lost | accepted | declined
  language          TEXT NOT NULL DEFAULT 'pt-BR',
  total_tokens      INTEGER NOT NULL DEFAULT 0,
  started_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ended_at          TIMESTAMPTZ,
  billed            BOOLEAN NOT NULL DEFAULT FALSE,
  billed_at         TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_sf_conversations_clinic_id
  ON sf_conversations (clinic_id);

CREATE INDEX IF NOT EXISTS idx_sf_conversations_session_id
  ON sf_conversations (session_id);

CREATE INDEX IF NOT EXISTS idx_sf_conversations_type_outcome
  ON sf_conversations (conversation_type, outcome);

-- ============================================================
-- 3. sf_appointments — payment tracking columns
-- ============================================================

ALTER TABLE sf_appointments
  ADD COLUMN IF NOT EXISTS payment_status TEXT NOT NULL DEFAULT 'unpaid',
  -- payment_status: unpaid | pending_pix | paid
  ADD COLUMN IF NOT EXISTS payment_tx_id  UUID;

-- ============================================================
-- 4. sf_clinic_payment_config — PIX in-conversation config
-- ============================================================

CREATE TABLE IF NOT EXISTS sf_clinic_payment_config (
  clinic_id            UUID PRIMARY KEY,
  pix_discount_pct     INTEGER NOT NULL DEFAULT 0,
  pix_expiry_minutes   INTEGER NOT NULL DEFAULT 30,
  asaas_subconta_id    TEXT,
  enabled              BOOLEAN NOT NULL DEFAULT FALSE,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 5. sf_payment_transactions — patient PIX payments
-- ============================================================

CREATE TABLE IF NOT EXISTS sf_payment_transactions (
  id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  clinic_id            UUID NOT NULL,
  session_id           TEXT NOT NULL,
  appointment_id       UUID,
  original_price_brl   NUMERIC(10, 2) NOT NULL,
  discount_pct         INTEGER NOT NULL DEFAULT 0,
  final_price_brl      NUMERIC(10, 2) NOT NULL,
  asaas_charge_id      TEXT,
  status               TEXT NOT NULL DEFAULT 'offered',
  -- status: offered | paid | expired | cancelled
  paid_at              TIMESTAMPTZ,
  expires_at           TIMESTAMPTZ,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sf_payment_transactions_clinic_id
  ON sf_payment_transactions (clinic_id);

CREATE INDEX IF NOT EXISTS idx_sf_payment_transactions_appointment_id
  ON sf_payment_transactions (appointment_id);
