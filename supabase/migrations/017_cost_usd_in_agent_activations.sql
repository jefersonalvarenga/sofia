-- Migration 017 — cost_usd column in sf_agent_activations
--
-- Stores the Anthropic-priced USD cost of each agent activation, computed
-- from prompt_tokens + completion_tokens by app/core/pricing.py.
--
-- NUMERIC(10, 6) holds up to 9999.999999 USD per activation, well above
-- any plausible single-call cost.
--
-- Default 0 covers historical rows and deterministic agents (no LLM call).
--
-- Idempotent. Note: the planning issue ([EASAA-25](../../../EASAA/issues/EASAA-25))
-- referenced this migration as 016, but 016 was claimed by the schema-drift
-- audit (seed_vitoria_assistant_profile, ADR 0002). Renumbered to 017.
-- C11 ([EASAA-32](../../../EASAA/issues/EASAA-32)) views migration is
-- consequently renumbered 017 → 018.
--
-- Related: docs/adr/0002-dna-canonical.md.

ALTER TABLE public.sf_agent_activations
  ADD COLUMN IF NOT EXISTS cost_usd NUMERIC(10, 6) NOT NULL DEFAULT 0;

-- Index supports the cost-per-clinic dashboard rollups planned in C11.
CREATE INDEX IF NOT EXISTS idx_sf_activations_cost_clinic
  ON public.sf_agent_activations (clinic_id, created_at)
  WHERE cost_usd > 0;
