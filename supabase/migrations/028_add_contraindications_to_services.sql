-- Migration 028 — add contraindications to sf_clinic_services
--
-- Adds a `contraindications text[]` column to sf_clinic_services so each
-- clinic can curate a per-service list of contraindication terms. The
-- SCHEDULE_INTAKE sub-agent reads this list and asks the LLM to semantically
-- match each intake answer against it (no Python regex involved).
--
-- Spec: kb/07-MVP/Tech/03-Discussoes/schedule/01 - Spec SCHEDULE_INTAKE.md §5
--
-- Idempotent.

BEGIN;

ALTER TABLE public.sf_clinic_services
    ADD COLUMN IF NOT EXISTS contraindications text[] NOT NULL DEFAULT '{}';

COMMENT ON COLUMN public.sf_clinic_services.contraindications IS
  'Per-service contraindication terms (pt-BR, free text). Read by SCHEDULE_INTAKE agent and semantically matched against patient answers. Empty array = no automated escalation for this service.';

COMMIT;
