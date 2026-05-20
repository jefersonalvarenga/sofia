-- Migration 027 — sf_intake_questions
--
-- Source-of-truth table for clinical intake questions used by the
-- SCHEDULE_INTAKE sub-agent.
--
-- Spec: kb/07-MVP/Tech/03-Discussoes/schedule/01 - Spec SCHEDULE_INTAKE.md §4.1
--
-- Rows can be either:
--   * clinic baseline   (service_id IS NULL) — applies to every service
--   * service override  (service_id = X)     — appended on top of baseline
--
-- The agent merges them via UNION ordered by "order".
-- Idempotent: re-running this migration is safe.

BEGIN;

CREATE TABLE IF NOT EXISTS public.sf_intake_questions (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    clinic_id     uuid NOT NULL REFERENCES public.sf_clinics(id) ON DELETE CASCADE,
    service_id    uuid NULL REFERENCES public.sf_clinic_services(id) ON DELETE CASCADE,
    "order"       int NOT NULL,
    question_text text NOT NULL,
    category      text NOT NULL,
    is_required   boolean NOT NULL DEFAULT true,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),

    -- Unique order *within scope* (baseline OR a single service).
    CONSTRAINT sf_intake_questions_order_unique
        UNIQUE (clinic_id, service_id, "order")
);

CREATE INDEX IF NOT EXISTS idx_sf_intake_questions_clinic
    ON public.sf_intake_questions(clinic_id)
    WHERE service_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_sf_intake_questions_service
    ON public.sf_intake_questions(clinic_id, service_id)
    WHERE service_id IS NOT NULL;

-- RLS disabled — same convention as the rest of the sf_* tables
-- (access enforced at the application layer with the service-role key).
ALTER TABLE public.sf_intake_questions DISABLE ROW LEVEL SECURITY;

COMMENT ON TABLE public.sf_intake_questions IS
  'Clinical intake questions used by SCHEDULE_INTAKE sub-agent. service_id NULL = clinic baseline; service_id NOT NULL = override per service. Union loaded by app/repositories/intake_questions.py.';

COMMENT ON COLUMN public.sf_intake_questions.category IS
  'Free-text category (medicamentos, alergias, gestacao, cronicas, pele, custom, ...). No enum constraint in MVP.';

COMMIT;
