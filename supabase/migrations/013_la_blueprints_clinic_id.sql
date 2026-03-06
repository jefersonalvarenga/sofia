-- Migration 013: add clinic_id to la_blueprints
-- Links Blueprint directly to sf_clinics so Sofia can query by clinic_id

ALTER TABLE public.la_blueprints
  ADD COLUMN IF NOT EXISTS clinic_id UUID REFERENCES public.sf_clinics(id) ON DELETE SET NULL;

-- Index for Sofia's query: SELECT ... WHERE clinic_id = ? ORDER BY created_at DESC LIMIT 1
CREATE INDEX IF NOT EXISTS idx_la_blueprints_clinic_created
  ON public.la_blueprints (clinic_id, created_at DESC);

COMMENT ON COLUMN public.la_blueprints.clinic_id IS
  'FK para sf_clinics. Preenchido pelo Legacy Analyzer ao salvar o Blueprint no Supabase. Sofia usa para carregar o Blueprint mais recente da clínica.';
