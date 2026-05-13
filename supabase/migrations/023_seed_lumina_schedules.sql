-- Migration 017: seed Lumina Estética — Agenda Geral weekly schedule pattern

-- Ensure unique constraint exists for idempotent seed
ALTER TABLE public.sf_resource_schedules
  ADD CONSTRAINT IF NOT EXISTS sf_resource_schedules_resource_dow_start_key
  UNIQUE (resource_id, day_of_week, start_time);

DO $$
DECLARE
  v_resource_id UUID;
BEGIN
  SELECT id INTO v_resource_id
  FROM public.sf_resources
  WHERE clinic_id = '0d6d8eaf-6efa-4aaf-9845-de4b0d0f608c'
    AND name = 'Agenda Geral'
  LIMIT 1;

  IF v_resource_id IS NULL THEN
    RAISE NOTICE 'Agenda Geral resource not found for Lumina — seed skipped';
    RETURN;
  END IF;

  INSERT INTO public.sf_resource_schedules
    (resource_id, day_of_week, start_time, end_time, valid_from, valid_until)
  VALUES
    (v_resource_id, 0, '09:00', '12:00', NULL, NULL),
    (v_resource_id, 0, '15:00', '19:00', NULL, NULL),
    (v_resource_id, 1, '14:00', '17:00', NULL, NULL),
    (v_resource_id, 4, '09:00', '12:00', NULL, NULL)
  ON CONFLICT (resource_id, day_of_week, start_time) DO NOTHING;
END $$;
