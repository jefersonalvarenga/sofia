-- Migration 004: Add clinic_name to clinic_profiles + seed config for clinica-sgen
-- la_client: slug=sgen, name='Sorriso Da Gente', id=4569fed6-7b4a-41f1-918e-ffb1e5ba0403
--
-- IMPORTANT: After running this migration, capture the clinic_id from the SELECT at the
-- bottom and configure it as clinic_id in the n8n workflow for clinica-sgen.

-- 1. Add clinic_name column to clinic_profiles (fixes manager.py bug)
ALTER TABLE public.clinic_profiles
  ADD COLUMN IF NOT EXISTS clinic_name TEXT;

-- 2. Seed all clinic tables for sgen
DO $$
DECLARE
  v_clinic_id     UUID;
  v_la_client_id  UUID := '4569fed6-7b4a-41f1-918e-ffb1e5ba0403';
  v_clinic_name   TEXT := 'Sorriso Da Gente';
BEGIN

  -- 3. Insert clinics record (FK parent for all clinic_* tables) if not exists
  SELECT id INTO v_clinic_id FROM public.clinics WHERE name = v_clinic_name LIMIT 1;

  IF v_clinic_id IS NULL THEN
    INSERT INTO public.clinics (id, name)
    VALUES (gen_random_uuid(), v_clinic_name)
    RETURNING id INTO v_clinic_id;
  END IF;

  RAISE NOTICE 'clinic_id for sgen: %', v_clinic_id;

  -- 4. Upsert clinic_profiles
  INSERT INTO public.clinic_profiles (id, clinic_id, clinic_name, assistant_name, avg_ticket, address)
  SELECT gen_random_uuid(), v_clinic_id, v_clinic_name, 'Sofia', 0, ''
  WHERE NOT EXISTS (
    SELECT 1 FROM public.clinic_profiles WHERE clinic_id = v_clinic_id
  );

  -- Update clinic_name if row already existed (e.g. re-run)
  UPDATE public.clinic_profiles
  SET clinic_name = v_clinic_name
  WHERE clinic_id = v_clinic_id AND clinic_name IS NULL;

  -- 5. Seed clinic_services from la_chat_analyses topics
  --    topics column is stored as a JSONB string ('[\"topic1\", \"topic2\"]')
  --    so we cast: (ca.topics #>> '{}')::jsonb to get the actual array
  INSERT INTO public.clinic_services (id, clinic_id, name)
  SELECT
    gen_random_uuid(),
    v_clinic_id,
    trim(topic_str)
  FROM (
    SELECT DISTINCT jsonb_array_elements_text(
      (ca.topics #>> '{}')::jsonb
    ) AS topic_str
    FROM public.la_chat_analyses ca
    WHERE ca.client_id = v_la_client_id
      AND ca.topics IS NOT NULL
  ) t
  WHERE trim(topic_str) <> ''
  ON CONFLICT DO NOTHING;

  -- 6. Seed clinic_business_rules with generic dental-clinic rules
  INSERT INTO public.clinic_business_rules (id, clinic_id, rule_type, content)
  VALUES
    (gen_random_uuid(), v_clinic_id, 'horario_atendimento', 'Segunda a sexta das 9h às 18h'),
    (gen_random_uuid(), v_clinic_id, 'agendamento',         'Agendamentos feitos pelo WhatsApp ou telefone'),
    (gen_random_uuid(), v_clinic_id, 'convenio',            'Não aceitamos convênios, somente pagamento particular')
  ON CONFLICT DO NOTHING;

END $$;

-- Verification: run this after migration to confirm seeded data and capture clinic_id
SELECT
  cp.clinic_id,
  cp.clinic_name,
  cp.assistant_name,
  cp.avg_ticket,
  (SELECT COUNT(*) FROM public.clinic_services cs WHERE cs.clinic_id = cp.clinic_id) AS services_count,
  (SELECT COUNT(*) FROM public.clinic_business_rules cbr WHERE cbr.clinic_id = cp.clinic_id) AS rules_count
FROM public.clinic_profiles cp
WHERE cp.clinic_name = 'Sorriso Da Gente';
