-- Migration 014: Seed Clínica Vitória (Iris greeting smoke)
--
-- Cria a clínica usada como fixture do smoke end-to-end da Iris:
--   1. sf_clinics                      → row da clínica
--   2. sf_clinic_profiles              → assistant_name='Iris', avg_ticket, address
--   3. la_analysis_jobs (synthetic)    → job manual marcado como done para satisfazer FK NOT NULL
--   4. la_blueprints                   → blueprint_json mínimo no schema esperado por
--                                        app/session/manager.py:load_style
--
-- Idempotente: usa NOT EXISTS / WHERE NOT EXISTS para permitir replay seguro.
-- Não altera nenhuma tabela; apenas INSERT — sem risco para dados de Sofia em prod
-- (Lumina/Sgen/demais clínicas).
--
-- Schema do blueprint_json alinhado com EASAA-23 / app/session/manager.py:load_style:
--   shadow_dna_profile / agent_identity / conversational_flow.
-- (Difere do schema "g1_identidade/g2_tom_voz/..." gerado pelo Legacy Analyzer em prod;
-- esta seed é manual para o MVP da Iris e será substituída por um Blueprint real
-- quando rodarmos o LA na Vitória.)

DO $$
DECLARE
  v_clinic_id   UUID;
  v_clinic_name TEXT := 'Clínica Vitória';
  v_job_id      UUID;
  v_blueprint   JSONB := jsonb_build_object(
    'shadow_dna_profile', jsonb_build_object(
      'tone_classification', 'Acolhedor profissional',
      'average_response_length_tokens', 80,
      'common_objections', '[]'::jsonb
    ),
    'agent_identity', jsonb_build_object(
      'personality_traits', jsonb_build_array('empática', 'objetiva'),
      'forbidden_terms', '[]'::jsonb
    ),
    'conversational_flow', jsonb_build_object(
      'greeting_style', jsonb_build_object(
        'example', 'Olá! Que bom ter você aqui na Clínica Vitória 💖. Como posso te ajudar?'
      ),
      'closing_style', jsonb_build_object(
        'example', 'Obrigada pelo contato!'
      ),
      'attendance_flow', '[]'::jsonb
    ),
    'metadata', jsonb_build_object(
      'source', 'manual_seed',
      'seeded_by_migration', '014_seed_clinica_vitoria',
      'clinic_name', 'Clínica Vitória'
    )
  );
BEGIN

  -- 1. sf_clinics
  SELECT id INTO v_clinic_id
  FROM public.sf_clinics
  WHERE name = v_clinic_name
  LIMIT 1;

  IF v_clinic_id IS NULL THEN
    INSERT INTO public.sf_clinics (id, name)
    VALUES (gen_random_uuid(), v_clinic_name)
    RETURNING id INTO v_clinic_id;
  END IF;

  RAISE NOTICE 'clinic_id Vitória: %', v_clinic_id;

  -- 2. sf_clinic_profiles
  INSERT INTO public.sf_clinic_profiles (
    id, clinic_id, clinic_name, assistant_name, avg_ticket, address
  )
  SELECT
    uuid_generate_v4(), v_clinic_id, v_clinic_name, 'Iris', 1500,
    'Av. Brasil, 100 — Centro, São Paulo/SP'
  WHERE NOT EXISTS (
    SELECT 1 FROM public.sf_clinic_profiles WHERE clinic_id = v_clinic_id
  );

  -- 3. la_analysis_jobs sintético (necessário porque la_blueprints.job_id é NOT NULL + UNIQUE + FK).
  --    Marcado como done para refletir que o blueprint já está pronto.
  SELECT j.id INTO v_job_id
  FROM public.la_analysis_jobs j
  JOIN public.la_blueprints bp ON bp.job_id = j.id
  WHERE j.clinic_id = v_clinic_id
    AND bp.blueprint_json -> 'metadata' ->> 'seeded_by_migration' = '014_seed_clinica_vitoria'
  LIMIT 1;

  IF v_job_id IS NULL THEN
    INSERT INTO public.la_analysis_jobs (
      id, clinic_id, status, progress,
      total_conversations, processed_conversations,
      original_filename
    )
    VALUES (
      gen_random_uuid(), v_clinic_id, 'done', 100,
      0, 0,
      'manual-seed-clinica-vitoria.txt'
    )
    RETURNING id INTO v_job_id;

    -- 4. la_blueprints (insere apenas quando criamos o job sintético; idempotência via job UNIQUE).
    INSERT INTO public.la_blueprints (
      id, job_id, clinic_id, blueprint_json, knowledge_base_mapping
    )
    VALUES (
      gen_random_uuid(), v_job_id, v_clinic_id, v_blueprint, '{}'::jsonb
    );
  END IF;

  RAISE NOTICE 'la_analysis_jobs sintético: %', v_job_id;

END $$;

-- Verificação pós-migration: confirma a fixture e expõe o clinic_id para
-- ser usado em sf_instance_clinic_map (entrará em C7 quando o instance_name da
-- Evolution Iris-Vitória for confirmado).
SELECT
  c.id   AS clinic_id,
  c.name AS clinic_name,
  cp.assistant_name,
  cp.avg_ticket,
  cp.address,
  bp.blueprint_json -> 'conversational_flow' -> 'greeting_style' ->> 'example' AS greeting_example,
  bp.blueprint_json -> 'shadow_dna_profile' ->> 'tone_classification'           AS tone
FROM public.sf_clinics c
LEFT JOIN public.sf_clinic_profiles cp ON cp.clinic_id = c.id
LEFT JOIN public.la_blueprints bp      ON bp.clinic_id = c.id
WHERE c.name = 'Clínica Vitória';
