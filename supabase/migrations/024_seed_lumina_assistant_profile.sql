-- Migration 024 — seed Lumina assistant profile (Aline persona)
--
-- Migra o seed da Aline da migration 015_sf_agent_profiles.sql original
-- (descartada conforme ADR 0002 / Decisao 2 do vault) para a tabela
-- canonica sf_assistant_profile (criada em 015_drift_snapshot_2026_05_04.sql).
--
-- Mapping de colunas (sf_agent_profiles antiga -> sf_assistant_profile nova):
--   persona_name        -> [sem coluna correspondente; preservado em comment]
--   tone                -> tom_voz
--   personality_traits  -> [sem coluna direta; injetado em qualificacao_tipica como contexto]
--   attendance_flow     -> fluxo_padrao_atendimento (TEXT[] -> JSONB)
--   greeting_example    -> saudacao_inicial[0] (TEXT -> JSONB array)
--   closing_example     -> despedida_padrao[0] (TEXT -> JSONB array)
--   avg_response_tokens -> [sem coluna; e default do projeto Iris, nao da clinica]
--   forbidden_terms     -> contraindicacao_policy.forbidden_terms (TEXT[] -> JSONB nested)
--   common_objections   -> objecoes_recorrentes (TEXT[] -> JSONB array of strings)
--
-- Idempotente: ON CONFLICT (clinic_id) DO NOTHING.
--
-- IMPORTANTE: persona_name "Aline" nao tem destino direto no schema novo.
-- Ate o dashboard de onboarding suportar campo de nome de assistente,
-- a referencia fica neste seed e em comentario na linha do INSERT.

DO $$
DECLARE
  v_clinic_id UUID := '0d6d8eaf-6efa-4aaf-9845-de4b0d0f608c';  -- Lumina Estetica Avancada
BEGIN
  -- persona_name = 'Aline' (preservado para futura coluna assistant_name no schema)
  INSERT INTO public.sf_assistant_profile (
    clinic_id,
    tom_voz,
    saudacao_inicial,
    despedida_padrao,
    fluxo_padrao_atendimento,
    objecoes_recorrentes,
    contraindicacao_policy,
    qualificacao_tipica
  ) VALUES (
    v_clinic_id,
    'Informal e acolhedor',
    '["Oi! Tudo bem? Aqui é a Aline da Lumina Estética. Vi que você entrou em contato — como posso te ajudar hoje? 😊"]'::jsonb,
    '["Perfeito! Agendamento confirmado. Qualquer dúvida pode chamar por aqui. Até lá! 🌟"]'::jsonb,
    '[
      "Saudar e entender o interesse",
      "Apresentar serviço ou tirar dúvida",
      "Oferecer avaliação gratuita",
      "Confirmar agendamento"
    ]'::jsonb,
    '["preço alto", "medo de resultado não natural", "não tem tempo"]'::jsonb,
    '{"forbidden_terms": ["convênio", "SUS", "plano de saúde"]}'::jsonb,
    '["empática", "objetiva", "entusiasta com estética", "não pressiona"]'::jsonb
  )
  ON CONFLICT (clinic_id) DO NOTHING;
END $$;

-- Verification: confirm Lumina assistant profile was seeded
SELECT
  ap.id,
  ap.clinic_id,
  ap.tom_voz,
  ap.saudacao_inicial,
  jsonb_array_length(ap.fluxo_padrao_atendimento) AS flow_steps,
  jsonb_array_length(ap.objecoes_recorrentes)     AS objections_count
FROM public.sf_assistant_profile ap
WHERE ap.clinic_id = '0d6d8eaf-6efa-4aaf-9845-de4b0d0f608c';
