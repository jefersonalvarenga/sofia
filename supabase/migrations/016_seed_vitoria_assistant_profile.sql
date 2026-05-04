-- Migration 016 — seed sf_assistant_profile for Clínica Vitória
--
-- Per ADR 0002 (canonical DNA), `sf_assistant_profile` is the tier-1 source
-- read by `load_style()`. Migration 014 only seeded the legacy
-- `la_blueprints` shape for Vitória; the Iris greeting smoke
-- ([EASAA-23](../../../EASAA/issues/EASAA-23)) will read tier 1 once C7/C8
-- (EASAA-28 / EASAA-29) implement the new lookup.
--
-- This migration adds the canonical row so the smoke exercises the real
-- code path, not the fallback.
--
-- Idempotent: UPSERT on (clinic_id) which is UNIQUE in sf_assistant_profile.
-- Depends on:
--   - 014_seed_clinica_vitoria.sql      (creates the sf_clinics row)
--   - 015_drift_snapshot_2026_05_04.sql (creates the sf_assistant_profile table)
--
-- Related: docs/adr/0002-dna-canonical.md, EASAA-36, EASAA-23.

DO $$
DECLARE
  v_clinic_id   UUID;
  v_clinic_name TEXT := 'Clínica Vitória';
BEGIN

  SELECT id INTO v_clinic_id
  FROM public.sf_clinics
  WHERE name = v_clinic_name
  LIMIT 1;

  IF v_clinic_id IS NULL THEN
    RAISE EXCEPTION 'Clínica Vitória not found — run 014_seed_clinica_vitoria first';
  END IF;

  INSERT INTO public.sf_assistant_profile (
    clinic_id,
    tom_voz,
    nivel_formalidade,
    uso_emoji_frequencia,
    uso_emoji_tipos,
    comprimento_msg_tipico,
    quebra_de_msg,
    saudacao_inicial,
    despedida_padrao,
    politica_preco,
    momento_revela_preco,
    educacao_tecnica,
    qualificacao_tipica,
    prova_social_uso,
    mencao_profissional,
    politica_sinal,
    objecoes_recorrentes,
    contraindicacao_policy,
    fluxo_padrao_atendimento,
    como_confirma_agendamento,
    follow_up_apos_silencio,
    faq_extraido,
    procedimentos_explicados,
    casos_de_escalation
  )
  VALUES (
    v_clinic_id,
    'cordial_amigavel',
    'voce',
    'media',
    jsonb_build_array('💖', '😊', '✨'),
    'mix',
    'algumas_mensagens_curtas',
    jsonb_build_array(
      'Olá! Que bom ter você aqui na Clínica Vitória 💖. Como posso te ajudar?',
      'Oi! Bem-vinda à Clínica Vitória ✨ Em que posso ajudar hoje?'
    ),
    jsonb_build_array(
      'Obrigada pelo contato! 💖',
      'Qualquer dúvida estou por aqui 😊'
    ),
    'mix',
    'apos_qualificacao',
    'media',
    jsonb_build_array('procedimento_de_interesse', 'horario_preferido', 'primeira_vez_no_local'),
    'moderada',
    'sob_demanda',
    jsonb_build_object('valor_padrao', 100, 'forma_pagamento', 'pix', 'reembolsavel', false),
    jsonb_build_array(),
    jsonb_build_object(),
    jsonb_build_array('greeting', 'qualificacao', 'apresenta_servico_e_preco', 'verifica_disponibilidade_agenda', 'confirma_agendamento'),
    'Agendado então: {data} às {hora} para {servico}. Te esperamos na Av. Brasil, 100 — Centro!',
    jsonb_build_object('intervalo_horas', 24, 'tenta_quantas_vezes', 1, 'tom', 'cordial_amigavel'),
    jsonb_build_array(),
    jsonb_build_array(),
    jsonb_build_array()
  )
  ON CONFLICT (clinic_id) DO UPDATE SET
    tom_voz                   = EXCLUDED.tom_voz,
    nivel_formalidade         = EXCLUDED.nivel_formalidade,
    uso_emoji_frequencia      = EXCLUDED.uso_emoji_frequencia,
    uso_emoji_tipos           = EXCLUDED.uso_emoji_tipos,
    comprimento_msg_tipico    = EXCLUDED.comprimento_msg_tipico,
    quebra_de_msg             = EXCLUDED.quebra_de_msg,
    saudacao_inicial          = EXCLUDED.saudacao_inicial,
    despedida_padrao          = EXCLUDED.despedida_padrao,
    politica_preco            = EXCLUDED.politica_preco,
    momento_revela_preco      = EXCLUDED.momento_revela_preco,
    educacao_tecnica          = EXCLUDED.educacao_tecnica,
    qualificacao_tipica       = EXCLUDED.qualificacao_tipica,
    prova_social_uso          = EXCLUDED.prova_social_uso,
    mencao_profissional       = EXCLUDED.mencao_profissional,
    politica_sinal            = EXCLUDED.politica_sinal,
    objecoes_recorrentes      = EXCLUDED.objecoes_recorrentes,
    contraindicacao_policy    = EXCLUDED.contraindicacao_policy,
    fluxo_padrao_atendimento  = EXCLUDED.fluxo_padrao_atendimento,
    como_confirma_agendamento = EXCLUDED.como_confirma_agendamento,
    follow_up_apos_silencio   = EXCLUDED.follow_up_apos_silencio,
    updated_at                = now();

  RAISE NOTICE 'sf_assistant_profile seeded for Vitória: clinic_id=%', v_clinic_id;

END $$;

-- Verification: confirm the canonical row is in place and surface what
-- load_style() will project as the tier-1 greeting/closing/tone for Iris.
SELECT
  c.id    AS clinic_id,
  c.name  AS clinic_name,
  ap.tom_voz,
  ap.nivel_formalidade,
  ap.saudacao_inicial -> 0           AS first_greeting,
  ap.despedida_padrao -> 0           AS first_closing,
  ap.fluxo_padrao_atendimento        AS attendance_flow
FROM public.sf_clinics c
LEFT JOIN public.sf_assistant_profile ap ON ap.clinic_id = c.id
WHERE c.name = 'Clínica Vitória';
