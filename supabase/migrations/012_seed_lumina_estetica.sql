-- Migration 012: Seed clínica Lumina Estética Avançada (ICP real para testes)
--
-- Clínica de estética premium, 100% particular, ticket médio R$1.800
-- Assistente: Aline | Endereço: Av. Paulista, 1000 — São Paulo/SP
--
-- IMPORTANT: After running this migration, capture the clinic_id from the SELECT at the
-- bottom and configure it as clinic_id in the n8n workflow for clinica-lumina.

DO $$
DECLARE
  v_clinic_id   UUID;
  v_clinic_name TEXT := 'Lumina Estética Avançada';
BEGIN

  -- 1. Insert sf_clinics record if not exists
  SELECT id INTO v_clinic_id FROM public.sf_clinics WHERE name = v_clinic_name LIMIT 1;

  IF v_clinic_id IS NULL THEN
    INSERT INTO public.sf_clinics (id, name)
    VALUES (gen_random_uuid(), v_clinic_name)
    RETURNING id INTO v_clinic_id;
  END IF;

  RAISE NOTICE 'clinic_id for lumina: %', v_clinic_id;

  -- 2. Upsert sf_clinic_profiles
  INSERT INTO public.sf_clinic_profiles (id, clinic_id, clinic_name, assistant_name, avg_ticket, address)
  SELECT gen_random_uuid(), v_clinic_id, v_clinic_name, 'Aline', 1800, 'Av. Paulista, 1000 — Bela Vista, São Paulo/SP'
  WHERE NOT EXISTS (
    SELECT 1 FROM public.sf_clinic_profiles WHERE clinic_id = v_clinic_id
  );

  -- 3. Seed sf_clinic_services
  INSERT INTO public.sf_clinic_services (id, clinic_id, name, price)
  VALUES
    (gen_random_uuid(), v_clinic_id, 'Botox (1 área)',                  600.00),
    (gen_random_uuid(), v_clinic_id, 'Botox (3 áreas)',                1400.00),
    (gen_random_uuid(), v_clinic_id, 'Preenchimento labial',           1200.00),
    (gen_random_uuid(), v_clinic_id, 'Preenchimento malar',            1500.00),
    (gen_random_uuid(), v_clinic_id, 'Bioestimulador de colágeno',     2200.00),
    (gen_random_uuid(), v_clinic_id, 'Lipo enzimática (flanco)',       1800.00),
    (gen_random_uuid(), v_clinic_id, 'Fio de PDO',                    2500.00),
    (gen_random_uuid(), v_clinic_id, 'Peeling químico',                450.00),
    (gen_random_uuid(), v_clinic_id, 'Skinbooster',                    900.00),
    (gen_random_uuid(), v_clinic_id, 'Microagulhamento',               350.00),
    (gen_random_uuid(), v_clinic_id, 'Harmonização facial completa',  4500.00),
    (gen_random_uuid(), v_clinic_id, 'Rinomodelação',                 1800.00),
    (gen_random_uuid(), v_clinic_id, 'Avaliação facial (gratuita)',       0.00)
  ON CONFLICT DO NOTHING;

  -- 4. Seed sf_clinic_business_rules
  INSERT INTO public.sf_clinic_business_rules (id, clinic_id, rule_type, content)
  VALUES
    (gen_random_uuid(), v_clinic_id, 'convenio',
      'Não trabalhamos com convênios. Atendimento 100% particular.'),
    (gen_random_uuid(), v_clinic_id, 'pagamento',
      'Parcelamos em até 18x no cartão. Aceitamos PIX com 5% de desconto.'),
    (gen_random_uuid(), v_clinic_id, 'agendamento',
      'Agendamentos pelo WhatsApp. Primeiro passo: avaliação facial gratuita e sem compromisso.'),
    (gen_random_uuid(), v_clinic_id, 'horario_atendimento',
      'Segunda a sábado das 9h às 19h.'),
    (gen_random_uuid(), v_clinic_id, 'origem_lead',
      'Recebemos pacientes principalmente via anúncios no Instagram e Google.'),
    (gen_random_uuid(), v_clinic_id, 'garantia',
      'Retoque gratuito em até 15 dias após o procedimento.'),
    (gen_random_uuid(), v_clinic_id, 'promocao',
      'Promoções relâmpago divulgadas no Instagram @lumina.estetica.')
  ON CONFLICT DO NOTHING;

END $$;

-- Verification: run this after migration to confirm seeded data and capture clinic_id
SELECT
  cp.clinic_id,
  cp.clinic_name,
  cp.assistant_name,
  cp.avg_ticket,
  cp.address,
  (SELECT COUNT(*) FROM public.sf_clinic_services cs WHERE cs.clinic_id = cp.clinic_id) AS services_count,
  (SELECT COUNT(*) FROM public.sf_clinic_business_rules cbr WHERE cbr.clinic_id = cp.clinic_id) AS rules_count
FROM public.sf_clinic_profiles cp
WHERE cp.clinic_name = 'Lumina Estética Avançada';
