-- Migration 022: sf_resources — abstração da agenda (profissional, sala, genérica)
-- (originally numbered 014; renumbered to 022 to resolve collision with 014_seed_clinica_vitoria.sql)
-- Modelo OOA: Organization → Unit (sf_clinics) → Resource → Channel
-- Resource resolve QUAL agenda consultar ao buscar slots.

CREATE TABLE public.sf_resources (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clinic_id   UUID NOT NULL REFERENCES public.sf_clinics(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL CHECK (type IN ('professional', 'room', 'generic')),
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Lookup: dado um clinic_id, quais resources ativos existem?
CREATE INDEX sf_resources_clinic_active_idx
    ON public.sf_resources (clinic_id, is_active);

-- RLS desabilitado — controle via API key na camada de aplicacao
ALTER TABLE public.sf_resources DISABLE ROW LEVEL SECURITY;

COMMENT ON TABLE public.sf_resources IS
  'Abstração da agenda da clínica. type=professional para agenda por profissional nomeado,
   type=room para sala de procedimento, type=generic para slot único sem distinção.';

COMMENT ON COLUMN public.sf_resources.type IS
  'professional | room | generic — determina como Sofia apresenta o agendamento ao paciente';

-- Seed: Lumina Estética tem resource genérico por padrão (agenda única)
DO $$
DECLARE
  v_clinic_id UUID := '0d6d8eaf-6efa-4aaf-9845-de4b0d0f608c';
BEGIN
  INSERT INTO public.sf_resources (clinic_id, name, type)
  VALUES (v_clinic_id, 'Agenda Geral', 'generic')
  ON CONFLICT DO NOTHING;
END $$;
