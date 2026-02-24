-- Migration 005: Mapeamento instanceName (Evolution API) → clinic_id (Sofia)
-- Necessário porque public."Instance" é Prisma-managed e não pode receber colunas extras.
-- Para adicionar uma nova clínica: INSERT INTO instance_clinic_map (instance_name, clinic_id) VALUES (...).

CREATE TABLE IF NOT EXISTS public.instance_clinic_map (
  instance_name TEXT PRIMARY KEY,
  clinic_id     UUID NOT NULL REFERENCES public.clinics(id),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Seed: clinica-sgen → Sorriso Da Gente (clinic_id gerado pela migration 004)
INSERT INTO public.instance_clinic_map (instance_name, clinic_id)
VALUES ('clinica-sgen', 'a4a04b17-0158-48b2-b4e3-1175825c84c4')
ON CONFLICT (instance_name) DO UPDATE SET clinic_id = EXCLUDED.clinic_id;
