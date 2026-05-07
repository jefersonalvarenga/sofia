-- Migration 017 — sf_messages + RLS por tenant
--
-- Iris C3 ([EASAA-24](../../../EASAA/issues/EASAA-24)).
--
-- Cria a tabela `sf_messages` que serve como guardião atômico de
-- idempotência para o webhook Evolution → FastAPI da Iris (C7,
-- [EASAA-28](../../../EASAA/issues/EASAA-28)). UNIQUE (clinic_id, wamid)
-- permite `INSERT ... ON CONFLICT DO NOTHING RETURNING` — se a linha já
-- existe, a Iris responde 200 sem reprocessar.
--
-- Também habilita RLS nas tabelas tenant-scoped do core. Política única
-- por tabela: `clinic_id = current_setting('app.current_tenant', true)::uuid`.
-- O segundo argumento `true` faz `current_setting` devolver NULL quando a
-- variável não está setada — assim a query bloqueia (zero rows) em vez de
-- estourar `unrecognized configuration parameter`.
--
-- Importante: o backend FastAPI usa `service_role`, que **bypassa** RLS.
-- A defesa real continua sendo `clinic_id` injetado em toda query no
-- código (padrão já estabelecido no Sofia). RLS aqui é cinto+suspensório:
-- protege caso um dia abramos endpoints com JWT/anon. Ver
-- [ADR 0001 — Iris tenant isolation](../../docs/adr/0001-iris-tenant-isolation.md)
-- (a ser criado em C12, [EASAA-33](../../../EASAA/issues/EASAA-33)).

BEGIN;

-- ============================================================================
-- 1. sf_messages
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.sf_messages (
  id            UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
  clinic_id     UUID        NOT NULL REFERENCES public.sf_clinics(id) ON DELETE CASCADE,
  session_id    TEXT        REFERENCES public.sf_sessions(session_id) ON DELETE SET NULL,
  wamid         TEXT        NOT NULL,
  direction     TEXT        NOT NULL CHECK (direction IN ('inbound','outbound')),
  content       TEXT,
  message_type  TEXT        NOT NULL DEFAULT 'text',
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT sf_messages_clinic_wamid_unique UNIQUE (clinic_id, wamid)
);

CREATE INDEX IF NOT EXISTS idx_sf_messages_session_created
  ON public.sf_messages (clinic_id, session_id, created_at DESC);

COMMENT ON TABLE  public.sf_messages IS
  'Iris message log. Idempotência via UNIQUE (clinic_id, wamid). sf_sessions.history continua como denormalized read view.';
COMMENT ON COLUMN public.sf_messages.wamid IS
  'WhatsApp message id (Evolution payload). Único por clínica.';
COMMENT ON COLUMN public.sf_messages.direction IS
  '`inbound` = paciente -> Iris. `outbound` = Iris -> paciente.';

-- ============================================================================
-- 2. RLS: enable + policy por tabela
--
-- Política única (FOR ALL) por tabela. service_role bypassa.
-- ============================================================================

-- sf_clinics — usa id (não clinic_id) como tenant key.
ALTER TABLE public.sf_clinics ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS sf_clinics_tenant_isolation ON public.sf_clinics;
CREATE POLICY sf_clinics_tenant_isolation ON public.sf_clinics
  FOR ALL TO PUBLIC
  USING (id = current_setting('app.current_tenant', true)::uuid)
  WITH CHECK (id = current_setting('app.current_tenant', true)::uuid);

-- sf_sessions
ALTER TABLE public.sf_sessions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS sf_sessions_tenant_isolation ON public.sf_sessions;
CREATE POLICY sf_sessions_tenant_isolation ON public.sf_sessions
  FOR ALL TO PUBLIC
  USING (clinic_id = current_setting('app.current_tenant', true)::uuid)
  WITH CHECK (clinic_id = current_setting('app.current_tenant', true)::uuid);

-- sf_messages
ALTER TABLE public.sf_messages ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS sf_messages_tenant_isolation ON public.sf_messages;
CREATE POLICY sf_messages_tenant_isolation ON public.sf_messages
  FOR ALL TO PUBLIC
  USING (clinic_id = current_setting('app.current_tenant', true)::uuid)
  WITH CHECK (clinic_id = current_setting('app.current_tenant', true)::uuid);

-- sf_agent_activations
ALTER TABLE public.sf_agent_activations ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS sf_agent_activations_tenant_isolation ON public.sf_agent_activations;
CREATE POLICY sf_agent_activations_tenant_isolation ON public.sf_agent_activations
  FOR ALL TO PUBLIC
  USING (clinic_id = current_setting('app.current_tenant', true)::uuid)
  WITH CHECK (clinic_id = current_setting('app.current_tenant', true)::uuid);

-- sf_clinic_profiles
ALTER TABLE public.sf_clinic_profiles ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS sf_clinic_profiles_tenant_isolation ON public.sf_clinic_profiles;
CREATE POLICY sf_clinic_profiles_tenant_isolation ON public.sf_clinic_profiles
  FOR ALL TO PUBLIC
  USING (clinic_id = current_setting('app.current_tenant', true)::uuid)
  WITH CHECK (clinic_id = current_setting('app.current_tenant', true)::uuid);

-- sf_customers (já tem RLS ativada; idempotente)
ALTER TABLE public.sf_customers ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS sf_customers_tenant_isolation ON public.sf_customers;
CREATE POLICY sf_customers_tenant_isolation ON public.sf_customers
  FOR ALL TO PUBLIC
  USING (clinic_id = current_setting('app.current_tenant', true)::uuid)
  WITH CHECK (clinic_id = current_setting('app.current_tenant', true)::uuid);

-- la_blueprints
ALTER TABLE public.la_blueprints ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS la_blueprints_tenant_isolation ON public.la_blueprints;
CREATE POLICY la_blueprints_tenant_isolation ON public.la_blueprints
  FOR ALL TO PUBLIC
  USING (clinic_id = current_setting('app.current_tenant', true)::uuid)
  WITH CHECK (clinic_id = current_setting('app.current_tenant', true)::uuid);

COMMIT;

-- ============================================================================
-- Smoke manual (resultado capturado em 2026-05-07 via Supabase MCP):
--
--   -- 1. Sem setting (role authenticated, no service_role bypass):
--   SET ROLE authenticated;
--   SELECT count(*) FROM public.sf_clinics;       -- 0  ✅
--   SELECT count(*) FROM public.sf_messages;      -- 0  ✅
--   RESET ROLE;
--
--   -- 2. Com setting (clinic_id de Clínica Vitória):
--   BEGIN;
--   SET LOCAL ROLE authenticated;
--   SET LOCAL app.current_tenant = '57952a29-e228-4cac-b5fa-3d20ba478f5d';
--   SELECT count(*) FROM public.sf_clinics;       -- 1  ✅ (só Vitória)
--   SELECT count(*) FROM public.la_blueprints;    -- 1  ✅
--   SELECT count(*) FROM public.sf_clinic_profiles; -- 1 ✅
--   COMMIT;
--
--   -- 3. Idempotência via UNIQUE:
--   INSERT INTO public.sf_messages (clinic_id, wamid, direction, content)
--   VALUES ('<vitoria>','smoke-001','inbound','oi')
--   ON CONFLICT (clinic_id, wamid) DO NOTHING RETURNING id;  -- 1 row
--
--   -- repetir → 0 rows, count(wamid='smoke-001') = 1  ✅
--
-- service_role bypassa RLS (Supabase SQL Editor + supabase-py com chave
-- service_role). Esperado e documentado.
-- ============================================================================
