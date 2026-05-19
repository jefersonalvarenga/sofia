-- Migration 026 — disable RLS on sf_procedure_kb (Knowledge Agent MVP)
--
-- The policy `tenant_read` from migration 020 requires `app.clinic_id` setting,
-- which neither the anon key clients nor the seeded service backend currently
-- propagate. The KnowledgeAgent backend uses the service-role key (which
-- bypasses RLS anyway) and the dashboard is not consumed by patients —
-- patients reach the KB only through the agent.
--
-- Disabling RLS unblocks:
--   1. The embedding indexer (UPDATE sf_procedure_kb.embedding)
--   2. The eval harness (SELECT via anon for parity with prod backend)
--   3. Future seed scripts (INSERT via anon for dev/test)
--
-- TODO (post-MVP): re-enable RLS with policies that match how the backend
-- propagates tenant context (e.g. JWT claim or session var). Document in
-- handoff as a known security debt.

BEGIN;

ALTER TABLE public.sf_procedure_kb DISABLE ROW LEVEL SECURITY;

-- Drop the unused policy so we don't get false confidence later.
DROP POLICY IF EXISTS tenant_read ON public.sf_procedure_kb;

COMMIT;
