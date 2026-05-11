-- Migration 021 — match_procedure_kb RPC function (EASAA-143)
--
-- Exposes a pgvector cosine-similarity search as a Supabase RPC so the
-- KnowledgeSpecialist can call it via the SDK without raw SQL.
-- Idempotent (CREATE OR REPLACE).

BEGIN;

CREATE OR REPLACE FUNCTION public.match_procedure_kb(
  p_tenant_id  UUID,
  p_embedding  vector(1536),
  p_top_k      INT DEFAULT 4
)
RETURNS TABLE (
  id        UUID,
  procedure TEXT,
  title     TEXT,
  body      TEXT,
  similarity FLOAT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    id,
    procedure,
    title,
    body,
    1 - (embedding <=> p_embedding) AS similarity
  FROM public.sf_procedure_kb
  WHERE tenant_id = p_tenant_id
    AND embedding IS NOT NULL
  ORDER BY embedding <=> p_embedding
  LIMIT p_top_k;
$$;

COMMIT;
