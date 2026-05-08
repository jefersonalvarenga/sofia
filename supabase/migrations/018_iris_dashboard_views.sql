-- Migration 018 — Iris dashboard views
--
-- Cria as views SQL de observabilidade do critério de pronto de EASAA-20:
--   iris_latency_per_node  — latência média e p95 por nó, por hora
--   iris_cost_per_message  — custo USD por conversa (session_id), por clínica
--   iris_error_rate_per_node — taxa de erro por nó, por hora
--
-- Adiciona coluna `status` em sf_agent_activations para simplificar
-- iris_error_rate_per_node. C4 (EASAA-25) não incluiu a coluna para não
-- atrasar o merge; adicionamos aqui com DEFAULT 'success' (zero impacto em
-- rows históricas).
--
-- As views iris_latency_per_node e iris_cost_per_message já existiam em prod
-- de um heartbeat anterior. Esta migration adiciona a coluna status e recria
-- iris_error_rate_per_node para usar status em vez do proxy JSONB.
--
-- Numeração: 017 foi ocupada duas vezes (C3 + C4). C11 (EASAA-32) é 018.
-- Relacionado: EASAA-32, EASAA-20.

BEGIN;

-- ============================================================================
-- 1. Coluna `status` em sf_agent_activations
-- ============================================================================

ALTER TABLE public.sf_agent_activations
  ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'success'
  CHECK (status IN ('success', 'error', 'blocked'));

COMMENT ON COLUMN public.sf_agent_activations.status IS
  'Resultado do nó. success=normal, error=exceção/LLM fail, blocked=idempotência.';

-- ============================================================================
-- 2. iris_latency_per_node — idempotent (CREATE OR REPLACE)
-- ============================================================================

CREATE OR REPLACE VIEW public.iris_latency_per_node AS
SELECT
    clinic_id,
    agent_name                                                       AS node_name,
    DATE_TRUNC('hour', started_at)                                   AS bucket,
    COUNT(*)                                                         AS calls,
    AVG(duration_ms)                                                 AS avg_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms)        AS p95_ms
FROM public.sf_agent_activations
WHERE started_at IS NOT NULL AND duration_ms IS NOT NULL
GROUP BY clinic_id, agent_name, DATE_TRUNC('hour', started_at);

COMMENT ON VIEW public.iris_latency_per_node IS
  'Latência (ms) média e p95 por nó Iris, por clínica, agregada por hora.';

-- ============================================================================
-- 3. iris_cost_per_message — idempotent (CREATE OR REPLACE)
-- ============================================================================

CREATE OR REPLACE VIEW public.iris_cost_per_message AS
SELECT
    clinic_id,
    session_id,
    MIN(started_at)                    AS session_start,
    COUNT(*)                           AS activations,
    SUM(cost_usd)                      AS total_cost_usd,
    SUM(total_tokens)                  AS total_tokens
FROM public.sf_agent_activations
WHERE session_id IS NOT NULL
GROUP BY clinic_id, session_id;

COMMENT ON VIEW public.iris_cost_per_message IS
  'Custo USD agregado por conversa (session_id), por clínica.';

-- ============================================================================
-- 4. iris_error_rate_per_node — DROP + CREATE (usa status column, não JSONB)
-- ============================================================================

DROP VIEW IF EXISTS public.iris_error_rate_per_node;

CREATE VIEW public.iris_error_rate_per_node AS
SELECT
    clinic_id,
    agent_name                                                         AS node_name,
    DATE_TRUNC('hour', started_at)                                     AS bucket,
    COUNT(*)                                                           AS total,
    COUNT(*) FILTER (WHERE status = 'error')                           AS errors,
    ROUND(
        (COUNT(*) FILTER (WHERE status = 'error'))::NUMERIC
        / NULLIF(COUNT(*), 0),
        4
    )                                                                  AS error_rate
FROM public.sf_agent_activations
WHERE started_at IS NOT NULL
GROUP BY clinic_id, agent_name, DATE_TRUNC('hour', started_at);

COMMENT ON VIEW public.iris_error_rate_per_node IS
  'Taxa de erro (0.0–1.0) por nó Iris, por clínica, agregada por hora. Usa status column.';

COMMIT;
