-- Migration 018 — Iris observability views (C11 / EASAA-32)
--
-- Three read-only views over sf_agent_activations to power the Iris MVP
-- dashboard. Plain views (not materialized): row volume is small and
-- freshness matters more than scan cost at this stage. Switch to
-- materialized views when the table grows past ~10M rows or the dashboard
-- query budget gets tight.
--
-- Numbering note: this migration was originally specified as 017, but
-- 017 was claimed twice (017_cost_usd_in_agent_activations from C4 /
-- EASAA-25, and 017_iris_messages_and_rls from C3). Renumbered to 018
-- per coordination comment on EASAA-32.
--
-- Depends on:
--   - 002_sofia_agent_activations.sql (base table)
--   - 009_sofia_v2_observability.sql (started_at, duration_ms, messages, data)
--   - 017_cost_usd_in_agent_activations.sql (cost_usd column)

CREATE OR REPLACE VIEW public.iris_latency_per_node AS
SELECT
  clinic_id,
  agent_name AS node_name,
  DATE_TRUNC('hour', started_at) AS bucket,
  COUNT(*) AS calls,
  AVG(duration_ms) AS avg_ms,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_ms
FROM public.sf_agent_activations
WHERE started_at IS NOT NULL
  AND duration_ms IS NOT NULL
GROUP BY clinic_id, agent_name, DATE_TRUNC('hour', started_at);

COMMENT ON VIEW public.iris_latency_per_node IS
  'Per-clinic, per-agent latency rollup bucketed by hour. avg_ms + p95_ms over duration_ms.';

CREATE OR REPLACE VIEW public.iris_cost_per_message AS
SELECT
  clinic_id,
  session_id,
  MIN(started_at) AS session_start,
  COUNT(*) AS activations,
  SUM(cost_usd) AS total_cost_usd,
  SUM(total_tokens) AS total_tokens
FROM public.sf_agent_activations
WHERE session_id IS NOT NULL
GROUP BY clinic_id, session_id;

COMMENT ON VIEW public.iris_cost_per_message IS
  'Per-session cost rollup: USD and token totals per (clinic_id, session_id).';

-- Error proxy rationale: sf_agent_activations has no `status` column today.
-- We approximate errors with two heuristics:
--   1. data->>'status' = 'error'  (set by app code when an exception path runs)
--   2. messages -> 0 ->> 'content' ILIKE 'desculpe%'
--      (the canonical apology prefix used by error fallback messages)
-- Promote a real `status` column in a follow-up migration once the app
-- code consistently sets it; then this view should switch to that single
-- predicate.
CREATE OR REPLACE VIEW public.iris_error_rate_per_node AS
SELECT
  clinic_id,
  agent_name AS node_name,
  DATE_TRUNC('hour', started_at) AS bucket,
  COUNT(*) AS total,
  COUNT(*) FILTER (
    WHERE data ->> 'status' = 'error'
       OR (messages -> 0 ->> 'content') ILIKE 'desculpe%'
  ) AS errors,
  (
    COUNT(*) FILTER (
      WHERE data ->> 'status' = 'error'
         OR (messages -> 0 ->> 'content') ILIKE 'desculpe%'
    )::FLOAT
    / NULLIF(COUNT(*), 0)
  ) AS error_rate
FROM public.sf_agent_activations
WHERE started_at IS NOT NULL
GROUP BY clinic_id, agent_name, DATE_TRUNC('hour', started_at);

COMMENT ON VIEW public.iris_error_rate_per_node IS
  'Per-clinic, per-agent error-rate rollup bucketed by hour. Error proxy: data.status=error OR assistant message starts with "Desculpe".';
