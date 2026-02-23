-- Migration 002: Create sf_agent_activations audit table
-- Tracks every agent activation for observability and debugging

CREATE TABLE IF NOT EXISTS public.sf_agent_activations (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id    TEXT,
  chat_id       UUID,
  agent_name    TEXT NOT NULL,
  triggered_by  TEXT,
  reasoning     TEXT,
  processing_ms FLOAT,
  sofia_version TEXT NOT NULL DEFAULT '1.0',
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sf_activations_session
  ON public.sf_agent_activations(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_sf_activations_agent
  ON public.sf_agent_activations(agent_name);
