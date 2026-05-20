-- Migration 029 — schedule_session_data column on sf_sessions
ALTER TABLE public.sf_sessions
  ADD COLUMN IF NOT EXISTS schedule_session_data jsonb NOT NULL DEFAULT '[]'::jsonb;

COMMENT ON COLUMN public.sf_sessions.schedule_session_data IS
  'Cross-agent state for SCHEDULE_* sub-agents. List of {name, data} entries (evaluation / service / confirmation / reminder). Populated by ScheduleIntakeAgent and successors; survives across turns.';
