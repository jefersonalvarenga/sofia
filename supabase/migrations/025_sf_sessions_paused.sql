-- Migration 025 — sf_sessions.paused flag for escalation handoff
--
-- EASAA-214 — Escalation specialist: acolhimento + flag paused.
--
-- Adds a `paused` column to `sf_sessions` so the escalation specialist
-- can mark a conversation as paused after sending the acolhimento message.
-- The Iris pipeline checks this flag in `load_context` and skips all
-- processing for paused conversations — the bot stops responding until a
-- human manually sets paused=false (out of scope for this issue).

BEGIN;

ALTER TABLE public.sf_sessions
  ADD COLUMN IF NOT EXISTS paused BOOLEAN DEFAULT false;

COMMIT;
