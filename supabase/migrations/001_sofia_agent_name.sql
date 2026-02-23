-- Migration 001: Add agent_name column to messages table
-- Allows Sofia to tag which agent generated each message in the conversation

ALTER TABLE public.messages
  ADD COLUMN IF NOT EXISTS agent_name TEXT;

CREATE INDEX IF NOT EXISTS idx_messages_agent_name
  ON public.messages(agent_name)
  WHERE agent_name IS NOT NULL;
