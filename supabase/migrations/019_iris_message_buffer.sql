-- Migration 019 — sf_message_buffer + iris_try_flush_conversation RPC
--
-- Iris debounce accumulator ([EASAA-141](../../../EASAA/issues/EASAA-141)).
--
-- A paciente que digita "oi" + "quero agendar" + "amanhã de tarde" em
-- 5 segundos não pode receber 3 respostas. O webhook (C7) persiste cada
-- inbound em `sf_messages` (idempotência mantida) e também enfileira em
-- `sf_message_buffer`. Cada chegada agenda um background task no app que
-- dorme `IRIS_DEBOUNCE_MS` (default 8s) e então chama
-- `iris_try_flush_conversation`. A RPC serializa via `pg_advisory_xact_lock`
-- por (clinic_id, remote_jid) e devolve TRUE+payload só pro task cuja
-- mensagem é a mais recente — todos os outros desistem porque encontram
-- alguma mensagem unflushed mais nova. O resultado é uma única chamada
-- da pipeline para todas as mensagens da janela, concatenadas com `\n`.
--
-- RLS: service_role bypassa. Política mesma do sf_messages.

BEGIN;

-- ============================================================================
-- 1. sf_message_buffer
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.sf_message_buffer (
  id              UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
  clinic_id       UUID         NOT NULL REFERENCES public.sf_clinics(id) ON DELETE CASCADE,
  remote_jid      TEXT         NOT NULL,
  message_id      UUID         NOT NULL REFERENCES public.sf_messages(id) ON DELETE CASCADE,
  content         TEXT         NOT NULL,
  instance_name   TEXT         NOT NULL,
  push_name       TEXT,
  message_type    TEXT         NOT NULL DEFAULT 'text',
  wamid           TEXT         NOT NULL,
  trace_id        TEXT,
  created_at      TIMESTAMPTZ  NOT NULL DEFAULT clock_timestamp(),
  flushed_at      TIMESTAMPTZ,
  CONSTRAINT sf_message_buffer_message_id_unique UNIQUE (message_id)
);

CREATE INDEX IF NOT EXISTS idx_sf_message_buffer_pending
  ON public.sf_message_buffer (clinic_id, remote_jid, created_at)
  WHERE flushed_at IS NULL;

COMMENT ON TABLE  public.sf_message_buffer IS
  'Iris debounce queue. Each inbound webhook enqueues a row; a background task flushes after IRIS_DEBOUNCE_MS via iris_try_flush_conversation. flushed_at is set atomically with the pipeline trigger.';
COMMENT ON COLUMN public.sf_message_buffer.message_id IS
  'FK to sf_messages.id. UNIQUE so re-enqueuing the same logical message is impossible.';
COMMENT ON COLUMN public.sf_message_buffer.flushed_at IS
  'NULL = pending. Set to now() inside the flush RPC transaction. Older flushed rows are retained for forensic replay.';
COMMENT ON COLUMN public.sf_message_buffer.created_at IS
  'clock_timestamp() (not now()) so timestamps advance within a transaction — guarantees the watermark ordering used by iris_try_flush_conversation.';

-- ============================================================================
-- 2. RLS — same shape as sf_messages
-- ============================================================================

ALTER TABLE public.sf_message_buffer ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS sf_message_buffer_tenant_isolation ON public.sf_message_buffer;
CREATE POLICY sf_message_buffer_tenant_isolation ON public.sf_message_buffer
  FOR ALL TO PUBLIC
  USING (clinic_id = current_setting('app.current_tenant', true)::uuid)
  WITH CHECK (clinic_id = current_setting('app.current_tenant', true)::uuid);

-- ============================================================================
-- 3. iris_try_flush_conversation RPC
--
--   Returns one row:
--     flushed              BOOL    — true when this caller is the canonical
--                                    flush task (no newer unflushed message exists).
--     message_ids          UUID[]  — sf_messages.id of every flushed row,
--                                    ordered chronologically.
--     concatenated_content TEXT    — message contents joined with `\n`.
--     buffer_count         INT     — number of buffer rows flushed.
--     latest_buffer_id     UUID    — id of the newest flushed buffer row;
--                                    its push_name / wamid / instance_name
--                                    are used as the pipeline trigger anchor.
--
--   The lock is transaction-scoped (`pg_advisory_xact_lock`). It is released
--   automatically when the function returns and the RPC transaction commits.
-- ============================================================================

CREATE OR REPLACE FUNCTION public.iris_try_flush_conversation(
  p_clinic_id          UUID,
  p_remote_jid         TEXT,
  p_watermark_buffer_id UUID
)
RETURNS TABLE (
  flushed              BOOLEAN,
  message_ids          UUID[],
  concatenated_content TEXT,
  buffer_count         INTEGER,
  latest_buffer_id     UUID
)
LANGUAGE plpgsql
AS $$
DECLARE
  v_lock_key       BIGINT;
  v_watermark_at   TIMESTAMPTZ;
  v_has_newer      BOOLEAN;
BEGIN
  -- ----- per-conversation advisory lock (transaction-scoped) -----
  v_lock_key := hashtextextended(p_clinic_id::text || ':' || p_remote_jid, 0);
  PERFORM pg_advisory_xact_lock(v_lock_key);

  -- ----- watermark of the caller's own buffer row -----
  SELECT created_at INTO v_watermark_at
  FROM public.sf_message_buffer
  WHERE id = p_watermark_buffer_id;

  IF v_watermark_at IS NULL THEN
    -- Caller's row vanished (shouldn't happen). Treat as already handled.
    RETURN QUERY SELECT FALSE, ARRAY[]::UUID[], ''::TEXT, 0, NULL::UUID;
    RETURN;
  END IF;

  -- ----- bail if a newer unflushed message exists; its task will flush -----
  SELECT EXISTS (
    SELECT 1
    FROM public.sf_message_buffer
    WHERE clinic_id  = p_clinic_id
      AND remote_jid = p_remote_jid
      AND flushed_at IS NULL
      AND created_at > v_watermark_at
  ) INTO v_has_newer;

  IF v_has_newer THEN
    RETURN QUERY SELECT FALSE, ARRAY[]::UUID[], ''::TEXT, 0, NULL::UUID;
    RETURN;
  END IF;

  -- ----- atomic flush: mark rows + return aggregated payload -----
  RETURN QUERY
  WITH flushed_rows AS (
    UPDATE public.sf_message_buffer
       SET flushed_at = now()
     WHERE clinic_id  = p_clinic_id
       AND remote_jid = p_remote_jid
       AND flushed_at IS NULL
    RETURNING id, message_id, content, created_at
  )
  SELECT
    TRUE,
    ARRAY(SELECT message_id FROM flushed_rows ORDER BY created_at),
    COALESCE(
      (SELECT string_agg(content, E'\n' ORDER BY created_at) FROM flushed_rows),
      ''
    ),
    (SELECT COUNT(*)::INT FROM flushed_rows),
    (SELECT id FROM flushed_rows ORDER BY created_at DESC LIMIT 1);
END;
$$;

COMMENT ON FUNCTION public.iris_try_flush_conversation IS
  'Iris debounce: atomically claim and flush all pending sf_message_buffer rows for (clinic_id, remote_jid). Returns flushed=FALSE if a newer unflushed message exists — its background task will become the canonical flush.';

COMMIT;
