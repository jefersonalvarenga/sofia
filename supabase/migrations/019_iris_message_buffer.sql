-- Migration 019 — sf_message_buffer (debounce accumulator)
--
-- EASAA-141: acumula mensagens por (clinic_id, conversation_id) numa janela
-- de debounce antes de disparar a pipeline. Impede respostas fragmentadas
-- quando o paciente envia múltiplas mensagens em sequência.

BEGIN;

CREATE TABLE IF NOT EXISTS public.sf_message_buffer (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    clinic_id       uuid        NOT NULL,
    conversation_id text        NOT NULL,   -- remote_jid normalizado
    message_id      uuid        NOT NULL,   -- FK → sf_messages.id (informacional)
    wamid           text        NOT NULL,
    content         text        NOT NULL,
    push_name       text        NOT NULL DEFAULT '',
    message_type    text        NOT NULL DEFAULT 'text',
    instance_name   text        NOT NULL DEFAULT '',
    phone           text        NOT NULL DEFAULT '',
    flush_after     timestamptz NOT NULL,   -- deadline da janela (pode ser estendido)
    flushed         boolean     NOT NULL DEFAULT false,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Índice para o worker buscar janelas vencidas de forma eficiente
CREATE INDEX IF NOT EXISTS sf_message_buffer_flush
    ON public.sf_message_buffer (clinic_id, conversation_id, flushed, flush_after);

COMMIT;
