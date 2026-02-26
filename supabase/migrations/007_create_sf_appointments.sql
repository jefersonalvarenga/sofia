-- Migration 007: Creates Sofia's own appointments table.
-- The legacy 'appointments' table has incompatible schema
-- (no clinic_id, uses 'date'/'service' instead of 'scheduled_at'/'service_name').

CREATE TABLE public.sf_appointments (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clinic_id       UUID NOT NULL REFERENCES public.clinics(id) ON DELETE CASCADE,
    customer_id     UUID REFERENCES public.customers(id) ON DELETE SET NULL,
    session_id      TEXT REFERENCES public.sf_sessions(session_id) ON DELETE SET NULL,
    remote_jid      TEXT NOT NULL,
    patient_name    TEXT,
    service_name    TEXT,
    scheduled_at    TIMESTAMPTZ NOT NULL,
    status          TEXT NOT NULL DEFAULT 'scheduled',
    source          TEXT NOT NULL DEFAULT 'sofia',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- For slot availability queries (clinic_id + scheduled_at range)
CREATE INDEX sf_appointments_clinic_slot_idx
    ON public.sf_appointments (clinic_id, scheduled_at);

-- For reset flow (remote_jid filter)
CREATE INDEX sf_appointments_remote_jid_idx
    ON public.sf_appointments (remote_jid);

-- RLS disabled — access controlled via API key at application layer
ALTER TABLE public.sf_appointments DISABLE ROW LEVEL SECURITY;
