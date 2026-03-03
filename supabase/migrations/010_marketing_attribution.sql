-- ============================================================================
-- Migration 010: Marketing Attribution Layer
-- Sofia 2.1 — tracks click → contact → appointment → revenue
-- ============================================================================

-- ---------------------------------------------------------------------------
-- sf_ad_clicks: one row per ad click (gclid / fbclid)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.sf_ad_clicks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clinic_id       UUID NOT NULL REFERENCES public.sf_clinics(id) ON DELETE CASCADE,
    ref_code        TEXT NOT NULL,
    gclid           TEXT,                    -- Google Ads click ID
    fbclid          TEXT,                    -- Meta Ads click ID
    utm_source      TEXT,                    -- "google" | "meta"
    utm_medium      TEXT,                    -- "cpc" | "social"
    utm_campaign    TEXT,
    utm_content     TEXT,
    utm_term        TEXT,
    landing_page    TEXT,
    clicked_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 days'),
    resolved        BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at     TIMESTAMPTZ,
    resolved_jid    TEXT,
    customer_id     UUID REFERENCES public.sf_customers(id) ON DELETE SET NULL,
    -- conversion upload tracking
    conversion_uploaded_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT sf_ad_clicks_ref_code_clinic_unique UNIQUE (clinic_id, ref_code)
);

-- Uniqueness per clinic. Expired codes cannot be recycled (safe for small clinics —
-- 36^3 = 46,656 combinations per clinic is ample).
-- NOTE: partial index WHERE expires_at > NOW() cannot be used (NOW() is not IMMUTABLE).
CREATE INDEX IF NOT EXISTS idx_sf_ad_clicks_lookup
    ON public.sf_ad_clicks (ref_code, clinic_id, expires_at);

CREATE INDEX IF NOT EXISTS idx_sf_ad_clicks_campaign
    ON public.sf_ad_clicks (clinic_id, utm_campaign, clicked_at);

CREATE INDEX IF NOT EXISTS idx_sf_ad_clicks_gclid
    ON public.sf_ad_clicks (gclid)
    WHERE gclid IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sf_ad_clicks_fbclid
    ON public.sf_ad_clicks (fbclid)
    WHERE fbclid IS NOT NULL;

ALTER TABLE public.sf_ad_clicks DISABLE ROW LEVEL SECURITY;


-- ---------------------------------------------------------------------------
-- sf_ad_spend: manual ad spend input (Google Ads API / Meta Ads API is Phase 2)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.sf_ad_spend (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clinic_id       UUID NOT NULL REFERENCES public.sf_clinics(id) ON DELETE CASCADE,
    platform        TEXT NOT NULL,           -- "google" | "meta"
    campaign_name   TEXT NOT NULL,           -- must match utm_campaign exactly
    spend_date      DATE NOT NULL,
    spend_brl       NUMERIC(12, 2) NOT NULL,
    impressions     INTEGER,
    clicks_platform INTEGER,
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT sf_ad_spend_unique_day
        UNIQUE (clinic_id, platform, campaign_name, spend_date)
);

ALTER TABLE public.sf_ad_spend DISABLE ROW LEVEL SECURITY;


-- ---------------------------------------------------------------------------
-- sf_clinic_ad_accounts: per-clinic Google Ads + Meta credentials
-- Accessed only via service_role key (never anon)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.sf_clinic_ad_accounts (
    clinic_id                           UUID PRIMARY KEY REFERENCES public.sf_clinics(id),
    -- Google Ads
    google_ads_enabled                  BOOLEAN NOT NULL DEFAULT FALSE,
    google_ads_customer_id              TEXT,    -- "123-456-7890"
    google_ads_developer_token          TEXT,    -- EasyScale shared developer token
    google_ads_refresh_token            TEXT,    -- OAuth token for this clinic's account
    google_conversion_action_lead       TEXT,    -- resource name of conversion action
    google_conversion_action_appointment TEXT,
    google_conversion_action_purchase   TEXT,
    -- Meta (Facebook/Instagram)
    meta_enabled                        BOOLEAN NOT NULL DEFAULT FALSE,
    meta_pixel_id                       TEXT,
    meta_access_token                   TEXT,    -- pixel/dataset token
    meta_test_event_code                TEXT,    -- for sandbox testing (TEST12345)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.sf_clinic_ad_accounts DISABLE ROW LEVEL SECURITY;


-- ---------------------------------------------------------------------------
-- FKs on existing tables
-- ---------------------------------------------------------------------------

-- sf_appointments: conversion event
ALTER TABLE public.sf_appointments
    ADD COLUMN IF NOT EXISTS attribution_id UUID
        REFERENCES public.sf_ad_clicks(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_sf_appointments_attribution
    ON public.sf_appointments (attribution_id)
    WHERE attribution_id IS NOT NULL;

-- sf_customers: first-touch attribution
ALTER TABLE public.sf_customers
    ADD COLUMN IF NOT EXISTS first_attribution_id UUID
        REFERENCES public.sf_ad_clicks(id) ON DELETE SET NULL;


-- ---------------------------------------------------------------------------
-- Analytic views
-- ---------------------------------------------------------------------------

-- Funnel: click → contact → booking
CREATE OR REPLACE VIEW public.vw_attribution_funnel AS
SELECT
    ac.id             AS click_id,
    ac.clinic_id,
    ac.ref_code,
    ac.utm_source,
    ac.utm_campaign,
    ac.utm_content,
    ac.gclid,
    ac.fbclid,
    ac.clicked_at,
    ac.resolved                                                AS contact_made,
    ac.resolved_at                                             AS contacted_at,
    ap.id             AS appointment_id,
    ap.service_name   AS booked_service,
    ap.scheduled_at   AS booked_at,
    ap.status         AS appointment_status,
    EXTRACT(EPOCH FROM (ac.resolved_at - ac.clicked_at)) / 3600.0
                                                               AS hours_click_to_contact,
    EXTRACT(EPOCH FROM (ap.created_at - ac.resolved_at)) / 3600.0
                                                               AS hours_contact_to_booking,
    CASE
        WHEN ap.id IS NOT NULL THEN 'booked'
        WHEN ac.resolved = TRUE  THEN 'contacted'
        ELSE 'clicked'
    END                                                        AS funnel_stage
FROM public.sf_ad_clicks ac
LEFT JOIN public.sf_appointments ap
    ON  ap.attribution_id = ac.id
    AND ap.status NOT IN ('cancelled', 'no_show');


-- Aggregated campaign performance: CAC, ROAS, CPL, CPA
CREATE OR REPLACE VIEW public.vw_campaign_performance AS
WITH clicks AS (
    SELECT
        ac.clinic_id,
        ac.utm_source                                          AS platform,
        ac.utm_campaign                                        AS campaign_name,
        DATE_TRUNC('week', ac.clicked_at)                      AS week_start,
        COUNT(*)                                               AS total_clicks,
        COUNT(*) FILTER (WHERE ac.resolved = TRUE)             AS total_contacts,
        COUNT(DISTINCT ap.id)
            FILTER (WHERE ap.id IS NOT NULL)                   AS total_bookings
    FROM public.sf_ad_clicks ac
    LEFT JOIN public.sf_appointments ap
        ON  ap.attribution_id = ac.id
        AND ap.status NOT IN ('cancelled', 'no_show')
    GROUP BY 1, 2, 3, 4
),
spend AS (
    SELECT
        clinic_id, platform, campaign_name,
        DATE_TRUNC('week', spend_date) AS week_start,
        SUM(spend_brl)                 AS spend_brl
    FROM public.sf_ad_spend
    GROUP BY 1, 2, 3, 4
),
avg_ticket AS (
    SELECT clinic_id, COALESCE(avg_ticket, 0) AS avg_ticket
    FROM public.sf_clinic_profiles
)
SELECT
    c.clinic_id,
    c.platform,
    c.campaign_name,
    c.week_start,
    c.total_clicks,
    c.total_contacts,
    c.total_bookings,
    ROUND(100.0 * c.total_contacts::NUMERIC / NULLIF(c.total_clicks,   0), 2) AS ctr_pct,
    ROUND(100.0 * c.total_bookings::NUMERIC / NULLIF(c.total_contacts, 0), 2) AS contact_to_booking_pct,
    COALESCE(s.spend_brl, 0)                                                   AS spend_brl,
    ROUND(COALESCE(s.spend_brl, 0) / NULLIF(c.total_contacts, 0), 2)          AS cpl_brl,
    ROUND(COALESCE(s.spend_brl, 0) / NULLIF(c.total_bookings, 0), 2)          AS cac_brl,
    ROUND(c.total_bookings * COALESCE(at.avg_ticket, 0), 2)                   AS revenue_proxy_brl,
    ROUND(
        (c.total_bookings * COALESCE(at.avg_ticket, 0))
        / NULLIF(COALESCE(s.spend_brl, 0), 0),
        2
    )                                                                          AS roas
FROM clicks c
LEFT JOIN spend      s  USING (clinic_id, platform, campaign_name, week_start)
LEFT JOIN avg_ticket at USING (clinic_id);
