-- Migration 008: Padronizar nomes de tabelas para prefixo sf_
-- PostgreSQL atualiza automaticamente todas as FK constraints ao renomear tabelas.

BEGIN;

ALTER TABLE public.clinics               RENAME TO sf_clinics;
ALTER TABLE public.customers             RENAME TO sf_customers;
ALTER TABLE public.clinic_profiles       RENAME TO sf_clinic_profiles;
ALTER TABLE public.clinic_services       RENAME TO sf_clinic_services;
ALTER TABLE public.clinic_offers         RENAME TO sf_clinic_offers;
ALTER TABLE public.clinic_business_rules RENAME TO sf_clinic_business_rules;
ALTER TABLE public.instance_clinic_map   RENAME TO sf_instance_clinic_map;
ALTER TABLE public.active_chats          RENAME TO sf_active_chats;

COMMIT;
