-- Seed — baseline intake questions (5 questions, service_id IS NULL)
--
-- Applies a starter set of clinical intake questions to ALL clinics.
-- Idempotent (skips clinics that already have a baseline row at order 1).
--
-- Spec: kb/07-MVP/Tech/03-Discussoes/schedule/01 - Spec SCHEDULE_INTAKE.md §4.3
--
-- Run AFTER migration 027.

BEGIN;

WITH target_clinics AS (
    SELECT c.id AS clinic_id
    FROM public.sf_clinics c
    WHERE NOT EXISTS (
        SELECT 1
        FROM public.sf_intake_questions q
        WHERE q.clinic_id = c.id
          AND q.service_id IS NULL
          AND q."order" = 1
    )
)
INSERT INTO public.sf_intake_questions
    (clinic_id, service_id, "order", question_text, category, is_required)
SELECT
    tc.clinic_id,
    NULL::uuid AS service_id,
    v."order",
    v.question_text,
    v.category,
    true AS is_required
FROM target_clinics tc
CROSS JOIN (
    VALUES
        (1, 'Você toma algum medicamento regularmente? Se sim, qual(is)?', 'medicamentos'),
        (2, 'Tem alguma alergia conhecida? Especialmente a anestésicos ou produtos aplicados na pele?', 'alergias'),
        (3, 'Está grávida ou amamentando?', 'gestacao'),
        (4, 'Tem alguma doença crônica como diabetes, pressão alta, problemas cardíacos ou no sistema imunológico?', 'cronicas'),
        (5, 'Como está sua pele neste momento? Tem feridas, inflamações, ou está sensível?', 'pele')
) AS v("order", question_text, category);

COMMIT;
