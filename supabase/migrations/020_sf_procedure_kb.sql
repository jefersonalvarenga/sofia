-- Migration 020 — sf_procedure_kb (EASAA-143)
--
-- Knowledge base for Iris Knowledge Specialist (RAG + pgvector).
-- One row per chunk (procedure section). Embeddings via pgvector.
-- Idempotent — safe to run against existing DB.

BEGIN;

-- Enable pgvector extension (already done in prod via EASAA-22; guard for fresh clones).
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.sf_procedure_kb (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id   UUID        NOT NULL REFERENCES public.sf_clinics(id) ON DELETE CASCADE,
  procedure   TEXT        NOT NULL,         -- e.g. "Botox", "Harmonização Facial"
  title       TEXT        NOT NULL,         -- chunk title / section heading
  body        TEXT        NOT NULL,         -- chunk text (≤ ~500 tokens)
  embedding   vector(1536),                 -- text-embedding-3-small; NULL until indexed
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sf_procedure_kb_tenant_idx
  ON public.sf_procedure_kb (tenant_id);

-- IVFFlat index for cosine similarity (only useful once embeddings are present).
-- `lists` = sqrt(approx rows); start at 5 for dev, tune for prod.
CREATE INDEX IF NOT EXISTS sf_procedure_kb_embedding_idx
  ON public.sf_procedure_kb USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 5)
  WHERE embedding IS NOT NULL;

-- ============================================================================
-- RLS
-- ============================================================================

ALTER TABLE public.sf_procedure_kb ENABLE ROW LEVEL SECURITY;

-- Service-role key bypasses RLS (used by Sofia backend).
-- Authenticated users (dashboard) can read their own tenant rows.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'sf_procedure_kb' AND policyname = 'tenant_read'
  ) THEN
    CREATE POLICY tenant_read ON public.sf_procedure_kb
      FOR SELECT
      USING (tenant_id = (current_setting('app.clinic_id', true))::uuid);
  END IF;
END$$;

-- ============================================================================
-- Seed data — Clínica Vitória (tenant_id from migration 014)
-- embeddings are NULL; keyword fallback in KnowledgeSpecialist handles smoke.
-- ============================================================================

DO $$
DECLARE
  vitoria_id UUID;
BEGIN
  SELECT id INTO vitoria_id FROM public.sf_clinics WHERE slug = 'clinica-vitoria' LIMIT 1;
  IF vitoria_id IS NULL THEN
    RETURN; -- dev env without seed clinic; skip
  END IF;

  INSERT INTO public.sf_procedure_kb (tenant_id, procedure, title, body) VALUES

  (vitoria_id, 'Botox', 'O que é Botox?',
   'Botox (toxina botulínica) é aplicado em pequenas doses para relaxar músculos e suavizar rugas de expressão. '
   'Procedimento ambulatorial, dura 15–30 minutos. Efeito surge em 3–7 dias e dura em média 4–6 meses.'),

  (vitoria_id, 'Botox', 'Quem pode fazer Botox?',
   'Botox é indicado para adultos saudáveis que desejam suavizar rugas de expressão (testa, pés-de-galinha, glabela). '
   'Contraindicado em gestantes, lactantes, pessoas com doenças neuromusculares (miastenia gravis) '
   'e pessoas em uso de anticoagulantes — nesses casos é necessária avaliação presencial com a equipe médica.'),

  (vitoria_id, 'Botox', 'Cuidados pós-Botox',
   'Evite deitar nas primeiras 4 horas. Não massageie a área tratada. '
   'Evite atividade física intensa por 24h. Evite exposição solar direta e calor excessivo por 48h. '
   'Resultados finais são visíveis em 7–14 dias.'),

  (vitoria_id, 'Harmonização Facial', 'O que é Harmonização Facial?',
   'Harmonização Facial é um conjunto de procedimentos minimamente invasivos que equilibra as proporções do rosto. '
   'Pode incluir preenchimento com ácido hialurônico, botox, bioestimuladores de colágeno e fios de PDO. '
   'O plano é personalizado por biótipo e objetivo de cada paciente.'),

  (vitoria_id, 'Harmonização Facial', 'Duração e recuperação da Harmonização Facial',
   'A sessão dura 1–3 horas dependendo dos procedimentos incluídos. '
   'Inchaço e hematomas leves são normais nos primeiros dias. '
   'Resultado pode ser visto em 7–21 dias; ácido hialurônico dura 12–18 meses, '
   'bioestimuladores de colágeno têm efeito crescente por até 2 anos.'),

  (vitoria_id, 'Limpeza de Pele', 'O que é Limpeza de Pele Profunda?',
   'Limpeza de pele profissional remove cravos, células mortas e impurezas com extração manual e peeling químico leve. '
   'Recomendada a cada 30–45 dias para peles oleosas ou com acne leve. '
   'Dura aproximadamente 60 minutos. Pele pode ficar levemente avermelhada por algumas horas após.'),

  (vitoria_id, 'Preenchimento', 'Preenchimento Labial com Ácido Hialurônico',
   'Preenchimento labial adiciona volume, define contorno e hidrata os lábios. '
   'Usamos ácido hialurônico, substância natural do organismo. '
   'Procedimento dura 30–45 minutos, efeito imediato com resultado final em 7 dias. '
   'Duração média: 6–12 meses. Pode causar inchaço leve nas primeiras 24–48h.'),

  (vitoria_id, 'Bioestimulador', 'Sculptra / Radiesse — Bioestimuladores de Colágeno',
   'Bioestimuladores de colágeno estimulam a produção natural de colágeno para rejuvenescer a pele de forma gradual. '
   'Indicado para flacidez, perda de volume e rugas profundas. '
   'Geralmente 2–3 sessões com intervalo de 4–6 semanas. Efeito dura 2–3 anos.'),

  (vitoria_id, 'Fios de PDO', 'Fios de PDO — Sustentação e Colágeno',
   'Fios de PDO são implantados sob a pele para sustentar tecidos flácidos e estimular colágeno. '
   'Indicado para flacidez de face, pescoço e papada. '
   'Procedimento com anestesia local, dura 45–60 minutos. '
   'Resultados progressivos por 6–12 meses; fios se dissolvem naturalmente.'),

  (vitoria_id, 'Peeling', 'Peeling Químico',
   'Peeling químico aplica ácidos (glicólico, mandélico, TCA) para renovar a superfície da pele. '
   'Trata manchas, acne, textura irregular e rugas finas. '
   'Intensidade varia de superficial a médio/profundo conforme indicação. '
   'Pele pode descamar por 3–7 dias após peeling médio. Protetor solar é obrigatório.')

  ON CONFLICT DO NOTHING;
END$$;

COMMIT;
