# ADR 0001 — Iris tenant isolation: backend é defesa real, RLS é cinto+suspensório

- **Status:** Accepted
- **Date:** 2026-05-07
- **Deciders:** CTO (this ADR), CEO (sign-off via PR review)
- **Story:** [EASAA-20](../../../EASAA/issues/EASAA-20) — Iris greeting smoke (fundação arquitetural)
- **This ticket:** [EASAA-33](../../../EASAA/issues/EASAA-33) — C12
- **Related:**
  [EASAA-24](../../../EASAA/issues/EASAA-24) (C3 — `sf_messages` + RLS, migration 017),
  [EASAA-28](../../../EASAA/issues/EASAA-28) (C7 — webhook Evolution → FastAPI),
  [EASAA-29](../../../EASAA/issues/EASAA-29) (C8 — pipeline Iris),
  [ADR 0002](./0002-dna-canonical.md) (canonical clinic DNA)

## Context

EasyScale opera uma plataforma AI-first para clínicas estéticas brasileiras. A
mesma codebase (`sofia`, evoluindo para Iris) e a mesma instância Supabase
(`brasilnatech`) servem **N clínicas distintas**. Cada clínica é um tenant —
chamado `clinic_id` no código histórico do Sofia, `tenant_id` na spec
original da Iris.

Isolamento de tenant é o invariante #1 da plataforma: paciente da clínica A
**nunca** pode ver, modificar, ou treinar a Iris da clínica B. Vazamento aqui
é one-way door: vira incidente de privacidade, possivelmente LGPD, e mata a
confiança das clínicas pagantes.

A fundação que estamos cimentando agora ([EASAA-20](../../../EASAA/issues/EASAA-20))
vai ser herdada por 7+ stories de specialists futuros (Router, FAQ, Scheduler,
HumanEscalation, Closure, etc.). Esta ADR fixa o **padrão único** de
isolamento de tenant que essas stories e qualquer feature nova devem seguir.

### Stack relevante para esta decisão

- **Linguagem/runtime:** FastAPI + Python 3.11. Endpoint serve um único
  webhook Evolution → pipeline LangGraph.
- **Data layer:** `supabase-py` 2.10 (cliente REST/PostgREST). O backend
  autentica como `service_role` — chave administrativa que **bypassa RLS**
  por design do PostgREST.
- **Postgres:** Supabase (Postgres 15) gerenciado. Tabelas tenant-scoped têm
  prefixo `sf_` por herança histórica do Sofia.
- **Convenção atual de query:** toda função em `app/session/manager.py`,
  `app/agents/*/agent.py`, etc., recebe `clinic_id` como argumento explícito
  e injeta `.eq("clinic_id", clinic_id)` em cada query Supabase.
- **Ponto de entrada do tenant:** `sf_instance_clinic_map` resolve
  `instance_name` (Evolution) → `clinic_id`. A partir daí, `clinic_id` é
  carregado em `Session` e propagado por todo o pipeline.

### O que está em jogo nesta ADR

Três decisões correlatas que precisam ser fixadas juntas porque um padrão
incompleto vaza tenants:

1. Onde mora a defesa real de isolamento — código de aplicação ou banco?
2. Como garantimos idempotência de webhooks (Evolution reentrega `wamid`s)
   sem violar isolamento ou abrir corrida entre tenants?
3. Como nomeamos os conceitos para que o time corporativo, specialists
   futuros, e a documentação Iris falem a mesma língua?

## Decision

### 1. Backend é a defesa real

**Toda query Supabase recebe `clinic_id` explícito como argumento e injeta
`.eq("clinic_id", clinic_id)` em todo `select`, `update`, `delete`, e em todo
payload de `insert`/`upsert`.** Não existe query sem `clinic_id`. Code review
bloqueia PR sem isso.

**Enforcement:** lint custom AST em `tests/iris/test_lint_tenant_id.py`
(planejado — owner C# de hardening, ainda não criado em código). O lint
percorre o AST de `app/**/*.py` e falha se encontra:

- chamada `client.table("sf_*")` cujo encadeamento subsequente não contém
  `.eq("clinic_id", …)`;
- payload de `insert`/`upsert` que não inclui chave `clinic_id`;
- exceções explícitas (e.g. `sf_instance_clinic_map`, que **resolve**
  `clinic_id` e portanto não pode exigir a coluna como filtro) marcadas via
  comentário `# tenant-lint: exempt — <razão>`.

Até o lint existir, code review humano + esta ADR são o gate. O risco de
drift é alto se a regra ficar só na cabeça de quem revisa, então a primeira
prioridade pós-greeting-smoke é fechar este gap.

### 2. RLS habilitada como cinto+suspensório

`migration 017_iris_messages_and_rls.sql` ([EASAA-24](../../../EASAA/issues/EASAA-24))
habilita `ROW LEVEL SECURITY` e cria policy única por tabela tenant-scoped:

```sql
ALTER TABLE public.sf_messages ENABLE ROW LEVEL SECURITY;
CREATE POLICY sf_messages_tenant_isolation ON public.sf_messages
  FOR ALL TO PUBLIC
  USING (clinic_id = current_setting('app.current_tenant', true)::uuid)
  WITH CHECK (clinic_id = current_setting('app.current_tenant', true)::uuid);
```

O segundo argumento `true` em `current_setting('app.current_tenant', true)`
faz a função devolver `NULL` quando a variável não está setada — assim a
policy bloqueia (zero rows) em vez de estourar
`unrecognized configuration parameter`. Smoke manual capturado no rodapé da
migration confirma: `SET ROLE authenticated; SELECT count(*) FROM sf_clinics`
retorna `0` sem o setting; com `SET LOCAL app.current_tenant =
'<clinic_id>'`, retorna apenas a row daquela clínica.

**Hoje a policy não bloqueia nada no caminho hot.** Dois motivos:

1. O backend usa `service_role`, que **bypassa RLS** por design do
   PostgREST/Supabase. Isso é esperado e necessário para o webhook
   Evolution funcionar sem trazer JWT por mensagem.
2. O `supabase-py` 2.10 não expõe `SET LOCAL` por sessão — é cliente REST
   stateless. Mesmo se trocássemos de `service_role` para `anon`/JWT, sem
   `SET LOCAL app.current_tenant` por request a policy filtraria tudo (zero
   rows) e o app quebraria.

RLS, portanto, é **proteção futura** ativada quando — e só quando — abrirmos
um caminho cliente direto (e.g. dashboard do paciente lendo Supabase via
JWT/anon). A policy já está no banco, testada e versionada. No dia em que
esse caminho surgir, a checklist é: (a) trocar `service_role` por
JWT-by-clinic com claim `app_metadata.clinic_id`, (b) adicionar middleware
que executa `SET LOCAL app.current_tenant = <clinic_id>` por request via
psycopg direto (não supabase-py), (c) remover `service_role` do hot path.

Em outras palavras: backend é o cinto que segura o sistema. RLS é o
suspensório que dorme guardado no armário até o dia em que precisarmos
dele — e está limpo, testado, pronto.

### 3. Convenção de nomenclatura: `clinic_id` ≡ `tenant_id`, `sf_*` ≡ Iris core

- **`clinic_id` (Sofia/código) ≡ `tenant_id` (spec Iris).** Não vamos
  renomear. `clinic_id` está em ~41 ocorrências em `app/`, em todas as
  migrations, e em produção do Sofia. Rename em massa é churn one-way door
  sem ganho de produto.
- **Tabelas com prefixo `sf_` são as tabelas core da Iris.** A spec original
  da Iris falava em `clinics`, `messages`, `sessions`, etc. Usar os nomes
  `sf_*` evita migration de rename em produção do Sofia (que segue rodando
  para Lumina/Sgen) e respeita o princípio "boring tech / mudanças
  cirúrgicas" estabelecido no [plano EASAA-20](../../../EASAA/issues/EASAA-20#document-plan).
- **`la_blueprints`** (Legacy Analyzer) fica sem prefixo `sf_` por herança e
  porque é cross-clinic (RLS-protegida via `clinic_id` ainda assim).

Consequência prática: specialists futuros leem `clinic_id` no código,
escrevem `clinic_id` em PRs novos, e tratam o termo `tenant_id` como
sinônimo da spec quando alinharem com clientes/board. Documentação Iris
externa pode usar `tenant_id`; código usa `clinic_id`.

### 4. Idempotência atômica via `UNIQUE (clinic_id, wamid)` + `ON CONFLICT DO NOTHING`

Evolution reentrega webhooks (timeout, retry, replay manual). Sem
idempotência atômica, a Iris responde duas vezes a uma mensagem do
paciente, gera dois `agent_run`s, dobra custo de LLM, e — pior — pode
inserir registros conflitantes em `sf_sessions.history`.

**Decisão:**

```sql
CREATE TABLE public.sf_messages (
  id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  clinic_id UUID NOT NULL REFERENCES public.sf_clinics(id) ON DELETE CASCADE,
  wamid     TEXT NOT NULL,
  ...
  CONSTRAINT sf_messages_clinic_wamid_unique UNIQUE (clinic_id, wamid)
);
```

```python
result = (
    supabase.table("sf_messages")
    .insert(payload, returning="representation")
    .on_conflict("clinic_id,wamid")
    .execute()
)
if not result.data:
    return  # already processed; webhook returns 200 sem reprocessar
```

`UNIQUE (clinic_id, wamid)` — não `UNIQUE (wamid)` — é deliberado: dois
tenants podem coincidentemente ter `wamid`s iguais (Evolution reusa IDs
entre instâncias em alguns casos), e a unicidade real é por tenant.

**Sem advisory lock.** A ideia foi considerada e rejeitada (escalation
prévia, registrada em [EASAA-20](../../../EASAA/issues/EASAA-20)): advisory
lock por `wamid` adiciona round-trip Postgres antes de cada insert,
serializa entradas concorrentes da mesma clínica desnecessariamente, e
falha aberto se o lock não é liberado por crash. `UNIQUE` + `ON CONFLICT`
é atômico no banco, sem estado externo, sem TTL para gerenciar.

### 5. GreetingAgent determinístico, LLM Haiku para nodes contextuais

A Iris original previa LLM Haiku 4.5 com structured output em **todos** os
nodes. O Sofia já tem `app/agents/greeting/agent.py` determinístico (zero
LLM, zero token) que lê `clinic_style.greeting_example` da DNA da clínica
(ver [ADR 0002](./0002-dna-canonical.md)) e adapta opener + emoji.

**Decisão:** GreetingAgent permanece determinístico. Latência ~0ms vs.
600–1200ms do Haiku, custo R$0 vs. ~R$0.001/msg. Atende com folga o
critério "<3s end-to-end" do plano. O determinismo é seguro porque a
saída do greeting é por construção previsível (saudação inicial baseada
em template da clínica) — não há ambiguidade contextual.

**LLM Haiku entra para nodes onde a saída é variável e contextual:**
Router (intent classification multi-class), FAQ (resposta livre baseada
em conhecimento da clínica), Scheduler (extração de slot, desambiguação
de dia, confirmação). Esses nodes precisam de raciocínio sobre input do
paciente — determinismo aqui seria regressão.

A decisão liga a tenant isolation porque define **qual caminho carrega
DNA da clínica via `load_style(clinic_id)`**: o GreetingAgent lê o
`greeting_example` do `clinic_style` na entrada do pipeline, sem
LLM-call intermediário, mantendo o `clinic_id` no contrato direto da
função. Ver [ADR 0002](./0002-dna-canonical.md) para a precedence chain
de DNA.

## Consequences

### Positivas

- **Defesa testável.** O lint AST (uma vez implementado) verifica
  estaticamente que toda query carrega `clinic_id`. Não depende de
  cobertura de teste, não depende de RLS estar ativa, não depende de
  comportamento runtime.
- **Latência menor.** Nenhum middleware tocando `SET LOCAL` por request,
  nenhum advisory lock, GreetingAgent zero-LLM. Iris responde em ms onde a
  spec original previa centenas.
- **Custo menor.** Zero LLM no greeting, sem round-trip extra para advisory
  lock, sem gestão de estado de lock externo.
- **Código mais simples.** Padrão único e explícito: `clinic_id` no
  parâmetro da função, `.eq("clinic_id", clinic_id)` na query. Specialists
  futuros copiam o padrão sem ler 50 páginas de doc.
- **RLS pronta para o dia em que precisar.** Migration 017 versiona as
  policies. Se um caminho cliente direto surgir (dashboard, embed),
  nenhuma migration adicional necessária — só trocar a credencial e
  adicionar middleware de `SET LOCAL`.

### Negativas

- **Lint custom precisa manutenção.** Quando convenções de query mudarem
  (e.g. helper que encapsula `client.table("sf_…")` por trás de uma
  abstração), o AST walker precisa entender o novo padrão. Sem esse
  upgrade, o lint vira false-negative silencioso.
- **RLS hoje é "decoração" sem `SET LOCAL`.** Risco de leitura errada por
  algum specialist futuro: "RLS está ativa, então estou seguro" — quando
  na verdade o `service_role` bypassa tudo. Mitigação: esta ADR e os
  comentários in-file da migration 017 documentam isso explicitamente.
- **Disciplina de code review é obrigatória até o lint existir.** Janela
  de exposição entre o merge desta ADR e o merge do lint AST. Mitigação:
  CTO faz code review pessoal de toda PR Iris até C8 fechar, e a próxima
  story de hardening prioriza o lint.

## Alternatives considered

### A. RLS com `SET LOCAL` via psycopg direto, sem `service_role`

Reescrever o data layer para usar `psycopg` (asyncpg) direto, com
connection pool por request, e middleware FastAPI que executa
`SET LOCAL app.current_tenant = <clinic_id>` antes de qualquer query.
Backend autentica como role `authenticated` ou role-per-tenant.

- ✅ RLS vira defesa real, não decoração.
- ✅ Defense-in-depth genuíno.
- ❌ Reescrita do data layer (~12 arquivos em `app/session`, `app/agents`,
  scripts de eval). Custo estimado: 20–30h de engenharia, mais smoke
  completo de regressão do Sofia em prod (Lumina, Sgen).
- ❌ Migration de credenciais em todos os ambientes (dev, staging, prod
  Supabase compartilhado).
- ❌ Não compatível com `supabase-py`, que é o cliente padrão do
  ecossistema Supabase. Sai do "boring tech" + perde acesso a features
  próximas (auth helpers, storage, realtime).

**Rejeitado nesta story.** Reabriremos a discussão quando o primeiro
endpoint exposed-to-client surgir — aí a reescrita paga porque RLS
**precisa** funcionar, e o supabase-py REST não atende JWT-per-request
com `SET LOCAL`.

### B. Advisory lock por `wamid` para idempotência

```sql
SELECT pg_advisory_xact_lock(hashtext(wamid));
-- ...check if exists; if not, insert...
```

- ✅ Funciona para idempotência cross-table (não exige `UNIQUE` em uma
  tabela específica).
- ❌ Round-trip extra ao banco antes de cada insert.
- ❌ Falha aberto se o cliente crashar mid-transaction antes do `INSERT`,
  permitindo que a próxima entrega entre.
- ❌ Serializa entradas concorrentes que **deveriam** ser independentes
  (mensagens de tenants diferentes com `wamid`s iguais — possível
  no Evolution).
- ❌ Estado oculto: dev novo lendo o código não vê o lock no schema, só
  na chamada Python. `UNIQUE` + `ON CONFLICT` é declarativo e visível na
  migration.

**Rejeitado.** `UNIQUE (clinic_id, wamid)` resolve o problema de forma
declarativa, atômica, e legível.

### C. Service-role tudo, sem RLS

"Backend é a defesa, RLS não funciona mesmo, não habilitar" — opção
minimalista.

- ✅ Migration mais curta, sem código RLS no banco.
- ❌ Quando o caminho cliente direto surgir, alguém precisa lembrar de
  habilitar RLS retroativamente, escrever as policies, e testar — em um
  momento em que o time vai estar focado na nova feature.
- ❌ Time futuro não tem como saber, lendo o schema, qual era a intenção
  de isolamento.

**Rejeitado.** Cinto+suspensório custa pouco (uma migration, ~80 linhas
SQL) e deixa intenção e enforcement futuro registrados no banco.

## Open questions

### Quando migrar para psycopg direto + `SET LOCAL` funcional?

**Resposta atual:** quando o primeiro endpoint exposed-to-client surgir —
dashboard do paciente, embed na landing da clínica, app mobile lendo
Supabase com JWT, etc. Hoje (greeting smoke) não temos nenhum.

**Trigger concreto:** issue criada para esse endpoint deve ter blocker
explícito apontando para uma child issue "C — migrar data layer para
psycopg direto + middleware de SET LOCAL". Esta ADR ganha status
Superseded ou Updated nesse momento.

### Quando o lint AST existe?

Dependência: terminar o greeting smoke em produção (EASAA-20) primeiro
para não bloquear a story principal. Próxima story de hardening cria
`tests/iris/test_lint_tenant_id.py` e o roda em CI. Owner: CTO ou
specialist designado pelo CEO.

## Reversibility

- **Decisão 1 (backend defense + lint):** two-way door. Trocável por
  qualquer outro padrão de enforcement (ORM type system, query builder
  com tenant baked-in, etc.). Custo: refactor das ~41 ocorrências de
  `clinic_id` em `app/`.
- **Decisão 2 (RLS habilitada com policies inativas):** two-way door
  trivial. `DROP POLICY` + `DISABLE ROW LEVEL SECURITY` é uma migration.
  Reverter o oposto também — `ENABLE` + `CREATE POLICY` já tem o template
  na migration 017.
- **Decisão 3 (`clinic_id` ≡ `tenant_id`, `sf_*` é core):** quase one-way
  door. Renomear em massa quebra Sofia em produção; `git grep` + sed pode
  ajudar mas o risco de regressão é alto. Esta é a decisão mais
  load-bearing — por isso documentada em ADR.
- **Decisão 4 (`UNIQUE (clinic_id, wamid)` + `ON CONFLICT`):** two-way
  door. Trocável por advisory lock ou outra estratégia via migration que
  remove a constraint e troca o caller code.
- **Decisão 5 (GreetingAgent determinístico):** two-way door. Trocar para
  LLM Haiku é editar `app/agents/greeting/agent.py` em poucas linhas; a
  contratos com `Session`/`load_style` não mudam.
