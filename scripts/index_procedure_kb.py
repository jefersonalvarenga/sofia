"""
Indexer — gera embeddings para chunks de sf_procedure_kb sem embedding.

Padrão:
  - lê todos os rows com embedding IS NULL (batch)
  - chama OpenAI text-embedding-3-small em batches de até 100 inputs
  - UPDATE em batch (1 por chunk; supabase-py não tem UPSERT em massa)
  - idempotente: rerun pega só o que falta

Uso:
    cd easyscale-sofia
    python scripts/index_procedure_kb.py                  # indexa tudo pendente
    python scripts/index_procedure_kb.py --tenant <uuid>  # restringe a uma clínica
    python scripts/index_procedure_kb.py --reindex        # re-indexa TUDO (ignora embeddings existentes)
    python scripts/index_procedure_kb.py --dry-run        # mostra o que seria feito, sem chamar API

Requer:
  OPENAI_API_KEY (no .env)
  SUPABASE_URL + SUPABASE_KEY (service_role, no .env)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence

warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# OpenAI batch limits (text-embedding-3-small):
#   - max 2048 inputs per request
#   - max ~8191 tokens per input
#   - we use 100 as conservative default (latency + retry-friendly)
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 100

# Retry policy (rate limit / transient errors)
MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 2.0


def _build_chunk_text(row: Dict[str, Any]) -> str:
    """Concatenate procedure + title + body so the embedding captures full context.

    Format: '<procedure> — <title>\n\n<body>'. Mirrors what `_build_context` does
    in agent.py so retrieval and answer-generation see consistent shape.
    """
    procedure = (row.get("procedure") or "").strip()
    title = (row.get("title") or "").strip()
    body = (row.get("body") or "").strip()
    header = f"{procedure} — {title}" if procedure and title else (procedure or title)
    return f"{header}\n\n{body}".strip()


def _embed_batch(client: Any, texts: Sequence[str]) -> List[List[float]]:
    """Embed `texts` in one OpenAI call with exponential backoff."""
    attempt = 0
    backoff = INITIAL_BACKOFF_SEC
    while True:
        attempt += 1
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=list(texts))
            embeddings = [item.embedding for item in resp.data]
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"OpenAI returned {len(embeddings)} embeddings for {len(texts)} inputs"
                )
            return embeddings
        except Exception as exc:  # noqa: BLE001  (we want broad retry here)
            if attempt > MAX_RETRIES:
                raise
            msg = str(exc).lower()
            transient = any(
                token in msg for token in ("rate limit", "timeout", "503", "502", "504", "connection")
            )
            if not transient and attempt > 1:
                # Non-transient on attempt >= 2 — surface immediately so we don't waste retries.
                raise
            print(
                f"  [retry {attempt}/{MAX_RETRIES}] embed batch failed ({type(exc).__name__}); "
                f"sleeping {backoff:.1f}s",
                flush=True,
            )
            time.sleep(backoff)
            backoff *= 2.0


def _persist_embeddings(supabase: Any, updates: List[Dict[str, Any]]) -> int:
    """Update sf_procedure_kb.embedding for each row. Returns count persisted.

    supabase-py 2.x doesn't ship a true batched UPDATE-by-id. We loop with
    `.update().eq("id", ...)`. With BATCH_SIZE=100 this is fine for MVP volumes.
    """
    persisted = 0
    for u in updates:
        try:
            supabase.table("sf_procedure_kb").update(
                {"embedding": u["embedding"]}
            ).eq("id", u["id"]).execute()
            persisted += 1
        except Exception as exc:
            print(f"  ! failed to persist embedding for {u['id']}: {exc}", flush=True)
    return persisted


def _fetch_pending(
    supabase: Any, tenant_id: Optional[str], reindex: bool, page_size: int = 500
) -> List[Dict[str, Any]]:
    """Fetch chunks needing embeddings. With reindex=True, fetch ALL rows."""
    query = supabase.table("sf_procedure_kb").select("id, tenant_id, procedure, title, body")
    if tenant_id:
        query = query.eq("tenant_id", tenant_id)
    if not reindex:
        query = query.is_("embedding", "null")
    # Supabase has a default page cap. Page through to be safe.
    results: List[Dict[str, Any]] = []
    offset = 0
    while True:
        page = query.range(offset, offset + page_size - 1).execute()
        rows = page.data or []
        results.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Index sf_procedure_kb chunks with OpenAI embeddings.")
    parser.add_argument("--tenant", help="Restrict indexing to this tenant_id (uuid).")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Re-embed ALL rows (default: only rows where embedding IS NULL).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without calling OpenAI or writing to Supabase.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Inputs per OpenAI request (default {BATCH_SIZE}, max 2048).",
    )
    args = parser.parse_args(argv)

    # Late imports so --help works without env.
    from app.core.config import get_settings  # noqa: WPS433
    from app.core.supabase_client import get_supabase  # noqa: WPS433
    import openai  # noqa: WPS433

    settings = get_settings()
    if not settings.openai_api_key:
        print("ERROR: OPENAI_API_KEY missing in environment.", file=sys.stderr)
        return 2
    if not settings.supabase_url or not settings.supabase_key:
        print("ERROR: SUPABASE_URL or SUPABASE_KEY missing in environment.", file=sys.stderr)
        return 2

    supabase = get_supabase()
    openai_client = openai.OpenAI(api_key=settings.openai_api_key)

    print(f"Indexer model={EMBED_MODEL} dim={EMBED_DIM} batch={args.batch_size}", flush=True)
    if args.tenant:
        print(f"Tenant filter: {args.tenant}", flush=True)
    if args.reindex:
        print("Reindex mode: ALL rows will be re-embedded.", flush=True)
    if args.dry_run:
        print("DRY-RUN — no API calls, no writes.", flush=True)

    rows = _fetch_pending(supabase, args.tenant, args.reindex)
    if not rows:
        print("Nothing to index. Database is up to date.")
        return 0

    print(f"Found {len(rows)} chunks to embed.", flush=True)

    total_persisted = 0
    t_start = time.perf_counter()
    for batch_start in range(0, len(rows), args.batch_size):
        batch = rows[batch_start : batch_start + args.batch_size]
        texts = [_build_chunk_text(row) for row in batch]
        ids = [row["id"] for row in batch]

        print(
            f"  batch {batch_start // args.batch_size + 1}: "
            f"{len(batch)} chunks ({batch_start + 1}-{batch_start + len(batch)} of {len(rows)})",
            flush=True,
        )

        if args.dry_run:
            for row, text in zip(batch, texts):
                preview = text.replace("\n", " ")[:80]
                print(f"    [dry] {row['id']}  -> '{preview}…'")
            continue

        embeddings = _embed_batch(openai_client, texts)
        updates = [
            {"id": rid, "embedding": emb}
            for rid, emb in zip(ids, embeddings)
        ]
        persisted = _persist_embeddings(supabase, updates)
        total_persisted += persisted
        print(f"    persisted {persisted}/{len(batch)}", flush=True)

    elapsed = time.perf_counter() - t_start
    print(
        f"Done. Persisted {total_persisted}/{len(rows)} embeddings in {elapsed:.1f}s.",
        flush=True,
    )
    return 0 if total_persisted == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
