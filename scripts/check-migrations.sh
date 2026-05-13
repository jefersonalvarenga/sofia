#!/usr/bin/env bash
# check-migrations.sh — fails if new migration files add tables not yet in prod
# Usage: ./scripts/check-migrations.sh [base-ref]
# Requires: SUPABASE_URL, SUPABASE_KEY (service role key)

set -euo pipefail

BASE_REF="${1:-origin/main}"
MIGRATIONS_DIR="supabase/migrations"

if [ -z "${SUPABASE_URL:-}" ] || [ -z "${SUPABASE_KEY:-}" ]; then
  echo "SUPABASE_URL and SUPABASE_KEY must be set."
  exit 1
fi

git fetch origin main --quiet 2>/dev/null || true

ADDED_FILES=$(git diff --name-only --diff-filter=A "${BASE_REF}...HEAD" \
  -- "${MIGRATIONS_DIR}/*.sql" 2>/dev/null || \
  git diff --name-only --diff-filter=A "${BASE_REF}" HEAD \
  -- "${MIGRATIONS_DIR}/*.sql")

if [ -z "$ADDED_FILES" ]; then
  echo "No new migration files detected. OK."
  exit 0
fi

echo "New migration files in this PR:"
echo "$ADDED_FILES"
echo ""

# Extract CREATE TABLE names (handles IF NOT EXISTS, schema prefix, quoted names)
TABLES=$(echo "$ADDED_FILES" | xargs grep -ihE \
  'CREATE[[:space:]]+TABLE[[:space:]]+(IF[[:space:]]+NOT[[:space:]]+EXISTS[[:space:]]+)?(public\.)?["`]?[a-z_][a-z0-9_]*["`]?' \
  | grep -oiE '(public\.)?`?"?[a-z_][a-z0-9_]*`?"?[[:space:]]*\(' \
  | sed -E 's/["`[:space:]()]+//g; s/^public\.//' \
  | sort -u)

if [ -z "$TABLES" ]; then
  echo "No CREATE TABLE statements found in new migrations. Marking as safe."
  exit 0
fi

echo "Tables to verify in Supabase prod:"
echo "$TABLES"
echo ""

MISSING=()
for TABLE in $TABLES; do
  HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" \
    -H "apikey: ${SUPABASE_KEY}" \
    -H "Authorization: Bearer ${SUPABASE_KEY}" \
    "${SUPABASE_URL}/rest/v1/${TABLE}?limit=0" || echo "000")

  if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "206" ]; then
    echo "  OK      ${TABLE}"
  else
    echo "  MISSING ${TABLE}  (HTTP ${HTTP_CODE})"
    MISSING+=("$TABLE")
  fi
done

echo ""
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "FAIL: The following tables are missing from Supabase prod:"
  printf '  - %s\n' "${MISSING[@]}"
  echo ""
  echo "Apply the pending migration(s) to Supabase prod BEFORE merging this PR."
  exit 1
fi

echo "All expected tables verified in prod. OK."
