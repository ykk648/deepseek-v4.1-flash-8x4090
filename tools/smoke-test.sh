#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/env.sh"

BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT:-8011}}
MODEL=${SERVED_MODEL_NAME:-deepseek-v4.1-flash}

curl --noproxy '*' --fail --silent --show-error "$BASE_URL/health"
printf '\n'
curl --noproxy '*' --fail --silent --show-error \
  "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: SM89-OK\"}],\"temperature\":0,\"max_tokens\":32}"
printf '\n'
curl --noproxy '*' --fail --silent --show-error \
  "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"input\":\"Reply with exactly: RESPONSES-OK\",\"temperature\":0,\"max_output_tokens\":32}"
printf '\n'
