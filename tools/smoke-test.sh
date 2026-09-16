#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/env.sh"

BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT:-8011}}
MODEL=${SERVED_MODEL_NAME:-deepseek-v4.1-flash}
# Valid 1 x 1 PNG used only to verify the multimodal request path.
IMAGE_DATA_URL='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z08sAAAAASUVORK5CYII='

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
  -d "{\"model\":\"$MODEL\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"Reply with exactly: RESPONSES-OK\"}]}],\"temperature\":0,\"max_output_tokens\":128}"
printf '\n'

if [[ "${ENABLE_VISION:-1}" == "1" ]]; then
  curl --noproxy '*' --fail --silent --show-error \
    "$BASE_URL/v1/responses" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\",\"image_url\":\"$IMAGE_DATA_URL\",\"detail\":\"auto\"},{\"type\":\"input_text\",\"text\":\"Reply with exactly: IMAGE-OK\"}]}],\"temperature\":0,\"max_output_tokens\":128}"
  printf '\n'
fi
