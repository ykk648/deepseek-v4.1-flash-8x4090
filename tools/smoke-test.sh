#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/env.sh"

BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT:-8011}}
MODEL=${SERVED_MODEL_NAME:-deepseek-v4.1-flash}
# Valid 1 x 1 PNG used only to verify the multimodal request path.
IMAGE_DATA_URL='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z08sAAAAASUVORK5CYII='

assert_exact_output() {
  local api=$1 expected=$2
  python -c '
import json
import sys

api, expected = sys.argv[1:]
data = json.load(sys.stdin)
if api == "chat":
    actual = data["choices"][0]["message"]["content"]
else:
    actual = "".join(
        item["text"]
        for output in data["output"]
        if output.get("type") == "message"
        for item in output.get("content", [])
        if item.get("type") == "output_text"
    )
if actual != expected:
    raise SystemExit(f"{api} output mismatch: expected {expected!r}, got {actual!r}")
print(f"{api}: {actual}")
' "$api" "$expected"
}

curl --noproxy '*' --fail --silent --show-error "$BASE_URL/health"
printf '\n'
chat_response=$(curl --noproxy '*' --fail --silent --show-error \
  "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: SM89-OK\"}],\"temperature\":0,\"max_tokens\":128}")
printf '%s' "$chat_response" | assert_exact_output chat SM89-OK

responses_response=$(curl --noproxy '*' --fail --silent --show-error \
  "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"Reply with exactly: RESPONSES-OK\"}]}],\"temperature\":0,\"max_output_tokens\":128}")
printf '%s' "$responses_response" | assert_exact_output responses RESPONSES-OK

compact_response=$(curl --noproxy '*' --fail --silent --show-error \
  "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"input\":[{\"type\":\"reasoning\",\"id\":\"rs_codex_compaction\",\"summary\":[],\"encrypted_content\":\"opaque-codex-reasoning\"},{\"type\":\"compaction\",\"id\":\"cmp_codex_compaction\",\"encrypted_content\":\"opaque-codex-compaction\"},{\"type\":\"item_reference\",\"id\":\"item_codex_metadata\"},{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"Reply with exactly: COMPACT-OK\"}]}],\"temperature\":0,\"max_output_tokens\":128}")
printf '%s' "$compact_response" | assert_exact_output compaction COMPACT-OK

if [[ "${ENABLE_VISION:-1}" == "1" ]]; then
  image_response=$(curl --noproxy '*' --fail --silent --show-error \
    "$BASE_URL/v1/responses" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\",\"image_url\":\"$IMAGE_DATA_URL\",\"detail\":\"auto\"},{\"type\":\"input_image\",\"image_url\":\"$IMAGE_DATA_URL\",\"detail\":\"auto\"},{\"type\":\"input_text\",\"text\":\"Reply with exactly: IMAGE-OK\"}]}],\"temperature\":0,\"max_output_tokens\":128}")
  printf '%s' "$image_response" | assert_exact_output image IMAGE-OK
fi
