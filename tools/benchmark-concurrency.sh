#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/env.sh"

BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT:-8011}}
MODEL=${SERVED_MODEL_NAME:-deepseek-v4.1-flash}
MODEL_DIR=${MODEL_DIR:-$ROOT/models/DeepSeek-V4.1-Flash}
RESULT_DIR=${RESULT_DIR:-$ROOT/results/local/concurrency}
INPUT_LEN=${INPUT_LEN:-8192}
OUTPUT_LEN=${OUTPUT_LEN:-512}
NUM_PROMPTS=${NUM_PROMPTS:-8}
CASE_MODE=${CASE_MODE:-all}
CONCURRENCIES=${CONCURRENCIES:-"1 2 3 4"}
WARMUP_CONCURRENCY=${WARMUP_CONCURRENCY:-4}
if [[ "$CASE_MODE" != "all" && "$CASE_MODE" != "cold" && "$CASE_MODE" != "warm" ]]; then
  echo "CASE_MODE must be all, cold, or warm" >&2
  exit 2
fi
for concurrency in $CONCURRENCIES "$WARMUP_CONCURRENCY"; do
  if [[ ! "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
    echo "Concurrency values must be positive integers" >&2
    exit 2
  fi
done
mkdir -p "$RESULT_DIR"

run_case() {
  local label=$1 concurrency=$2 seed=$3
  vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL" \
    --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --tokenizer "$MODEL_DIR" \
    --tokenizer-mode deepseek_v41 \
    --dataset-name random \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$concurrency" \
    --ignore-eos --temperature 0 --seed "$seed" --disable-tqdm \
    --ready-check-timeout-sec 180 \
    --save-result --result-dir "$RESULT_DIR" \
    --result-filename "$label.json" \
    2>&1 | tee "$RESULT_DIR/$label.log"
  python - "$RESULT_DIR/$label.json" "$NUM_PROMPTS" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    result = json.load(handle)
if result.get("failed", 0) or result.get("completed") != int(sys.argv[2]):
    raise SystemExit("Benchmark did not complete every request successfully")
PY
}

# Identical seeded prompts and counts keep the warm-cache comparison matched.
if [[ "$CASE_MODE" != "cold" ]]; then
  run_case warmup "$WARMUP_CONCURRENCY" 123
  for concurrency in $CONCURRENCIES; do
    run_case "warm-c$concurrency" "$concurrency" 123
  done
fi

# Distinct seeds avoid cross-case prefix reuse, but can change DSpark acceptance.
if [[ "$CASE_MODE" != "warm" ]]; then
  for concurrency in $CONCURRENCIES; do
    run_case "cold-c$concurrency" "$concurrency" "$((456 + concurrency))"
  done
fi
