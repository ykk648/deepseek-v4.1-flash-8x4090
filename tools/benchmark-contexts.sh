#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/env.sh"

BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT:-8011}}
MODEL=${SERVED_MODEL_NAME:-deepseek-v4.1-flash}
MODEL_DIR=${MODEL_DIR:-$ROOT/models/DeepSeek-V4.1-Flash}
RESULT_DIR=${RESULT_DIR:-$ROOT/results/local}
mkdir -p "$RESULT_DIR"

run_case() {
  local label=$1 input_len=$2 output_len=$3 prompts=$4 seed=$5
  vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL" \
    --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --tokenizer "$MODEL_DIR" \
    --tokenizer-mode deepseek_v41 \
    --dataset-name random \
    --input-len "$input_len" \
    --output-len "$output_len" \
    --num-prompts "$prompts" \
    --max-concurrency 1 \
    --ignore-eos \
    --temperature 0 \
    --seed "$seed" \
    --disable-tqdm \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "$label.json"
}

run_case 1k-256 1024 256 3 42
run_case 8k-512 8192 512 3 0
run_case 32k-128 32768 128 1 0
run_case 128k-128 131072 128 1 0
run_case 256k-128 261888 128 1 0
