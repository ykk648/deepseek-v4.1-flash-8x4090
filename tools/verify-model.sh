#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${1:-${MODEL_DIR:-$PROJECT_DIR/models/DeepSeek-V4.1-Flash}}"

required=(config.json tokenizer.json tokenizer_config.json model.safetensors.index.json)
for file in "${required[@]}"; do
  test -s "$MODEL_DIR/$file" || {
    echo "Missing model file: $MODEL_DIR/$file" >&2
    exit 1
  }
done

index_shards=$(
  { grep -o 'model-[0-9]\{5\}-of-00048\.safetensors' \
      "$MODEL_DIR/model.safetensors.index.json" || true; } | sort -u | wc -l
)
if [[ "$index_shards" -ne 48 ]]; then
  echo "Expected 48 shards in model index, found $index_shards" >&2
  exit 1
fi

missing=0
for shard_num in $(seq 1 48); do
  printf -v shard '%05d' "$shard_num"
  file="$MODEL_DIR/model-${shard}-of-00048.safetensors"
  if ! test -s "$file"; then
    echo "Missing model shard: $file" >&2
    missing=1
  fi
done
test "$missing" -eq 0 || exit 1

echo "Model file set is complete: $MODEL_DIR"
