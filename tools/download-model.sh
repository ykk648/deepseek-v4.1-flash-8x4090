#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$PROJECT_DIR/env.sh"

DEST="${MODEL_DIR:-$PROJECT_DIR/models/DeepSeek-V4.1-Flash}"
mkdir -p "$DEST"

echo 'The checkpoint is about 475.25 GiB. Keep at least 550 GiB free.'
echo 'Downloading from ModelScope without the browsing proxy.'
modelscope download \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --max-workers "${DOWNLOAD_WORKERS:-4}" \
  --local_dir "$DEST"

"$PROJECT_DIR/tools/verify-model.sh" "$DEST"
