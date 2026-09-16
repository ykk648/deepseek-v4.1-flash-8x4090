#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

PYTHON_VERSION=${PYTHON_VERSION:-3.12}
VENV=${VENV:-"$ROOT/.venv"}
UV_CACHE_DIR=${UV_CACHE_DIR:-"$ROOT/.uv-cache"}
PYPI_INDEX_URL=${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
RELEASE_TAG=v0.28.1rc1-vision9-sm89-sm120-cu130
VLLM_WHEEL='vllm-0.28.1rc1.dev517%2Bglm53.dsv41.vision11.sm89sm120.cu130-cp312-cp312-linux_x86_64.whl'
VLLM_SHA256=e1c8313e6a8b58ec3feecaffb37fc3fda8e61ba4ecff624853b24216b7eb97ed
RELEASE_URL="https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/$RELEASE_TAG"

command -v uv >/dev/null || {
  echo 'uv is required: https://docs.astral.sh/uv/' >&2
  exit 1
}

export UV_CACHE_DIR UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}

uv venv --python "$PYTHON_VERSION" --seed "$VENV"
uv pip install --python "$VENV/bin/python" --torch-backend=cu130 \
  "$RELEASE_URL/$VLLM_WHEEL#sha256=$VLLM_SHA256" \
  'transformers==5.16.1' \
  'triton==3.7.1'
uv pip install --python "$VENV/bin/python" \
  --index-url "$PYPI_INDEX_URL" \
  'modelscope>=1.28'

"$ROOT/tools/apply-vllm-patches.sh" "$VENV"
VENV="$VENV" "$ROOT/tools/check-env.sh"
echo "Environment ready: $VENV"
