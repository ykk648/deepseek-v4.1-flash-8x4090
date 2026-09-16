#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$PROJECT_DIR/env.sh"

python - <<'PY'
import torch
import triton
import transformers
import vllm
import flashinfer

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("triton", triton.__version__)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("flashinfer", flashinfer.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

"$CUDA_HOME/bin/nvcc" --version | tail -n 1
