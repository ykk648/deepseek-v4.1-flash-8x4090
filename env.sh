#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$PROJECT_DIR/.env" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env"
fi

export VIRTUAL_ENV="${VENV:-$PROJECT_DIR/.venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.uv-cache}"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export MODELSCOPE_CACHE="$PROJECT_DIR/.cache/modelscope"
export FLASHINFER_WORKSPACE_BASE="$PROJECT_DIR/.cache/flashinfer-workspace"
export TORCH_EXTENSIONS_DIR="$PROJECT_DIR/.cache/torch-extensions"
export PYTORCH_KERNEL_CACHE_PATH="$PROJECT_DIR/.cache/torch/kernels"
export TRITON_CACHE_DIR="$PROJECT_DIR/.cache/triton"
export VLLM_USE_V2_MODEL_RUNNER=1

# Use the CUDA 13.0 compiler shipped in the environment. The host CUDA 12.8
# toolkit is too old for this cu130 build.
PYTHON_CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME="${CUDA_HOME:-$PYTHON_CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
if [[ -d "$CUDA_HOME/lib" ]]; then
  export CUDA_LIB_DIR="$CUDA_HOME/lib"
elif [[ -d "$CUDA_HOME/lib64" ]]; then
  export CUDA_LIB_DIR="$CUDA_HOME/lib64"
else
  echo "CUDA library directory not found under $CUDA_HOME" >&2
  return 1 2>/dev/null || exit 1
fi
export LIBRARY_PATH="$CUDA_LIB_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FLASHINFER_CUDA_ARCH_LIST=8.9
export TORCH_CUDA_ARCH_LIST=8.9

# This host has dual-stack loopback and two NICs. Pin collectives to one IPv4
# interface so Gloo does not mix AF_INET and AF_INET6 worker addresses.
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

# Keep model and dependency traffic away from the browsing-only proxy.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
