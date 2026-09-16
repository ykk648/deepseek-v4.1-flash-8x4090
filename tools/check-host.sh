#!/usr/bin/env bash
set -euo pipefail

echo '--- NVIDIA GPUs ---'
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version \
  --format=csv,noheader

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$gpu_count" -ne 8 ]]; then
  echo "Expected 8 GPUs, found $gpu_count" >&2
  exit 1
fi

echo '--- GPU topology ---'
nvidia-smi topo -m
echo '--- P2P read/write ---'
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
echo '--- RAM ---'
free -h
echo '--- Disk ---'
df -h "${MODEL_DIR:-.}"
