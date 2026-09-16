#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${1:-${VENV:-$ROOT/.venv}}
PYTHON="$VENV/bin/python"

test -x "$PYTHON" || {
  echo "Python not found in virtual environment: $PYTHON" >&2
  exit 1
}
command -v patch >/dev/null || {
  echo "The 'patch' command is required." >&2
  exit 1
}

SITE_PACKAGES=$(
  "$PYTHON" -c 'import site; print(site.getsitepackages()[0])'
)
TARGET="$SITE_PACKAGES/vllm/tokenizers/deepseek_v41.py"
PATCH_FILE="$ROOT/patches/vllm-deepseek-v41-responses.patch"
PATCHED='if part_type in ("text", "input_text", "output_text"):'
ORIGINAL='if part_type == "text":'

test -f "$TARGET" || {
  echo "DeepSeek V4.1 tokenizer source not found: $TARGET" >&2
  exit 1
}

if grep -Fq "$PATCHED" "$TARGET"; then
  echo 'Codex Responses input_text patch already applied.'
elif grep -Fq "$ORIGINAL" "$TARGET"; then
  patch --batch --forward -d "$SITE_PACKAGES" -p1 -i "$PATCH_FILE"
  echo 'Applied Codex Responses input_text patch.'
else
  echo 'Unsupported deepseek_v41.py version; refusing to patch.' >&2
  exit 1
fi

"$PYTHON" - <<'PY'
from vllm.tokenizers.deepseek_v41 import _normalize_messages

messages = [{
    "role": "user",
    "content": [{"type": "input_text", "text": "CODEX-OK"}],
}]
normalized = _normalize_messages(messages)
assert normalized[0]["content"] == "CODEX-OK"
print("Codex Responses input_text normalization passed.")
PY
