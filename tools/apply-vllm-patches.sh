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

REJECTION_TARGET="$SITE_PACKAGES/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
REJECTION_PATCH="$ROOT/patches/vllm-adaptive-verification-bounds.patch"
REJECTION_PATCHED='if logit_idx + 1 >= tl.num_programs(0):'

test -f "$REJECTION_TARGET" || {
  echo "Speculative rejection sampler source not found: $REJECTION_TARGET" >&2
  exit 1
}

if grep -Fq "$REJECTION_PATCHED" "$REJECTION_TARGET"; then
  echo 'Adaptive verification bounds patch already applied.'
elif grep -Fq 'def _compute_local_residual_mass_kernel(' "$REJECTION_TARGET"; then
  patch --batch --forward -d "$SITE_PACKAGES" -p1 -i "$REJECTION_PATCH"
  echo 'Applied adaptive verification bounds patch.'
else
  echo 'Unsupported rejection_sampler_utils.py version; refusing to patch.' >&2
  exit 1
fi

RESPONSES_UTILS_TARGET="$SITE_PACKAGES/vllm/entrypoints/openai/responses/utils.py"
RESPONSES_UTILS_PATCH="$ROOT/patches/vllm-responses-encrypted-reasoning.patch"
RESPONSES_UTILS_PATCHED='Ignoring unsupported encrypted reasoning item %s.'
RESPONSES_COMPACTION_PATCH="$ROOT/patches/vllm-responses-compaction-item.patch"
RESPONSES_COMPACTION_PATCHED='Ignoring unsupported encrypted compaction item'
RESPONSES_ROLELESS_PATCH="$ROOT/patches/vllm-responses-roleless-items.patch"
RESPONSES_ROLELESS_PATCHED='Ignoring unsupported roleless Responses item type'

test -f "$RESPONSES_UTILS_TARGET" || {
  echo "Responses API utils source not found: $RESPONSES_UTILS_TARGET" >&2
  exit 1
}

if grep -Fq "$RESPONSES_UTILS_PATCHED" "$RESPONSES_UTILS_TARGET"; then
  echo 'Codex encrypted reasoning patch already applied.'
elif grep -Fq 'raise ValueError("Encrypted content is not supported.")' "$RESPONSES_UTILS_TARGET"; then
  patch --batch --forward -d "$SITE_PACKAGES" -p1 -i "$RESPONSES_UTILS_PATCH"
  echo 'Applied Codex encrypted reasoning patch.'
else
  echo 'Unsupported Responses API utils version; refusing to patch.' >&2
  exit 1
fi

if grep -Fq "$RESPONSES_COMPACTION_PATCHED" "$RESPONSES_UTILS_TARGET"; then
  echo 'Codex compaction item patch already applied.'
elif grep -Fq 'item.get("type") == "function_call_output"' "$RESPONSES_UTILS_TARGET"; then
  patch --batch --forward -d "$SITE_PACKAGES" -p1 -i "$RESPONSES_COMPACTION_PATCH"
  echo 'Applied Codex compaction item patch.'
else
  echo 'Unsupported Responses API utils version; refusing to patch compaction items.' >&2
  exit 1
fi

if grep -Fq "$RESPONSES_ROLELESS_PATCHED" "$RESPONSES_UTILS_TARGET"; then
  echo 'Codex roleless Responses item patch already applied.'
elif grep -Fq "$RESPONSES_COMPACTION_PATCHED" "$RESPONSES_UTILS_TARGET"; then
  patch --batch --forward -d "$SITE_PACKAGES" -p1 -i "$RESPONSES_ROLELESS_PATCH"
  echo 'Applied Codex roleless Responses item patch.'
else
  echo 'Unsupported Responses API utils version; refusing to patch roleless items.' >&2
  exit 1
fi

"$PYTHON" - <<'PY'
from openai.types.responses import ResponseReasoningItem
from vllm.entrypoints.openai.responses.utils import (
    _construct_message_from_response_item,
)
from vllm.tokenizers.deepseek_v41 import _normalize_messages

messages = [{
    "role": "user",
    "content": [{"type": "input_text", "text": "CODEX-OK"}],
}]
normalized = _normalize_messages(messages)
assert normalized[0]["content"] == "CODEX-OK"
print("Codex Responses input_text normalization passed.")

encrypted_reasoning = ResponseReasoningItem(
    id="rs_codex_compaction",
    summary=[],
    type="reasoning",
    encrypted_content="opaque-codex-reasoning",
)
assert _construct_message_from_response_item(encrypted_reasoning) is None
print("Codex encrypted reasoning normalization passed.")

encrypted_compaction = {
    "type": "compaction",
    "id": "cmp_codex_compaction",
    "encrypted_content": "opaque-codex-compaction",
}
assert _construct_message_from_response_item(encrypted_compaction) is None
print("Codex encrypted compaction normalization passed.")

roleless_item = {
    "type": "item_reference",
    "id": "item_codex_metadata",
}
assert _construct_message_from_response_item(roleless_item) is None
print("Codex roleless Responses item normalization passed.")
PY

grep -Fq "$REJECTION_PATCHED" "$REJECTION_TARGET"
echo 'Adaptive verification bounds check passed.'
