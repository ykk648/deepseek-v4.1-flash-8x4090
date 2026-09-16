#!/usr/bin/env python3
import torch

from flashinfer.mla import (
    deepseek_v41_mixed_sparse_workspace_size,
    deepseek_v41_mixed_sparse_workspace_size_upper_bound,
)
from flashinfer.mla.deepseek_v41 import (
    _deepseek_v41_mixed_sparse_attention_with_inv_rope,
)
from vllm.models.deepseek_v4_1.nvidia.router import deepseek_v41_topk


def main() -> None:
    torch.manual_seed(0)
    logits = torch.randn(8, 384, device="cuda", dtype=torch.float32)
    bias = torch.randn(384, device="cuda", dtype=torch.float32) * 0.01

    weights, ids = deepseek_v41_topk(
        logits,
        bias,
        torch.int32,
        1.5,
        topk=6,
    )
    scores = torch.sqrt(torch.nn.functional.softplus(logits))
    expected_ids = torch.topk(scores + bias, 6, dim=-1).indices
    expected_weights = torch.gather(scores, 1, expected_ids)
    expected_weights *= 1.5 / expected_weights.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=2e-5, atol=2e-5)

    assert callable(_deepseek_v41_mixed_sparse_attention_with_inv_rope)
    workspace = deepseek_v41_mixed_sparse_workspace_size(1, 8, 128, 512)
    workspace_max = deepseek_v41_mixed_sparse_workspace_size_upper_bound(
        2048, 8, 128, 128, 512
    )
    print("DeepSeek V4.1 SM89 top-k passed")
    print("FlashInfer sparse MLA API passed", workspace, workspace_max)


if __name__ == "__main__":
    main()
