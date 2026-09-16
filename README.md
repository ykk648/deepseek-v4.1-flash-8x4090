# 8 x RTX 4090 48GB 部署 DeepSeek-V4.1-Flash：256K 实测与优化

> 在无 NVLink、无 CUDA P2P、双 NUMA 的单机 8 卡环境中，记录
> DeepSeek-V4.1-Flash 的可复现部署、失败条件、参数 A/B 和长上下文性能。

最后更新：2026-09-16

## 结论

本机最终采用：

```text
deepseek-ai/DeepSeek-V4.1-Flash 官方 checkpoint
+ yhfgyyf/vllm-deepseek-v4-sm89 Vision11 wheel
+ TP8 / EP8 / FP8 KV / Engram CPU offload
+ DSpark 5 adaptive verification
+ CED prefill
+ prefix caching
+ single-image input for Codex / Responses API
+ vLLM 默认 FULL_AND_PIECEWISE CUDA Graph
+ 256K context / 单请求低延迟
```

初始保守配置的单流 decode 约为 **11 token/s**。最终配置在随机输入基准中达到
**120-149 token/s decode**，接近 256K 的 261,888-token 请求完成且未 OOM。

这不是新的推理引擎或 kernel fork。模型支持、SM89 kernel、DSpark、CED 和 adaptive
verification 均来自
[yhfgyyf/vllm-deepseek-v4-sm89](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89)。
本仓库提供的是 8 x 48GB RTX 4090、无 P2P、256K 单用户场景的部署封装、参数取舍、
失败复现和同机实测。

## 本仓库补充了什么

上游当前 8 x 4090 示例面向更长 context、最多 4 条序列，并显式使用纯 `FULL`
CUDA Graph。本机实测后的差异如下：

| 项目 | 上游 8 x 4090 示例 | 本机最终配置 |
|---|---|---|
| 目标 | 最长 context、最多 4 sequences | 256K、单请求 Codex 低延迟 |
| CUDA Graph | 显式 `FULL` | 默认 `FULL_AND_PIECEWISE` |
| 显存利用率 | 0.98 | 0.92 |
| checkpoint 加载 | prefetch / 2 threads | lazy |
| 单请求图片上限 | 2 | 1（可配置） |
| P2P | 未限定 | 全部不可用，显式禁用 custom all-reduce |
| 验证 | 通用启动配置 | 1K-256K、prefix cache、Responses API、passkey |

最重要的兼容性结论：显式 `cudagraph_mode=FULL` 在本机 warmup 阶段稳定触发
Triton `CUDA illegal memory access`。移除显式 FULL 后，vLLM 默认采用
`FULL_AND_PIECEWISE`，decode 仍使用图捕获，同时 DSpark 5 adaptive、CED 和
256K 请求均能稳定运行。

## 最终性能

以下为 `vllm bench serve` 单并发随机输入，temperature 0、ignore EOS。Decode
由 `1000 / mean TPOT(ms)` 计算。CED 开启后并非完整 prefill 计算，因此下表的
prefill 是 `input tokens / TTFT` 代理指标，不能当作精确 FLOPS 吞吐。

| 输入 / 输出 | TTFT | Prefill 代理 | Decode |
|---:|---:|---:|---:|
| 1K / 256 | 0.20 s | 5,175 tok/s | 120.5 tok/s |
| 8K / 512 | 1.10 s | 7,453 tok/s | 149.3 tok/s |
| 32K / 128 | 3.77 s | 8,710 tok/s | 127.8 tok/s |
| 128K / 128 | 18.10 s | 7,241 tok/s | 136.5 tok/s |
| 261,888 / 128 | 56.14 s | 4,665 tok/s | 137.6 tok/s |

随机输入会改变 DSpark 接受率，因此 decode 有明显波动。8K / 512 的完整请求输出
吞吐为 **113.1 token/s**；它包含 prefill，不等于纯 decode。

### 相比初始配置

| 输入 | 初始 TTFT | 最终 TTFT | 初始 decode | 最终 decode |
|---:|---:|---:|---:|---:|
| 32K | 8.81 s | 3.77 s | 10.98 tok/s | 127.8 tok/s |
| 128K | 43.10 s | 18.10 s | 11.13 tok/s | 136.5 tok/s |
| 约 256K | 110.03 s | 56.14 s | 11.28 tok/s | 137.6 tok/s |

### Prefix cache

相同 32K prompt 连续请求：

```text
首次 TTFT：4.40 s
重复 TTFT：0.88 s
```

这对保留大段共同前缀的 Codex 连续对话比提高 batch token 预算更有效。

机器可读的汇总位于
[`results/benchmark-summary.json`](results/benchmark-summary.json)。

## 测试机器

| 项目 | 配置 |
|---|---|
| GPU | 8 x NVIDIA GeForce RTX 4090 48GB |
| 单卡显存 | 49,140 MiB |
| 架构 | Ada Lovelace / SM89 / Compute Capability 8.9 |
| GPU 互联 | 无 NVLink；CUDA P2P read/write 全部不可用 |
| CPU | 2 x Intel Xeon Gold 6462C，128 logical CPUs |
| NUMA | GPU 0-3 在 NUMA 0；GPU 4-7 在 NUMA 1 |
| RAM | 1 TiB，无 swap |
| NVIDIA driver | 580.178.04 |
| 文件系统 | XFS |

这套结果不适用于普通 24GB RTX 4090。Engram CPU offload 每个 rank 约占用
11.8 GiB pinned host memory；475.25 GiB checkpoint 的加载与页缓存也需要大量主机
内存。本机配置不能只按总显存估算。

模型的 64 个 attention heads 和 5120 hidden size 都不能被 7 整除，所以即使只有
7 张卡空闲，也不能使用 TP7。这里必须使用 TP8。

## 版本

| 组件 | 版本 |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4.1-Flash`，48 shards，475.25 GiB |
| vLLM | `0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130` |
| FlashInfer | `0.6.18+glm53.dsv41.vision2.sm89sm120.cu130.pt213` |
| Transformers | `5.16.1` |
| Triton | `3.7.1` |
| CUDA ABI | 13.0 |
| 上游 main | `678d8f531e5e498e4060b21df254d27467e0cef9` |
| wheel source | `86d35745c8797c9d2524877fb235084a6a5f5a7d` |

## 快速开始

### 1. 检查主机

```bash
./tools/check-host.sh
```

确认有 8 张 48GB RTX 4090，并查看 `nvidia-smi topo -m` 与 P2P 结果。其他拓扑
不应直接照搬 `NCCL_P2P_DISABLE=1`。

### 2. 安装环境

需要预先安装 [uv](https://docs.astral.sh/uv/) 和可用的 NVIDIA driver：

```bash
./setup_env.sh
```

脚本安装上游 Vision11 wheel，并通过 URL 中的 SHA256 校验。PyPI 依赖默认走清华
镜像；约 236 MiB 的专用 vLLM wheel 仍来自上游 GitHub Release，不应使用仅供网页
访问的小流量代理。可提前下载到本地后按需修改 `setup_env.sh`。

CUDA 13 编译器默认使用虚拟环境中 wheel 安装的
`nvidia/cu13`。若主机已有兼容 CUDA 13 toolkit，可在 `.env` 中覆盖 `CUDA_HOME`。

### 3. 下载模型

```bash
cp .env.example .env
# 修改 MODEL_DIR，确认目标磁盘至少有约 550 GiB 可用空间。
./tools/download-model.sh
```

模型通过 ModelScope 下载，不经过网页代理。下载完成后脚本会核对 index、tokenizer
和全部 48 个 safetensors 分片。当前公开脚本验证文件集合完整性；ModelScope 自身
负责下载校验，本仓库没有为 475 GiB checkpoint 额外发布一套 SHA256 清单。

### 4. 启动

```bash
./launch.sh
```

默认只监听 `127.0.0.1:8011`，模型名为 `deepseek-v4.1-flash`。首次遇到新的 shape
时 FlashInfer、Triton 或 TileLang 会进行 JIT，第一次请求可能出现延迟尖峰。

检查服务：

```bash
./tools/smoke-test.sh
```

该脚本验证 `/health`、`/v1/chat/completions`、文本 `/v1/responses` 和单图
`/v1/responses`。

### 5. systemd user service

```bash
./tools/install-service.sh
systemctl --user start deepseek-v4.1-flash.service
systemctl --user status deepseek-v4.1-flash.service
journalctl --user -u deepseek-v4.1-flash.service -f
```

安装脚本会把当前仓库绝对路径写入用户 unit，不要求仓库位于固定目录。

## 复现实测

完整 context sweep 会运行 1K、8K、32K、128K 和接近 256K 的请求，耗时较长：

```bash
./tools/benchmark-contexts.sh
```

结果默认保存到 `results/local/`，该目录不应直接作为跨机器对比依据。不同模型版本、
随机输入、DSpark 接受率、GPU 时钟和 prefix cache 状态都会改变数字。

长文本检索检查：

```bash
./tools/passkey-test.py \
  --model-dir /data/models/DeepSeek-V4.1-Flash \
  --tokens 32000
```

本机 32K CED passkey 测试返回了正确的 `73918426`。这只验证明确事实检索，不代表
CED 对代码、推理和所有长上下文任务都质量无损。

## 参数 A/B

### DSpark 3 与 DSpark 5 adaptive

DSpark 3 已能达到上游 Issue #108 报告的百 token/s 级别。相同 8K / 512 随机负载
下，DSpark 5 adaptive 的 mean TPOT 为 6.70 ms，DSpark 3 为 6.93 ms，约快 3.3%。
提升不大，但 DSpark 5 adaptive 与当前上游推荐一致，因此作为默认配置。

### CED

开启 CED 后，本机 32K TTFT 从 9.25 s 降至 4.91 s，128K 从 43.10 s 降至
22.60 s，约 256K 从 110.03 s 降至 57.16 s。加入 DSpark 5 adaptive 后的最终
TTFT 进一步见上表。

CED 是实验性近似 prefill 路径。不要把输入 token / TTFT 与完整 attention prefill
吞吐混为一谈，也不要仅凭 passkey 测试断言质量等价。

### Batch token 预算

CED + DSpark 3 下，32K TTFT：

```text
4096 batch tokens：4.91 s
8192 batch tokens：5.00 s
```

8192 没有收益，还增加 workspace 压力，因此保留 4096。

### 纯 FULL CUDA Graph

显式加入：

```text
--compilation-config '{"cudagraph_mode":"FULL"}'
```

在本机两次启动都于 `compile_or_warm_up_model()` 阶段触发 Triton CUDA illegal
memory access。默认 `FULL_AND_PIECEWISE` 稳定，并完成 decode graph capture。这个
结论是本机兼容性 workaround，不代表纯 FULL 在所有 8 x 4090 机器上不可用。

## API 与安全

已验证：

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- Responses API 的 `input_image` 单图输入
- DeepSeek V4.1 reasoning parser
- 自动工具调用 parser

公开模板不包含 API key 校验，因此默认只监听 loopback。不要直接绑定公网。需要
局域网或公网访问时，应在 Caddy/Nginx 等反向代理层增加 TLS、认证、限流和访问控制。

Codex 使用 Responses API 的结构化内容块：

```json
{"type": "input_text", "text": "..."}
```

当前上游 V4.1 tokenizer 只接受同义的 `text`，会返回 `400`。本仓库通过
[`patches/vllm-deepseek-v41-responses.patch`](patches/vllm-deepseek-v41-responses.patch)
补齐 `input_text` / `output_text` 归一化，`setup_env.sh` 会自动应用并做幂等检查。
`tools/smoke-test.sh` 使用 Codex 同款结构化请求验证 `/v1/responses`。

默认允许每个 prompt 一张图片：

```dotenv
ENABLE_VISION=1
MAX_IMAGES_PER_PROMPT=1
```

这会传入 `--limit-mm-per-prompt '{"image":1}'`。Codex / Responses API 使用：

```json
{
  "role": "user",
  "content": [
    {"type": "input_image", "image_url": "data:image/png;base64,...", "detail": "auto"},
    {"type": "input_text", "text": "描述这张图片"}
  ]
}
```

设置 `ENABLE_VISION=0` 会恢复 `--language-model-only` 纯文本模式。每张图片必须能在
一个 prefill chunk 内处理；当前 `MAX_NUM_BATCHED_TOKENS=4096` 满足模型默认最多
1024 image tokens 的单图路径。

## 项目结构

```text
.
├── .env.example
├── deepseek-v4.1-flash.service
├── env.sh
├── launch.sh
├── patches/
│   └── vllm-deepseek-v41-responses.patch
├── setup_env.sh
├── results/
│   └── benchmark-summary.json
└── tools/
    ├── benchmark-contexts.sh
    ├── apply-vllm-patches.sh
    ├── check-env.sh
    ├── check-host.sh
    ├── download-model.sh
    ├── install-service.sh
    ├── passkey-test.py
    ├── smoke-test-kernels.py
    ├── smoke-test.sh
    └── verify-model.sh
```

## 相关项目

- [yhfgyyf/vllm-deepseek-v4-sm89](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89)：本仓库使用的核心 vLLM/FlashInfer 实现。
- [deepseek-ai/DeepSeek-V4.1-Flash](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash)：模型 checkpoint。
- [ykk648/deepseek-v4-flash-8x4090](https://github.com/ykk648/deepseek-v4-flash-8x4090)：上一代 V4-Flash 24GB 卡部署与引擎对比。

## License

本仓库的部署脚本和文档使用 MIT License。模型、vLLM fork、FlashInfer 及其他依赖
分别遵循各自许可证；本仓库不重新分发模型权重或第三方 wheel。
