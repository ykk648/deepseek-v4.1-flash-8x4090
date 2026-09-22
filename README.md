# 8 x RTX 4090 48GB 部署 DeepSeek-V4.1-Flash：256K 实测与优化

> 在无 NVLink、无 CUDA P2P、双 NUMA 的单机 8 卡环境中，记录
> DeepSeek-V4.1-Flash 的可复现部署、失败条件、参数 A/B 和长上下文性能。

最后更新：2026-09-22

## 结论

本机最终采用：

以下性能表记录 9 月 16 日单请求低延迟档；9 月 17 日的本机共享档改为
`MAX_NUM_SEQS=3`，并发对比见下节。公开模板仍默认单请求。

```text
deepseek-ai/DeepSeek-V4.1-Flash 官方 checkpoint
+ yhfgyyf/vllm-deepseek-v4-sm89 Vision11 wheel
+ TP8 / EP8 / FP8 KV / Engram CPU offload
+ DSpark 5 adaptive verification
+ CED prefill
+ prefix caching
+ two-image input for Codex / Responses API
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
| 单请求图片上限 | 2 | 2（可配置） |
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

## 多用户并发

2026-09-17 在保留图片输入、256K、DSpark 5 adaptive、CED、4096 batch tokens 和
0.92 显存利用率的情况下，将服务端调度上限临时设为 4，实测客户端 1/2/3/4 路。
每档 8 个约 8K 输入 / 512 输出请求，temperature=0、ignore EOS，64 个正式对比
请求均成功。未测试更高并发或持续数小时的压测。

| 客户端并发 | 新输入总输出 tok/s | 新输入平均 TTFT | 重复输入总输出 tok/s | 重复输入每路 decode tok/s |
|---:|---:|---:|---:|---:|
| 1 | 96.3 | 1.37 s | 132.8 | 137.4 |
| 2 | 122.7 | 1.82 s | 182.9 | 95.9 |
| 3 | 141.2 | 2.47 s | 230.8 | 85.9 |
| 4 | 165.8 | 2.76 s | 240.9 | 73.7 |

总输出吞吐包含 prefill，不能与每路 `1000 / mean TPOT(ms)` 的 decode 指标混用。
重复输入组使用同一 seed=123、同一批 8 个 prompt，先完整预热后再逐档测量。
新输入组使用不同 seed=457/458/459/460 避免跨档缓存复用，但不同输入和调度也会
改变 DSpark 接受率；单轮小样本不代表真实代码或图片负载的稳定吞吐。

本机选择 **3 路共享档**：重复输入下已经达到 4 路约 96% 的总吞吐，每人 decode
更快。新输入组则是 **4 路吞吐最高**，若主要追求总处理量且允许更长等待，可选 4。
2 路更偏重交互延迟。仅一个请求时不会为了凑满 batch 而等待其他用户。

最终 3 路服务再次测 8K/512：单路总输出 96.9 tok/s、纯 decode 125.6 tok/s；
3 路总输出 162.3 tok/s。16 次正式请求全部成功。启用双图后的 3 路服务 KV 容量为
1,037,389 tokens，256K 长度保持不变；这是容量评估，不是 3 路完整 256K 实测。
未对服务端上限 1 与上限 3 做同负载严格 A/B，不应据此宣称单路性能完全无损。

补充 32K/256（服务端上限 3，每档 6 请求，不同随机 seed）：

| 客户端并发 | 总输出 tok/s | 平均 TTFT | 每路 decode tok/s |
|---:|---:|---:|---:|
| 1 | 44.1 | 3.59 s | 114.8 |
| 2 | 41.7 | 5.99 s | 41.0 |
| 3 | 51.5 | 4.98 s | 26.4 |

18 次请求全部成功。多个长输入 prefill 会暂停其他请求的 decode，平均 TPOT 包含
这些暂停，所以每路速度会大幅下降。长输入短输出中，并发收益并不稳定，不能将
8K 的推荐直接外推到 32K/128K/256K；频繁提交大段新上下文且重视个人流畅度时，
更适合减少并发、允许排队。独立就绪探测可能预热首个 prompt，因此补测组不是
严格的全冷缓存比较。Responses 双图输入也在最终 3 路服务下重新验证通过。

```dotenv
MAX_NUM_SEQS=3
```

此参数修改后须重启服务才生效。调度上限不是完整 256K 请求的容量保证：4 路配置
启动时 KV 容量为 828,104 tokens（约 3.16 个 256K 请求），不能保证 4 个完整 256K
请求同时驻留。超过调度上限会排队，超过 KV 容量可能抢占重算。

复现前先把服务端 `MAX_NUM_SEQS` 设为至少 4，并确保没有其他用户请求干扰：

```bash
bash tools/benchmark-concurrency.sh
```

脚本另含 8 个预热请求，每次 vLLM bench 还会发送一次独立端点探测。原始结果保存在
`results/local/concurrency/`，精简汇总见
[`results/concurrency-summary.json`](results/concurrency-summary.json)。

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

该脚本验证 `/health`、`/v1/chat/completions`、文本 `/v1/responses` 和双图
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
- Responses API 的 `input_image` 双图输入
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

默认允许每个 prompt 两张图片，与上游 8 x 4090 示例一致：

```dotenv
ENABLE_VISION=1
MAX_IMAGES_PER_PROMPT=2
```

这会传入 `--limit-mm-per-prompt '{"image":2}'`。Codex / Responses API 使用：

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
一个 prefill chunk 内处理；当前 `MAX_NUM_BATCHED_TOKENS=4096` 可覆盖模型默认每张
最多 1024 image tokens 的双图路径。

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
