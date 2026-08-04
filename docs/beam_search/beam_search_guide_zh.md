# Beam Search 使用文档

## 目录

- [1. 快速启动和使用测试](#1-快速启动和使用测试)
  - [1.1 服务端启动](#11-服务端启动)
  - [1.2 客户端调用](#12-客户端调用)
  - [1.3 运行测试](#13-运行测试)
- [2. 功能详解](#2-功能详解)
  - [2.1 惩罚功能](#21-惩罚功能)
  - [2.2 约束解码](#22-约束解码)
  - [2.3 Cascade Attention](#23-cascade-attention)
  - [2.4 Decode-First 调度](#24-decode-first-调度)
  - [2.5 Mamba State 支持](#25-mamba-state-支持)
  - [2.6 LM Head Special Token IDs](#26-lm-head-special-token-ids)

---

## 1. 快速启动和使用测试

### 1.1 服务端启动

通过 `--enable-beam-search` 启用 beam search 采样模式：

```bash
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-beam-search \
    --trust-remote-code
```

启用 beam search 后，服务端会自动进行以下调整：
- 禁用 PD 分离 (disaggregation)
- 禁用流水线并行 (pipeline parallelism)
- 禁用 overlap schedule
- 禁用 chunked prefill
- 强制 `page_size = 1`

完整的服务端参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-beam-search` | bool | `False` | 启用 beam search 采样模式 |
| `--enable-beam-cascade-attention` | bool | `False` | 启用 cascade attention（需 flashinfer 后端） |
| `--beam-cascade-shared-step` | int | `16` | cascade attention 共享前缀长度的对齐步长 |
| `--beam-cascade-plan-lru-cap` | int | `64` | cascade attention 跨 batch plan LRU 缓存容量 |
| `--enable-decode-first` | bool | `False` | 优先 decode 而非发起新 prefill |
| `--beam-search-constraint-dict` | str | `None` | 前缀约束解码词典文件路径 |
| `--enable-beam-mamba-double-buffer` | bool | `False` | 启用双缓冲零拷贝 mamba state 剪枝（Mamba 模型专用） |
| `--lm-head-special-token-ids` | str | `None` | 限制 beam search 仅对指定 token ID 计算 LM head logits（见 [2.6](#26-lm-head-special-token-ids)） |

### 1.2 客户端调用

#### 使用 sglang Engine (本地调用)

```python
import sglang as sgl

# 启动 engine
engine = sgl.Engine(
    model_path="Qwen/Qwen3-1.7B",
    enable_beam_search=True,
    trust_remote_code=True,
)

# 设置采样参数
sampling_params = {
    "max_new_tokens": 10,       # 最大生成 token 数
    "n": 100,                      # beam width（同时等于返回序列数）
    "use_beam_search": True,     # 必须显式指定
}

outputs = engine.generate("Hello, how are you?", sampling_params=sampling_params)

# 获取 beam search 结果
beam_results = outputs["meta_info"]["beam_results"]
for i, beam in enumerate(beam_results):
    print(f"Beam {i}: {beam['text']} (score: {beam['meta_info'].get('sequence_score'))}")
```

**关键参数说明**：
- `n`：既是 beam width，也是返回的序列数量
- `use_beam_search`：必须设为 `True`
- `max_new_tokens`：每个 beam 的最大生成 token 数
- 每个 beam 的 `beam_candidates = beam_width * 4`（内部候选池为 beam width 的 4 倍，以更好地匹配 transformers 的全词表展开效果）

#### 使用 OpenAI 兼容 API

**使用 curl**:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "n": 100,
    "max_tokens": 10,
    "use_beam_search": true
  }'
```

**Chat Completion**:

```python
import openai

client = openai.Client(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello"}],
    n=100,                       # beam width
    max_tokens=10,
    extra_body={"use_beam_search": True},
)

for choice in response.choices:
    print(f"Choice {choice.index}: {choice.message.content}")
    # sequence_score 在 choice.sglext 中
```

### 1.3 运行测试

```bash
# 运行 beam search 对比测试（sglang vs HuggingFace transformers）
python test/registered/core/test_beam_search_diff.py -v

```

---

## 2. 功能详解

### 2.1 惩罚功能

Beam search 支持加性惩罚（additive penalties），通过 `sampling_params` 配置：

- **frequency_penalty**：频率惩罚
- **presence_penalty**：存在惩罚

> **注意**：`repetition_penalty` 和 `min_new_tokens` 在 beam search 模式下暂不支持。只有加性类型的 penalty 与 beam search 的 logprob 评分路径兼容。

#### 使用示例

```python
sampling_params = {
    "max_new_tokens": 10,
    "n": 100,
    "use_beam_search": True,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.3,
}
```

---

### 2.2 约束解码

Beam search 支持基于 Trie 的前缀约束解码，确保生成结果属于预定义词典集合。

#### 工作原理

从词典文件（每行一条文本）构建 Token 级前缀 Trie。解码时，每个 beam 的 logprobs 会被 mask，只允许导向词典中合法序列的 token。当 beam 到达合法序列终点时，允许生成 EOS token 以正常结束。

**词典文件格式**（每行一条记录）：
```
apple
banana
cherry
```

**支持两种加载模式**：
- **从原始词典构建**（`build_trie_from_dict`）：通过模型 tokenizer 多进程编码，首次较慢
- **从 pickle 加载**（`load_trie_from_pickle`）：跳过重新编码，秒级加载

**构造 pickle 文件**：使用 `python/tools/build_trie_pickle.py` 离线构建持久化的 trie pickle，避免每次启动服务时都需要重新编码词典：

```bash
python3 python/tools/build_trie_pickle.py \
    --model_path /path/to/model \
    --dict_path /path/to/dictionary.txt \
    --output /path/to/dictionary.trie.pkl \
    --max-tokens 3
```

其中 `--max-tokens` 为每条词典记录编码后应有的 token 数。

**环境变量**：
- `SGLANG_BEAM_SEARCH_TRIE_DICT_ENCODE_WORKERS`：编码并发进程数（默认 10）
- `SGLANG_BEAM_CONSTRAINT_DISABLE_CACHE`：设为 `1` 关闭内部张量缓存（以算力换显存）

#### 使用示例

**服务端启动**：
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-1.7B \
    --enable-beam-search \
    --beam-search-constraint-dict /path/to/dictionary.txt \
    --trust-remote-code
```

**客户端调用**：
```python
sampling_params = {
    "max_new_tokens": 10,
    "n": 100,
    "use_beam_search": True,
}
outputs = engine.generate(prompt, sampling_params=sampling_params)
# 生成的每个 beam 内容必属于词典中的某条完整序列
```

**约束限制**：
- 若合法候选数少于 beam_width，会用 dummy beam 填充；dummy beam 不参与候选排序
- 若连续生成中无可用的合法候选，请求会提前结束

---

### 2.3 Cascade Attention

Cascade attention，通过分离共享前缀和私有后缀的计算来减少 HBM 带宽消耗。

#### 适用场景

Cascade attention 适用于 **prompt 大于 1K token**、**beam width 大于 50** 的计算密集型场景。对于较小的 prompt 或较低的 beam width，两阶段拆分的额外开销可能超过带宽节省收益。

#### 工作原理

```
┌─────────────────────────────────────────────────────┐
│  Beam 0..K  ─── 共享前缀 (prompt) ──── Prefill Wrapper
│           一份 KV，K 个 Q                            │
├─────────────────────────────────────────────────────┤
│  Beam 0     ─── 私有后缀 ──── Decode Wrapper        │
│  Beam 1     ─── 私有后缀                             │
│  ...         (每 beam 行 q_len=1)                    │
├─────────────────────────────────────────────────────┤
│            merge_state(o_shared, ..., o_priv, ...)   │
└─────────────────────────────────────────────────────┘
```

在 beam search decode 中，同一请求的所有 beam 共享相同的 prompt 前缀。传统方式每个 beam 都要独立加载一份前缀 KV cache，cascade attention 改为只加载一次，在 prefill wrapper 中复用给全部 beam，再通过 decode 快路径处理各 beam 的私有后缀。

#### 限制与配置

- **必须使用 flashinfer 后端**：`--decode-attention-backend flashinfer`
- `--beam-cascade-shared-step`：共享前缀长度对齐步长（默认 16）。共享前缀长度会向下取整到 step 的整数倍，使得 prompt 长度相近的请求被归为同一组，从而跨 batch 复用预计算好的 prefill 调度计划。例如 step=16 时，prompt 长度 1023 和 1024 的请求都会对齐到 1008，共享同一份 plan。
- `--beam-cascade-plan-lru-cap`：跨 batch plan 复用的 LRU 缓存容量（默认 64，设为 0 禁用）。prefill 调度计划的生成有开销，相同长度结构的 batch 可以复用缓存的 plan。该参数控制最多缓存多少个不同的 plan，缓存满时淘汰最近最少使用的条目。

#### 使用示例

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-1.7B \
    --enable-beam-search \
    --enable-beam-cascade-attention \
    --beam-cascade-shared-step 16 \
    --beam-cascade-plan-lru-cap 64 \
    --attention-backend flashinfer \
    --decode-attention-backend flashinfer \
    --trust-remote-code
```

---

### 2.4 Decode-First 调度

Decode-first 是一种调度优化策略，当运行中的 batch 有 decode 工作时，优先执行 decode 而非发起新的 prefill。

#### 适用场景

Decode-first 调度适用于**生成 token 极少**的场景，这样运行中的 decode 工作可以快速完成，不会因优先 decode 而显著延迟下一轮 prefill。

#### 使用

```bash
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-beam-search \
    --enable-decode-first \
    --trust-remote-code
```

**注意**：`--enable-decode-first` 是独立的调度策略参数，可以与 beam search 或非 beam search 场景配合使用，但在 beam search 请求 prompt 长、decode 短的场景下收益最大。

---

### 2.5 Mamba State 支持

对于使用 Mamba / SSM（State Space Model）架构的模型（如 Qwen3.5），beam search 内置了对 recurrent state 的管理支持。

不同于标准 Transformer 模型的 token-indexed KV cache，Mamba 模型的 recurrent state 是 in-place 的 per-slot 状态。在 beam search 的展开与剪枝过程中，mamba state 会自动 fork 和 remap，确保每个 beam 分支持有正确的递归状态。无需额外配置，当模型使用混合内存池时自动生效。

#### 相关进程启动参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-beam-mamba-double-buffer` | bool | `False` | 启用双缓冲零拷贝 mamba state 剪枝。开启后 beam 剪枝时不再物理拷贝 mamba state，而是通过 COW（Copy-on-Write）映射实现零拷贝，在高 beam width 场景下可显著降低 mamba state 拷贝开销。代价是会额外占用约一倍 beam 数量的 mamba state 显存 |

**注意事项**：
- 若 mamba cache 空间不足，需增大 `--max-mamba-cache-size` 或减小 beam width / 并发数

---

### 2.6 LM Head Special Token IDs

`--lm-head-special-token-ids`：在 beam search decode 时只对指定 token ID 计算 LM head logits，把 matmul 从 `[B, H] × [H, V]` 降为 `[B, H] × [H, K]`。适用于 GenRec / SID 等输出落在小候选集上的场景。**必须与 `--enable-beam-search` 一起启用**（单独设置会直接报错）；仅 TP=1 生效；候选集需覆盖合法输出（通常含 EOS）。

**数值影响说明**：softmax / logprob 是在受限候选集 `K` 上计算，而不是全词表 `V`，因此 logprob 绝对值会与全词表 baseline 不同；累加后的 beam 分数可能改变序列排序。上线前请做开启/关闭该参数的 side-by-side diff，评估排序与质量差异是否可接受。

#### 参数用法

格式支持离散 ID、闭区间 `start:end`，以及混合写法：

```bash
--lm-head-special-token-ids 1,2,3,151643
--lm-head-special-token-ids 151669:153206
--lm-head-special-token-ids 151669:153206,151645
```

启动示例（GenRec SID 区间含 `<|sid_begin|>/<|sid_end|>` + EOS）：

```bash
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-beam-search \
    --lm-head-special-token-ids 151669:153206,151645 \
    --trust-remote-code
```

客户端仍用 `use_beam_search=True` 与 `n` 发起 beam search，无需改协议字段。

#### 实测加速（GenRec fp8，beam `n=50`）

相对全词表 baseline（同机 A/B，HTTP non-stream）：

| 并发 | RPS | 平均延迟 |
|---:|---:|---:|
| 1 | **~1.20×**（+20%） | **-17%** |
| 8 | **~1.13×**（+13%） | **-11%** |

SID 格式合法率与 beam_valid（落在 sid2vid）与全词表对齐（约 100% / 84%）。加速幅度取决于 `K/V` 与 decode 步数，不同模型/宽度会有差异。



