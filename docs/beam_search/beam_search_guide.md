# Beam Search Usage Guide

## Table of Contents

- [1. Quick Start and Testing](#1-quick-start-and-testing)
  - [1.1 Server Launch](#11-server-launch)
  - [1.2 Client Usage](#12-client-usage)
  - [1.3 Running Tests](#13-running-tests)
- [2. Feature Details](#2-feature-details)
  - [2.1 Penalty Support](#21-penalty-support)
  - [2.2 Constrained Decoding](#22-constrained-decoding)
  - [2.3 Cascade Attention](#23-cascade-attention)
  - [2.4 Decode-First Scheduling](#24-decode-first-scheduling)
  - [2.5 Mamba State Support](#25-mamba-state-support)
  - [2.6 Other Features](#26-other-features)

---

## 1. Quick Start and Testing

### 1.1 Server Launch

Enable beam search sampling mode via `--enable-beam-search`:

```bash
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-beam-search \
    --trust-remote-code
```

When beam search is enabled, the server automatically applies the following adjustments:
- Disables PD separation (disaggregation)
- Disables pipeline parallelism
- Disables overlap schedule
- Disables chunked prefill
- Forces `page_size = 1`

Full server arguments:

| Argument | Type | Default | Description |
|------|------|--------|------|
| `--enable-beam-search` | bool | `False` | Enable beam search sampling mode |
| `--enable-beam-cascade-attention` | bool | `False` | Enable cascade attention (requires flashinfer backend) |
| `--beam-cascade-shared-step` | int | `16` | Alignment step for cascade attention shared-prefix length |
| `--beam-cascade-plan-lru-cap` | int | `64` | Max entries in the cross-batch cascade plan LRU cache |
| `--enable-decode-first` | bool | `False` | Prioritize decode over launching new prefill batches |
| `--beam-search-constraint-dict` | str | `None` | Path to dictionary file for prefix-constrained decoding |
| `--enable-beam-mamba-double-buffer` | bool | `False` | Enable zero-copy double-buffered mamba state pruning (Mamba models only) |

### 1.2 Client Usage

#### Using sglang Engine (Local)

```python
import sglang as sgl

# Launch engine
engine = sgl.Engine(
    model_path="Qwen/Qwen3-1.7B",
    enable_beam_search=True,
    trust_remote_code=True,
)

# Configure sampling parameters
sampling_params = {
    "max_new_tokens": 10,       # Maximum generated tokens
    "n": 100,                      # Beam width (also equals number of returned sequences)
    "use_beam_search": True,     # Must be explicitly set
}

outputs = engine.generate("Hello, how are you?", sampling_params=sampling_params)

# Retrieve beam search results
beam_results = outputs["meta_info"]["beam_results"]
for i, beam in enumerate(beam_results):
    print(f"Beam {i}: {beam['text']} (score: {beam['meta_info'].get('sequence_score'))}")
```

**Key parameter notes**:
- `n`: Serves as both the beam width and the number of returned sequences
- `use_beam_search`: Must be set to `True`
- `max_new_tokens`: Maximum tokens to generate per beam
- Each beam internally uses `beam_candidates = beam_width * 4` (the candidate pool is 4x the beam width to better match transformers' full-vocabulary beam expansion)

#### Using OpenAI-Compatible API

**Using curl**:

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
    # sequence_score is in choice.sglext
```

### 1.3 Running Tests

```bash
# Run beam search comparison test (sglang vs HuggingFace transformers)
python test/registered/core/test_beam_search_diff.py -v

# Run beam search scheduler unit tests
python test/registered/scheduler/test_schedule_batch_beam_search_mixin.py -v

# Run beam search processor unit tests
python test/registered/scheduler/test_scheduler_beam_search_processor_mixin.py -v
```

---

## 2. Feature Details

### 2.1 Penalty Support

Beam search supports additive penalties, configured via `sampling_params`:

- **frequency_penalty**: Frequency-based penalty
- **presence_penalty**: Presence-based penalty

> **Note**: `repetition_penalty` and `min_new_tokens` are not currently supported in beam search mode. Only additive-type penalties are compatible with the beam search logprob-based scoring path.

#### Usage Example

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

### 2.2 Constrained Decoding

Beam search supports Trie-based prefix-constrained decoding, ensuring generated outputs belong to a predefined dictionary set.

#### How It Works

A **token-level prefix Trie** is built from a dictionary file (one entry per line). During decoding, each beam's logprobs are masked so that only tokens leading to valid dictionary entries are allowed. When a beam reaches a valid sequence endpoint, the EOS token is permitted for normal termination.

**Dictionary file format** (one entry per line):
```
apple
banana
cherry
```

**Two loading modes**:
- **Build from raw dictionary** (`build_trie_from_dict`): multi-process encoding via the model tokenizer; slower first time
- **Load from pickle** (`load_trie_from_pickle`): skips re-encoding, loads in seconds

**Building a pickle file**: Use `python/tools/build_trie_pickle.py` to pre-build a persistent trie pickle offline, avoiding re-encoding on every server startup:

```bash
python3 python/tools/build_trie_pickle.py \
    --model_path /path/to/model \
    --dict_path /path/to/dictionary.txt \
    --output /path/to/dictionary.trie.pkl \
    --max-tokens 3
```

Where `--max-tokens` is the expected number of tokens per dictionary entry after encoding.

**Environment variables**:
- `SGLANG_BEAM_SEARCH_TRIE_DICT_ENCODE_WORKERS`: Number of concurrent encode workers (default 10)
- `SGLANG_BEAM_CONSTRAINT_DISABLE_CACHE`: Set to `1` to disable internal Tensor caching (trade compute for memory)

#### Usage Example

**Server launch**:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-1.7B \
    --enable-beam-search \
    --beam-search-constraint-dict /path/to/dictionary.txt \
    --trust-remote-code
```

**Client call**:
```python
sampling_params = {
    "max_new_tokens": 10,
    "n": 100,
    "use_beam_search": True,
}
outputs = engine.generate(prompt, sampling_params=sampling_params)
# Every generated beam is guaranteed to belong to a complete dictionary entry
```

**Constraints and edge cases**:
- If the number of valid candidates is fewer than beam_width, dummy beams are used to pad. Dummy beams are excluded from candidate ranking.
- If no valid candidates are available during generation, the request terminates early.

---

### 2.3 Cascade Attention

Cascade attention is a hybrid two-stage attention mechanism designed for beam search, reducing HBM bandwidth by separating shared-prefix and private-suffix computation.

#### When to Use

Cascade attention is best suited for computationally intensive scenarios with **prompt > 1K tokens** and **beam width > 50**. For smaller prompts or lower beam widths, the overhead of the two-stage split may outweigh the bandwidth savings.

#### How It Works

```
┌─────────────────────────────────────────────────────┐
│  Beam 0..K  ─── Shared Prefix (prompt) ──── Prefill Wrapper
│            One copy of KV, K queries                │
├─────────────────────────────────────────────────────┤
│  Beam 0     ─── Private Suffix ──── Decode Wrapper  │
│  Beam 1     ─── Private Suffix                      │
│  ...         (q_len=1 per beam)                     │
├─────────────────────────────────────────────────────┤
│            merge_state(o_shared, ..., o_priv, ...)   │
└─────────────────────────────────────────────────────┘
```

In beam search decode, all beams of a single request share the same prompt prefix. Instead of loading the prefix KV cache separately for every beam, cascade attention loads it once in a prefill wrapper and reuses it across all beams, then processes each beam's private suffix via the decode fast path.

#### Constraints and Configuration

- **Must use flashinfer backend**: `--decode-attention-backend flashinfer`
- `--beam-cascade-shared-step`: Alignment step for shared-prefix length (default 16). The shared prefix length is rounded down to the nearest multiple of this step, allowing requests with similar prompt lengths to be grouped together and reuse the same pre-computed prefill scheduling plan across batches. For example, with step=16, prompts of length 1023 and 1024 both align to 1008 and share the same plan.
- `--beam-cascade-plan-lru-cap`: LRU cache capacity for cross-batch plan reuse (default 64; set to 0 to disable). Generating prefill scheduling plans has overhead; batches with the same length structure can reuse cached plans. This parameter controls the maximum number of distinct plans to cache, evicting the least recently used entry when the cache is full.

#### Usage Example

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

### 2.4 Decode-First Scheduling

Decode-first is a scheduling optimization that prioritizes decode on the existing batch over launching new prefill batches.

#### When to Use

Decode-first is best suited for scenarios where **very few tokens are generated**, so that in-flight decode completes quickly and the next prefill is not significantly delayed.

#### Usage

```bash
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-beam-search \
    --enable-decode-first \
    --trust-remote-code
```

**Note**: `--enable-decode-first` is an independent scheduling policy that works in both beam search and non-beam-search scenarios, but provides the greatest benefit when beam search requests have long prompts and short decode lengths.

---

### 2.5 Mamba State Support

For models using Mamba / SSM (State Space Model) architectures such as Qwen3.5, beam search includes built-in support for managing recurrent state during beam expansion and pruning.

Unlike standard Transformer models whose KV cache is token-indexed, Mamba models' recurrent state is in-place per-slot. As beams expand and prune across decode steps, the mamba state is automatically forked and remapped so that each beam branch carries the correct recurrent state. No special configuration is needed — this operates transparently when the model uses the hybrid memory pool.

#### Related Server Arguments

| Argument | Type | Default | Description |
|------|------|--------|------|
| `--enable-beam-mamba-double-buffer` | bool | `False` | Enable zero-copy double-buffered mamba state pruning. When enabled, beam pruning avoids physical mamba state copies by using COW (Copy-on-Write) mapping, significantly reducing mamba state copy overhead at high beam widths. The trade-off is additional mamba state memory consumption equal to roughly one batch of beam slots |

**Notes**:
- If mamba cache space is exhausted, increase `--max-mamba-cache-size` or reduce beam width / concurrency



