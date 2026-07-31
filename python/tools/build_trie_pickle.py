#!/usr/bin/env python3
"""离线构建 Trie Pickle 工具（多进程版）。

从原始词典文件读取 SID，校验格式后多进程调用 tokenizer batch encode，
构建 trie 结构并序列化为 pickle 文件。

用法:
  python3 python/tools/build_trie_pickle.py \
      --model_path /data/model/Qwen2.5-0.5B-Instruct \
      --dict_path  /data/model/dict.txt \
      --output /data/model/dict.trie.pkl \
      --max-tokens 3

输出:
  - .pkl 文件（仅 children 结构 + terminal 标志，不含 allowed_tensor GPU 缓存）
"""

import argparse
import os
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

from transformers import AutoTokenizer
from sglang.srt.managers.prefix_constrained_trie import (
    ConstraintTrie,
    ConstraintTrieNode,
)

# 匹配单个 SID 组件，如 <a_0>, <b_101>, <c_18>
SID_COMPONENT_RE = re.compile(r"<[a-z]+_\d+>")

# batch encode 大小
ENCODE_BATCH_SIZE = 4096

# 并发 encode 进程数
NUM_ENCODE_WORKERS = 48


# ---------------------------------------------------------------------------
# Worker（在子进程中执行）
# ---------------------------------------------------------------------------

# 每个 worker 进程级别的全局 tokenizer（通过 initializer 初始化一次）
_tokenizer: "AutoTokenizer | None" = None


def _init_worker(model_path: str) -> None:
    """Worker 进程初始化：加载 tokenizer（每个进程仅一次）。"""
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _encode_batch(args: Tuple[List[str], int]) -> List[List[int]]:
    """子进程入口：使用全局 tokenizer batch encode，校验 token 数。"""
    sids, max_tokens = args
    assert _tokenizer is not None, "worker 未正确初始化 tokenizer"
    encoded = _tokenizer(sids, add_special_tokens=False, padding=False)
    results: List[List[int]] = []
    for sid, token_ids in zip(sids, encoded["input_ids"]):
        if not token_ids:
            raise RuntimeError(f"切词为空: {sid!r}")
        if len(token_ids) != max_tokens:
            raise RuntimeError(
                f"token 数不符: {sid!r} 编码为 {token_ids} "
                f"(期望 {max_tokens} 个, 实际 {len(token_ids)} 个)"
            )
        results.append(token_ids)
    return results


# ---------------------------------------------------------------------------
# Trie 序列化
# ---------------------------------------------------------------------------


def trie_to_dict(node: ConstraintTrieNode) -> dict:
    d: dict = {"t": bool(node.terminal)}
    if node.children:
        d["c"] = {int(k): trie_to_dict(v) for k, v in node.children.items()}
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Build trie pickle from dict")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--dict_path", type=str, required=True)
    ap.add_argument("--output", type=str, required=True, help="Output .pkl file path")
    ap.add_argument("--max-tokens", type=int, required=True,
                    help="每条 SID 应有的 token 数（如 <a_0><b_1><c_2> 为 3）")
    args = ap.parse_args()

    print(f"[Validate] 校验 SID 格式 ...")
    t0 = time.perf_counter()
    all_sids: List[str] = []
    with open(args.dict_path, "r", encoding="utf-8") as f:
        for line in f:
            sid = line.strip()
            if not sid:
                continue
            components = SID_COMPONENT_RE.findall(sid)
            if "".join(components) != sid or len(components) != args.max_tokens:
                print(f"[Error] 格式非法: {sid!r}（期望 {args.max_tokens} 个组件）")
                sys.exit(1)
            all_sids.append(sid)
    print(f"  格式校验通过: {len(all_sids)} 条")

    # 拆分为 batch
    batches = []
    for i in range(0, len(all_sids), ENCODE_BATCH_SIZE):
        batches.append(all_sids[i : i + ENCODE_BATCH_SIZE])
    del all_sids  # 释放内存

    print(f"[Build] {NUM_ENCODE_WORKERS} 进程并行 encode，共 {len(batches)} 个 batch ...")
    trie = ConstraintTrie()
    total = 0
    dup = 0
    next_log = 200000

    with ProcessPoolExecutor(
        max_workers=NUM_ENCODE_WORKERS,
        initializer=_init_worker,
        initargs=(args.model_path,),
    ) as pool:
        # 提交所有 batch
        futures = {}
        for batch_idx, batch_sids in enumerate(batches):
            future = pool.submit(_encode_batch, (batch_sids, args.max_tokens))
            futures[future] = batch_idx

        # 按完成顺序处理结果
        for future in as_completed(futures):
            try:
                token_ids_list = future.result()
            except RuntimeError as e:
                print(f"[Error] {e}")
                sys.exit(1)

            for token_ids in token_ids_list:
                total += 1
                if not trie.add(token_ids):
                    dup += 1

            while total >= next_log:
                print(f"  read={total} unique={trie.num_sequences} elapsed={time.perf_counter() - t0:.1f}s")
                next_log += 200000

    if trie.num_sequences == 0:
        print("[Error] 没有构建出任何合法 SID")
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    st = trie.stats()
    print(f"[Build] done in {elapsed:.1f}s")
    print(f"  raw={total}  unique={st['num_sequences']:,}  dup={dup}")
    print(f"  nodes={st['num_nodes']:,}  edges={st['num_edges']:,}  "
          f"depth(min={st['min_depth']} max={st['max_depth']})")

    print(f"\n[Pickle] serializing ...")
    t1 = time.perf_counter()
    data = {
        "tree": trie_to_dict(trie.root),
        "num_sequences": trie.num_sequences,
        "num_nodes": trie.num_nodes,
        "num_edges": trie.num_edges,
        "max_depth": trie.max_depth,
        "min_depth": trie.min_depth,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"[Pickle] wrote {args.output}  ({size_mb:.1f} MB) in "
          f"{time.perf_counter() - t1:.1f}s")
    print(f"[Done] total elapsed {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
