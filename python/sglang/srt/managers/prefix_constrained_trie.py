"""前缀约束 Trie，用于 beam search 受限解码。

从词典文件（每行一条 SID 文本，经模型 tokenizer 切词）构建 token 级前缀树，
提供 O(1) 节点步进遍历和允许后继 token GPU 张量的节点级懒缓存。

支持两种加载模式：
  - 从原始词典构建: build_trie_from_dict(dict_path, tokenizer_path)
  - 从 pickle 加载:   load_trie_from_pickle(pickle_path)  （快速，跳过重新编码）

服务启动时优先尝试 pickle，失败则回退到原始词典构建。
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import torch

logger = logging.getLogger(__name__)

# 并发 encode 进程数，可通过环境变量 SGLANG_BEAM_SEARCH_TRIE_DICT_ENCODE_WORKERS 配置
NUM_ENCODE_WORKERS = int(os.environ.get("SGLANG_BEAM_SEARCH_TRIE_DICT_ENCODE_WORKERS", "10"))

# 节点级 allowed_tensor 懒缓存开关（环境变量）。
# 置为 "1"/"true"/"yes"/"on" 时关闭缓存：get_allowed_tensor 每次现算、
# 不读不写 node.allowed_tensor（以算力换显存），用于压测缓存对性能的影响。
# 默认开启缓存（与原行为一致）。
_DISABLE_CACHE_ENV = "SGLANG_BEAM_CONSTRAINT_DISABLE_CACHE"


def _resolve_cache_enabled() -> bool:
    """根据环境变量决定是否启用 allowed_tensor 节点级缓存。"""
    val = os.environ.get(_DISABLE_CACHE_ENV, "").strip().lower()
    disabled = val in ("1", "true", "yes", "on")
    return not disabled

@dataclass
class ConstraintTrieNode:
    """前缀约束 Trie 节点。

    属性:
        children: 下一 token ID → 子节点的映射（CPU 常驻，启动时构建，后续不变）。
        terminal: 若此节点为某条合法词典序列的终点则为 True。
        allowed_tensor: 惰性构建的 GPU 张量，包含此节点允许的下一 token ID。
                        当 terminal 为 True 时包含 eos token。
                        严禁为所有节点预构建（数百万节点会因 CUDA 缓存分配器
                        的每个张量分配开销导致 GPU 显存爆炸）。
    """

    children: Dict[int, "ConstraintTrieNode"] = field(default_factory=dict)
    terminal: bool = False
    allowed_tensor: Optional[torch.Tensor] = None


class ConstraintTrie:
    """Token 级前缀 Trie，带惰性 GPU 缓存的允许后继 token 张量。"""

    def __init__(self) -> None:
        self.root = ConstraintTrieNode()
        self.num_sequences = 0
        self.num_nodes = 1
        self.num_edges = 0
        self.max_depth = 0
        self.min_depth: Optional[int] = None
        # 节点级 allowed_tensor 懒缓存开关：默认开启，可由环境变量关闭（压测用）。
        self.cache_allowed_tensor = _resolve_cache_enabled()
        if not self.cache_allowed_tensor:
            logger.warning(
                f"[Trie] allowed_tensor node-cache DISABLED via {_DISABLE_CACHE_ENV}; "
                f"each get_allowed_tensor will recompute (trade compute for memory)"
            )

    def add(self, token_ids: Sequence[int]) -> bool:
        """向 Trie 中添加一条 token ID 序列。

        若序列为新序列（非重复）则返回 True。
        """
        if not token_ids:
            return False
        node = self.root
        for tid in token_ids:
            tid = int(tid)
            child = node.children.get(tid)
            if child is None:
                child = ConstraintTrieNode()
                node.children[tid] = child
                self.num_nodes += 1
                self.num_edges += 1
            node = child
        if node.terminal:
            return False
        node.terminal = True
        self.num_sequences += 1
        depth = len(token_ids)
        self.max_depth = max(self.max_depth, depth)
        self.min_depth = depth if self.min_depth is None else min(self.min_depth, depth)
        return True


    def step(self, node: Optional[ConstraintTrieNode], token: int) -> Optional[ConstraintTrieNode]:
        """从 *node* 沿 *token* 步进一步；返回子节点或 None。"""
        if node is None:
            return None
        return node.children.get(int(token))


    def get_allowed_tensor(
        self, node: Optional[ConstraintTrieNode], eos_token_ids: Set[int], device: torch.device = None
    ) -> torch.Tensor:
        """返回 *node* 的允许下一 token 的 GPU int64 张量。

        张量在首次访问时惰性构建并缓存在节点上（直接放在 *device* 上以避免
        每次 H2D 拷贝）。当 *node* 为终点时，追加 eos token 使 beam 能在合法
        序列终点正确结束。
        """
        if node is None:
            return _empty_allowed_tensor(device=device)
        # 缓存开启且已物化 → 直接复用。
        if self.cache_allowed_tensor and node.allowed_tensor is not None:
            return node.allowed_tensor
        allowed = list(node.children.keys())
        if node.terminal:
            allowed.extend(eos_token_ids)
        tensor = torch.tensor(sorted(allowed), dtype=torch.int64, device=device)
        # 仅在缓存开启时写回节点；关闭时现算现用、不占常驻显存。
        if self.cache_allowed_tensor:
            node.allowed_tensor = tensor
        return tensor

    def build_root_dense_mask(
        self, vocab_size: int, device: torch.device
    ) -> torch.Tensor:
        """构建密集 [vocab_size] -inf mask，根节点后继允许的位置置 0。"""
        mask = torch.full((vocab_size,), float("-inf"), dtype=torch.float32, device=device)
        allowed = list(self.root.children.keys())
        if allowed:
            mask[torch.tensor(allowed, dtype=torch.long, device=device)] = 0.0
        return mask

    def prebuild_shallow_allowed_tensors(
        self, device: torch.device, eos_token_ids: Set[int]
    ) -> int:
        """仅预构建 root（depth=0）和 depth=1 节点的 allowed_tensor。

        这两层保证每个请求都会访问，惰性构建只会增加冷启动延迟而毫无收益。
        更深层必须保持惰性以避免 GPU 显存爆炸。

        返回预构建的节点数。
        """
        # 缓存关闭时预构建无意义（get_allowed_tensor 不会写回），直接跳过。
        if not self.cache_allowed_tensor:
            logger.info("[Trie] cache disabled, skip prebuild_shallow_allowed_tensors")
            return 0
        count = 0
        # depth=0：根节点
        self.get_allowed_tensor(self.root, eos_token_ids, device=device)
        count += 1
        # depth=1：根节点的直接子节点
        for child in self.root.children.values():
            self.get_allowed_tensor(child, eos_token_ids, device=device)
            count += 1
        return count


    def stats(self) -> Dict[str, int]:
        return {
            "num_sequences": self.num_sequences,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "max_depth": self.max_depth,
            "min_depth": self.min_depth or 0,
        }


def _empty_allowed_tensor(device: torch.device = None) -> torch.Tensor:
    return torch.empty(0, dtype=torch.int64, device=device)


# 每个 worker 进程级别的全局 tokenizer（通过 initializer 初始化一次）
_encode_tokenizer = None


def _init_encode_worker(tokenizer_path: str) -> None:
    """Worker 进程初始化：加载 tokenizer（每个进程仅一次）。"""
    global _encode_tokenizer
    from transformers import AutoTokenizer

    _encode_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _encode_batch_tokens(args: tuple) -> List[List[int]]:
    """子进程入口：使用全局 tokenizer batch encode 一批 SID。"""
    (sids,) = args
    encoded = _encode_tokenizer(sids, add_special_tokens=False, padding=False)
    return encoded["input_ids"]


def build_trie_from_dict(
    dict_path: str,
    tokenizer_path: str,
) -> Optional[ConstraintTrie]:
    """从词典文件构建 ConstraintTrie（多进程编码）。

    子进程从 *tokenizer_path* 加载 tokenizer 并行编码。
    参数:
        dict_path: 词典文件路径（每行一条记录）。
        tokenizer_path: Tokenizer 路径（供子进程使用）。

    返回:
        构建好的 ConstraintTrie 实例。
    """
    ENCODE_BATCH_SIZE = 4096

    logger.info(f"[Trie] building from {dict_path} ({NUM_ENCODE_WORKERS} encode workers) ...")
    start_time = time.perf_counter()

    # 读取全部 SID
    all_sids: List[str] = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            sid = line.strip()
            if sid:
                all_sids.append(sid)

    # 拆分为 batch
    batches = []
    for i in range(0, len(all_sids), ENCODE_BATCH_SIZE):
        batches.append(all_sids[i : i + ENCODE_BATCH_SIZE])
    del all_sids

    trie = ConstraintTrie()
    total = 0
    bad = 0
    dup = 0
    next_log = 200000

    with ProcessPoolExecutor(
        max_workers=NUM_ENCODE_WORKERS,
        initializer=_init_encode_worker,
        initargs=(tokenizer_path,),
    ) as pool:
        futures = {}
        for batch_sids in batches:
            future = pool.submit(_encode_batch_tokens, (batch_sids,))
            futures[future] = len(batch_sids)  # 记录 batch 大小用于统计

        for future in as_completed(futures):
            ids_list = future.result()
            for token_ids in ids_list:
                total += 1
                if not token_ids:
                    bad += 1
                    continue
                if not trie.add(token_ids):
                    dup += 1

            while total >= next_log:
                elapsed = time.perf_counter() - start_time
                logger.info(f"[Trie] read={total} elapsed={elapsed:.1f}s")
                next_log += 200000

    if trie.num_sequences == 0:
        logger.warning(f"[Trie] no valid sequences built from {dict_path}")
        return None

    elapsed = time.perf_counter() - start_time
    st = trie.stats()
    logger.info(
        f"[Trie] done: raw={total} unique_sid={st['num_sequences']} "
        f"dup={dup} bad_encode={bad} elapsed={elapsed:.1f}s "
        f"nodes={st['num_nodes']} edges={st['num_edges']} "
        f"depth(min={st['min_depth']} max={st['max_depth']})"
    )
    return trie


def load_trie_from_pickle(pickle_path: str) -> Optional[ConstraintTrie]:
    """从 pickle 文件加载 ConstraintTrie。

    pickle 必须由 ``trie_to_dict`` 生成（见 ``build_trie_pickle.py``）。
    GPU 的 ``allowed_tensor`` 缓存不会被保存——它们将被惰性构建。

    若文件不是合法 pickle 或不包含预期的 trie 结构则返回 None。
    """
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, ValueError, TypeError, Exception) as e:
        logger.info(f"[Trie] failed to load pickle: {pickle_path} ({e})")
        return None

    # 校验顶层 key 是否完整
    for key in ("tree", "num_sequences", "num_nodes", "num_edges", "max_depth", "min_depth"):
        if key not in data:
            logger.warning(f"[Trie] invalid pickle, missing key {key!r}: {pickle_path}")
            return None

    num_nodes = [0]
    num_edges = [0]
    num_sequences = [0]

    root = _dict_to_node(data["tree"], num_nodes, num_edges, num_sequences)

    trie = ConstraintTrie()
    trie.root = root
    trie.num_sequences = num_sequences[0]
    trie.num_nodes = num_nodes[0]
    trie.num_edges = num_edges[0]
    trie.max_depth = data["max_depth"]
    trie.min_depth = data["min_depth"]

    logger.info(
        f"[Trie] loaded from pickle: {pickle_path} "
        f"sequences={trie.num_sequences} nodes={trie.num_nodes} "
        f"edges={trie.num_edges} depth(min={trie.min_depth} max={trie.max_depth})"
    )
    return trie


def _dict_to_node(d: dict, num_nodes: list, num_edges: list, num_sequences: list) -> ConstraintTrieNode:
    """从嵌套字典迭代重建 ConstraintTrieNode 树（避免栈溢出）。"""
    root = ConstraintTrieNode()
    root.terminal = d.get("t", False)
    if root.terminal:
        num_sequences[0] += 1
    num_nodes[0] += 1

    # 栈元素: (parent_node, child_token_id, child_dict)
    stack = []
    for child_id, child_dict in d.get("c", {}).items():
        stack.append((root, int(child_id), child_dict))

    while stack:
        parent, token_id, cd = stack.pop()
        node = ConstraintTrieNode()
        node.terminal = cd.get("t", False)
        if node.terminal:
            num_sequences[0] += 1
        num_nodes[0] += 1
        num_edges[0] += 1
        parent.children[token_id] = node
        for cid, cdict in cd.get("c", {}).items():
            stack.append((node, int(cid), cdict))

    return root
