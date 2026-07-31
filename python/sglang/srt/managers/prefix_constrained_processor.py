"""前缀约束处理器，用于 beam search 受限解码。

在 log_softmax 之后、topk 提取之前对 logprobs 施加基于 Trie 的前缀约束。
"""

from __future__ import annotations

import logging
from typing import List, Set

import torch

from sglang.srt.managers.beam_search_type import BeamSearchSequence
from sglang.srt.managers.prefix_constrained_trie import ConstraintTrie

logger = logging.getLogger(__name__)


class PrefixConstrainedProcessor:
    """对 beam-search logprobs 施加前缀 Trie 约束。

    两个生命周期阶段:
      - prefill: 所有 beam 从根节点开始 → 广播根节点密集 mask。
      - decode : 每个 beam 有自己的 trie_node → 通过 scatter 逐行 mask。

    Beam 的 *node* 状态由 BeamSearchProcessor 外部维护
    （见 :meth:`init_beam_node` / :meth:`step_beam_node`）。
    """

    def __init__(
        self,
        trie: ConstraintTrie,
        vocab_size: int,
        eos_token_ids: Set[int],
        device: torch.device,
    ):
        assert trie is not None, "PrefixConstrainedProcessor requires a non-None trie"
        self.trie = trie
        self.vocab_size = vocab_size
        self.eos_token_ids = eos_token_ids
        self.device = device

        # 预构建根节点密集 mask（用于 step==0 广播快速路径）。
        self._root_dense_mask = trie.build_root_dense_mask(vocab_size, device)
        # 仅预构建 root + depth=1 节点的 allowed_tensor
        n_prebuilt = trie.prebuild_shallow_allowed_tensors(device, eos_token_ids)
        logger.info(f"[Constraint] Pre-built allowed_tensor for {n_prebuilt} shallow nodes")

    @torch.no_grad()
    def apply_prefill(self, logprobs_row: torch.Tensor) -> None:
        """对 prefill 阶段的单行 logprobs 张量 [vocab] 做 mask。"""
        if logprobs_row.dim() != 1:
            return
        logprobs_row.add_(self._root_dense_mask.to(logprobs_row.dtype))

    @torch.no_grad()
    def apply_decode(
        self, logprobs: torch.Tensor, beams: List[BeamSearchSequence]
    ) -> None:
        """对 decode 阶段的多行 logprobs 张量 [n_beams, vocab] 做 mask（原地修改）。

        每个 beam 处在 Trie 的不同节点，只允许各自的后继 token，其余位置置 -inf。

        参数:
            logprobs: [n_beams, vocab] GPU 张量（原地修改）。
            beams: 有序的 beam 序列列表，其 trie_node 指针决定每行的允许下一 token。

        实现思路:
            核心差异是 "逐行(per-row)循环" vs "整体(batched)批量"：把 N 次小操作
            摊平成几次大操作，让 GPU 一次处理全部 beam，而不是排队跑 N 次。

            朴素做法——逐个 beam 处理，每个 beam 在自己那一行上单独"标记 allowed +
            赋值"，for 循环跑 n_beams 次，每次一个独立 GPU 操作：

                allowed_mask = torch.zeros(n_beams, vocab, dtype=torch.bool)
                for i, beam in enumerate(beams):
                    allowed = self.trie.get_allowed_tensor(beam.trie_node, ...)
                    allowed_mask[i, allowed] = True  # 逐行 scatter，n_beams 次小 kernel
                logprobs.masked_fill_(~allowed_mask, float("-inf"))

            问题在于 n_beams 次逐行 scatter = n_beams 次小 kernel 启动（256 beam
            实测约 6ms），且 mask 及其取反会分配两个 [n_beams, vocab] 的大张量。

            优化做法——先把所有 beam 的合法位置汇总成一组 (rows, cols) 坐标对
            （cols = 各 beam allowed 拼平；rows = repeat_interleave 展开的行号），
            再用三步整体完成（数据仍是二维，只是用坐标对一次性索引所有行）：
                1. 取值(gather)：saved = logprobs[rows, cols]  一次取出全部合法原值
                2. 清空(fill) ：logprobs.fill_(-inf)           整张表一次性置 -inf
                3. 赋值(scatter)：logprobs[rows, cols] = saved  一次性写回原值
            这样 GPU kernel 数从 n_beams+ 降到约 5 个，且仅额外分配一个
            [合法 token 总数] 的小张量，省去大 bool mask 的分配与遍历。
            （注：上述 kernel 估算基于 allowed_tensor 缓存命中；冷启动首次访问
            深层节点时仍需逐 beam 现算 + H2D 拷贝。）
        """
        if logprobs.size(0) == 0:
            return
        if logprobs.size(0) != len(beams):
            raise ValueError(
                f"logprobs row count ({logprobs.size(0)}) != beam count ({len(beams)})"
            )

        n_beams, __ = logprobs.shape
        device = logprobs.device

        # 每个 beam 的 allowed token（GPU int64 张量，多为缓存命中）及其长度。
        allowed_list = [
            self.trie.get_allowed_tensor(
                beam.trie_node, self.eos_token_ids, device=device
            )
            for beam in beams
        ]
        lengths = torch.tensor(
            [a.numel() for a in allowed_list], device=device, dtype=torch.long
        )

        # 拼平为一维列坐标 cols，并按各 beam 长度展开出对应行坐标 rows。
        cols = torch.cat(allowed_list)
        rows = torch.repeat_interleave(
            torch.arange(n_beams, device=device, dtype=torch.long), lengths
        )

        # gather 合法位置原值 → 整体置 -inf → 写回，等效 mask 但避免大张量分配。
        saved = logprobs[rows, cols]
        logprobs.fill_(float("-inf"))
        logprobs[rows, cols] = saved

    def init_beam_node(self, beam: BeamSearchSequence, first_token: int) -> None:
        """在 beam 的第一个 token 选定后设置 trie_node（prefill 阶段）。"""
        beam.trie_node = self.trie.step(self.trie.root, first_token)

    def step_beam_node(
        self,
        child_beam: BeamSearchSequence,
        parent_beam: BeamSearchSequence,
        token: int,
    ) -> None:
        """当 beam 扩展新 token 时前进 trie_node。"""
        child_beam.trie_node = self.trie.step(parent_beam.trie_node, token)
