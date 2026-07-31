from __future__ import annotations

"""FlashInfer 后端的 Beam Search 级联注意力模块。

级联注意力（混合两阶段）用于 beam search decode：
  - 共享 prompt 前缀 → BatchPrefillWithPagedKVCacheWrapper。
    每个请求的 K 个 beam 通过 qo_indptr 分组，共享 prompt KV 从 HBM 只加载一次，
    在 K 个查询间复用（q_len=K>1，一份 KV 对应多个 Q，适合 prefill 工作负载）。
  - 各 beam 私有后缀 → BatchDecodeWithPagedKVCacheWrapper。
    q_len=1，面向 decode 优化的快路径（一份 KV 对应一个 Q）。
  - 两段结果通过 flashinfer.cascade.merge_state 合并。

仅在后端使用单个 wrapper 时支持（无 SWA / 交叉注意力）。
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Optional

import torch
import torch.cuda.nvtx as nvtx

from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_flashinfer_available

if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        fast_decode_plan,
    )
    from flashinfer.cascade import merge_state

if TYPE_CHECKING:
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@dataclass
class BeamCascadeReplayContext:
    """Beam search 级联注意力在 CUDA graph 回放时所需的状态。"""

    # 每个 beam 行的共享前缀长度，padding 至 bs
    shared_lens: torch.Tensor
    # 按请求累加的 beam 数量（级联共享阶段 qo_indptr），padding 至 bs+1
    req_offsets: torch.Tensor
    # (bs, tuple(rids))，纯 CPU 数据无需 GPU 同步，连续回放同批请求不变时可复用共享阶段 plan
    shared_sig: Optional[tuple]


class BeamCascadeManager:
    """管理 FlashInfer 后端全部 beam search 级联状态与逻辑。

    级联注意力（混合两阶段）：
      - 共享 prompt 前缀段：用 prefill wrapper，一份 KV 对应多个 Q，共享 KV 只加载一次。
      - 各 beam 私有后缀段：用 decode wrapper，一份 KV 对应一个 Q，走 decode 快路径。
    """

    def __init__(self, backend: "FlashInferAttnBackend", model_runner: "ModelRunner"):
        self.b = backend

        # 两阶段 wrapper.plan() 所需的头数 / 维度 / dtype 元信息。
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.q_data_type = model_runner.dtype
        self.data_type = model_runner.kv_cache_dtype

        # 非 cuda graph 的级联两阶段 wrapper
        self.shared_wrapper_eager = BatchPrefillWithPagedKVCacheWrapper(
            backend.workspace_buffer,
            "NHD",
            backend=backend.prefill_backend,
        )
        self.private_wrapper_eager = BatchDecodeWithPagedKVCacheWrapper(
            backend.workspace_buffer,
            "NHD",
            backend=backend.decode_backend,
            use_tensor_cores=backend.decode_use_tensor_cores,
        )

        # CUDA graph 路径的级联 wrapper（capture 时按 bs 创建）：
        #   {bs: (shared_prefill_wrapper, private_decode_wrapper)}
        self.cuda_graph_wrappers: dict = {}
        # CUDA graph 路径的 KV page table buffer
        self._cuda_graph_kv_buffers: dict = {}

        # 跨 batch 级联 plan 缓存（LRU，缓存 flashinfer prefill 调度计划）
        self._plan_lru: "OrderedDict" = OrderedDict()
        self._plan_lru_cap = model_runner.server_args.beam_cascade_plan_lru_cap

        # CUDA graph replay 时的级联切分信息（shared_lens / req_offsets / shared_sig）。
        self.replay_ctx: Optional[BeamCascadeReplayContext] = None

        # ── 步间缓存（连续 decode 步请求集不变时跳过共享段重建）──
        # plan_eager 路径：请求集签名
        self._eager_sig: Optional[tuple] = None
        # _plan_graph 路径：上次写入共享段的 (wrapper_id, sig)
        self._cg_owner: Optional[tuple] = None
        # _plan_graph 路径：beam_shared_lens 的 CPU 副本（跳过共享段时仍需用于计算私有段长度）
        self._shared_cpu: Optional[torch.Tensor] = None

    def is_active(self, forward_batch: "ForwardBatch") -> bool:
        """本步是否应使用 beam search 级联注意力。"""
        return (
            forward_batch.is_beam_search
            and forward_batch.beam_cascade_enabled
            and forward_batch.beam_shared_lens is not None
            and forward_batch.beam_cascade_req_offsets is not None
        )

    def plan_eager(self, forward_batch: "ForwardBatch") -> bool:
        """规划 beam search decode 的混合两阶段级联（eager 模式）。

        返回 True 表示级联已激活并完成规划（后端跳过其正常的 decode 规划）；
        返回 False 表示未激活。

        共享阶段（prefill wrapper）：
            用 qo_indptr 把每个请求的 K 个 beam 归组，每个请求对应一段共享 KV，
            共享 KV 只加载一次，在 K 个查询间复用。
        私有阶段（decode wrapper）：
            每 beam 行一个 query，仅关注其私有后缀段 [shared:seq]
            （kv_start_idx = shared_len），q_len=1 走 decode 快路径。
        """
        if not self.is_active(forward_batch):
            return False

        from sglang.srt.layers.attention.flashinfer_backend import DecodeMetadata

        req_to_token = self.req_to_token
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        beam_shared_lens = forward_batch.beam_shared_lens
        # 形状 [num_reqs + 1]
        beam_cascade_req_offsets = forward_batch.beam_cascade_req_offsets

        batch_size = forward_batch.batch_size
        device = req_pool_indices.device
        num_reqs = int(beam_cascade_req_offsets.numel() - 1)

        rids = getattr(forward_batch, "rids", None)
        eager_sig = (batch_size, tuple(rids)) if rids is not None else None
        eager_unchanged = eager_sig is not None and self._eager_sig == eager_sig
        if not eager_unchanged:
            nvtx.range_push("cascade:build_shared")

            # 向量化构建共享 KV indices（避免逐请求 .item() 带来的 GPU 同步开销）。
            start_beams = beam_cascade_req_offsets[:num_reqs].to(torch.int64)
            shared_lengths = beam_shared_lens[start_beams].to(torch.int64)
            first_beam_pool = req_pool_indices[start_beams]
            total_shared = int(shared_lengths.sum().item())

            if total_shared > 0:
                shared_lengths_dev = shared_lengths.to(device, torch.int32)
                shared_kv_indptr = torch.zeros(
                    num_reqs + 1, dtype=torch.int32, device=device
                )
                shared_kv_indptr[1:] = torch.cumsum(shared_lengths_dev, dim=0)
                create_flashinfer_kv_indices_triton[(num_reqs,)](
                    req_to_token,
                    first_beam_pool,
                    shared_lengths_dev,
                    shared_kv_indptr,
                    None,  # kv_start_idx = 0
                    # 就地写入临时 indices buffer
                    (
                        shared_kv_indices_buf := torch.empty(
                            total_shared, dtype=torch.int32, device=device
                        )
                    ),
                    req_to_token.shape[1],
                )
                shared_kv_indices = shared_kv_indices_buf
            else:
                shared_kv_indices = torch.empty(0, dtype=torch.int32, device=device)
                shared_kv_indptr = torch.zeros(
                    num_reqs + 1, dtype=torch.int32, device=device
                )

            shared_kv_lp = torch.ones(num_reqs, dtype=torch.int32, device=device)
            shared_qo_indptr = beam_cascade_req_offsets.to(torch.int32)

            nvtx.range_push("cascade:plan_shared")
            self.shared_wrapper_eager.plan(
                shared_qo_indptr,
                shared_kv_indptr,
                shared_kv_indices,
                shared_kv_lp,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.b.page_size,
                causal=False,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                disable_split_kv=True,
            )
            nvtx.range_pop()  # cascade:plan_shared
            nvtx.range_pop()  # cascade:build_shared

            self._eager_sig = eager_sig

        nvtx.range_push("cascade:build_private")
        private_lens = seq_lens - beam_shared_lens.to(device, seq_lens.dtype)
        private_kv_indptr = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device
        )
        self.b.indices_updater_decode.call_begin_forward(
            self.private_wrapper_eager,
            req_pool_indices,
            private_lens,
            int(private_lens.sum().item()),
            private_kv_indptr,
            beam_shared_lens.to(seq_lens.dtype),  # kv_start_idx
            None,  # spec_info
            None,  # seq_lens_cpu（eager 模式：不覆盖 indptr）
        )
        nvtx.range_pop()  # cascade:build_private

        self.b.forward_metadata = DecodeMetadata(
            self.b.decode_wrappers,
            cascade_shared_wrapper=self.shared_wrapper_eager,
            cascade_private_wrapper=self.private_wrapper_eager,
        )
        return True

    def init_cuda_graph_buffers(self, max_bs: int, max_num_tokens: int):
        """为混合两阶段分配 CUDA graph 固定 buffer。

        共享阶段（prefill wrapper）需要自己的 qo/kv indptr+indices+lp；
        私有阶段（decode wrapper）的 buffer 在 capture 时按 bs 创建
        （传入 BatchDecode wrapper 构造函数）。
        """
        n_indices = max_num_tokens * self.b.max_context_len
        # 共享前缀只容纳所有请求的 prompt_lens 之和。beam search 中
        # beam_width ≥ 2 → max_requests ≤ max_bs/2，因此最多需要
        # n_indices//2 个条目。
        n_shared = n_indices // 2
        cg = {}
        cg["s_qo_indptr"] = torch.zeros(max_bs + 1, dtype=torch.int32, device="cuda")
        cg["s_kv_indptr"] = torch.zeros(max_bs + 1, dtype=torch.int32, device="cuda")
        cg["s_kv_indices"] = torch.zeros(n_shared, dtype=torch.int32, device="cuda")
        cg["s_kv_lp"] = torch.ones(max_bs, dtype=torch.int32, device="cuda")
        # 私有后缀容纳所有 beam 行的 private_lens 之和
        # （最大 max_bs * max_context_len，需要全量）。
        cg["p_kv_indices"] = torch.zeros(n_indices, dtype=torch.int32, device="cuda")
        cg["p_kv_indptr"] = torch.zeros(max_bs + 1, dtype=torch.int32, device="cuda")
        cg["p_kv_lp"] = torch.ones(max_bs, dtype=torch.int32, device="cuda")
        self._cuda_graph_kv_buffers = cg

    def capture(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        decode_wrappers: list,
    ) -> bool:
        """CUDA graph capture 时的级联规划。

        返回 True 表示级联已处理 decode capture（后端跳过其正常的 capture 分支）。
        decode_wrappers 仅用于填充 forward_metadata.decode_wrappers（forward_decode
        每层都会引用），cascade 路径下其输出会被 cascade 结果覆盖。
        """
        from sglang.srt.layers.attention.flashinfer_backend import DecodeMetadata

        cg = self._cuda_graph_kv_buffers
        nt = num_tokens
        # 共享前缀 prefill wrapper（q_len = 每请求 K 个 beam）。
        shared_w = BatchPrefillWithPagedKVCacheWrapper(
            self.b.workspace_buffer,
            "NHD",
            backend=self.b.prefill_backend,
            use_cuda_graph=True,
            qo_indptr_buf=cg["s_qo_indptr"][: nt + 1],
            paged_kv_indptr_buf=cg["s_kv_indptr"][: nt + 1],
            paged_kv_indices_buf=cg["s_kv_indices"],
            paged_kv_last_page_len_buf=cg["s_kv_lp"][:nt],
        )
        # 私有后缀 decode wrapper（q_len = 1，decode 快路径）。
        private_w = BatchDecodeWithPagedKVCacheWrapper(
            self.b.workspace_buffer,
            "NHD",
            backend=self.b.decode_backend,
            use_cuda_graph=True,
            use_tensor_cores=self.b.decode_use_tensor_cores,
            paged_kv_indptr_buffer=cg["p_kv_indptr"][: nt + 1],
            paged_kv_indices_buffer=cg["p_kv_indices"],
            paged_kv_last_page_len_buffer=cg["p_kv_lp"][:nt],
        )
        # Capture 用 dummy 切分：shared = seq_lens - 1，private = 1，每请求一个 beam。
        dummy_shared = (seq_lens - 1).clamp(min=0)
        dummy_beam_offsets = torch.arange(
            bs + 1, dtype=torch.int32, device=seq_lens.device
        )
        self._plan_graph(
            shared_w,
            private_w,
            req_pool_indices,
            seq_lens,
            dummy_shared,
            dummy_beam_offsets,
            num_tokens,
        )
        # 在上述真实 plan() 填充好 _cached_module 之后，将私有 decode wrapper
        # 的 begin_forward 替换为对 CUDA graph 安全的 fast_decode_plan。
        private_w.begin_forward = partial(fast_decode_plan, private_w)

        self.cuda_graph_wrappers[bs] = (shared_w, private_w)
        self.b.forward_metadata = DecodeMetadata(
            decode_wrappers,
            cascade_shared_wrapper=shared_w,
            cascade_private_wrapper=private_w,
        )
        return True

    def replay(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> bool:
        """CUDA graph replay 时的级联重新规划。

        cascade 一旦开启，所有 decode/idle CUDA graph 都以 cascade 路径 capture，
        replay 也必须走 cascade（replay_ctx 由 runner 保证恒非 None，非 beam / idle
        批次会得到退化的 shared_len=0 上下文）。在 graph replay 期间只有 buffer
        *内容* 起作用——Python forward_decode 主体不会重新执行，因此这里不触碰
        forward_metadata（已在 capture 时设置）。
        """
        assert (
            self.replay_ctx is not None
        ), "cascade 已开启但 replay_ctx 为 None，runner 应对所有批次构造上下文"
        assert (
            bs in self.cuda_graph_wrappers
        ), f"bs={bs} 的 cascade CUDA graph 未被 capture"

        ctx = self.replay_ctx
        shared_w, private_w = self.cuda_graph_wrappers[bs]
        self._plan_graph(
            shared_w,
            private_w,
            req_pool_indices[:bs],
            seq_lens[:bs],
            ctx.shared_lens[:bs],
            ctx.req_offsets[: bs + 1],
            bs,
            shared_sig=ctx.shared_sig,
            seq_lens_cpu=(seq_lens_cpu[:bs] if seq_lens_cpu is not None else None),
        )
        return True

    def _plan_graph(
        self,
        shared_wrapper: BatchPrefillWithPagedKVCacheWrapper,
        private_wrapper: BatchDecodeWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        beam_shared_lens: torch.Tensor,
        beam_cascade_req_offsets: torch.Tensor,
        num_tokens: int,
        shared_sig: Optional[tuple] = None,
        seq_lens_cpu: Optional[torch.Tensor] = None,
    ):
        """构建并规划 CUDA graph capture/replay 的混合级联（编排器）。"""
        device = req_pool_indices.device
        bs = num_tokens

        # ── 共享阶段（prefill wrapper）──────────────────────────────
        # 共享前缀（prompt）结构在 decode 各步之间不变：cg buffer 与 prefill plan
        # 仅需在请求集（或 bs / wrapper）变化时重建。通过与同步无关的 CPU 签名
        # （请求 id 元组）加上最后写入共享 cg buffer 的 wrapper 身份探测变化
        # （所有 bs 共用一套 cg buffer，bs 切换会使内容失效）。
        nvtx.range_push("cascade:build_shared")
        shared_unchanged = shared_sig is not None and self._cg_owner == (
            id(shared_wrapper),
            shared_sig,
        )
        if shared_unchanged:
            shared_cpu = self._shared_cpu
        else:
            (
                shared_cpu,
                s_qo_cpu,
                s_kv_cpu,
                s_lp_cpu,
                total_shared,
                plan_tag,
            ) = self._build_shared_indices(
                req_pool_indices,
                beam_shared_lens,
                beam_cascade_req_offsets,
                bs,
                device,
            )
            if getattr(shared_wrapper, "_sgl_plan_tag", None) != plan_tag:
                self._plan_shared_wrapper(
                    shared_wrapper,
                    bs,
                    s_qo_cpu,
                    s_kv_cpu,
                    s_lp_cpu,
                    total_shared,
                    plan_tag,
                )
                shared_wrapper._sgl_plan_tag = plan_tag
            # 记录持有者 + 缓存 CPU 共享长度，使后续步可完全跳过本阶段。
            self._shared_cpu = shared_cpu
            self._cg_owner = (id(shared_wrapper), shared_sig)
        nvtx.range_pop()  # cascade:build_shared

        self._plan_private_stage(
            private_wrapper,
            req_pool_indices,
            seq_lens,
            beam_shared_lens,
            shared_cpu,
            bs,
            device,
            seq_lens_cpu,
        )

    def _build_shared_indices(
        self,
        req_pool_indices: torch.Tensor,
        beam_shared_lens: torch.Tensor,
        beam_cascade_req_offsets: torch.Tensor,
        bs: int,
        device: torch.device,
    ):
        """构建共享阶段（prefill）的 CPU indptr、GPU KV indices，并返回 plan 指纹。

        全程向量化 CPU 构建（各一次 D2H），避免逐请求 .item() 同步。
        beam_cascade_req_offsets 已用重复尾部值填充至 bs+1，填充槽位 n_beams==0。

        返回 ``(shared_cpu, s_qo_cpu, s_kv_cpu, s_lp_cpu, total_shared, plan_tag)``。
        """
        cg = self._cuda_graph_kv_buffers

        offsets_cpu = beam_cascade_req_offsets.to("cpu")
        shared_cpu = beam_shared_lens.to("cpu")
        n_beams_per_req = (offsets_cpu[1 : bs + 1] - offsets_cpu[:bs]).to(torch.int64)
        real_mask = n_beams_per_req > 0
        start_beams = offsets_cpu[:bs].to(torch.int64)
        shared_lengths = torch.zeros(bs, dtype=torch.int64)
        shared_lengths[real_mask] = shared_cpu[start_beams[real_mask]].to(torch.int64)

        # CPU 端 indptr：既写入 cg buffer，也作为 plan() 输入以避免其内部 D2H。
        s_qo_cpu = torch.zeros(bs + 1, dtype=torch.int32)
        s_qo_cpu[1:] = torch.cumsum(n_beams_per_req, dim=0).to(torch.int32)
        s_kv_cpu = torch.zeros(bs + 1, dtype=torch.int32)
        s_kv_cpu[1:] = torch.cumsum(shared_lengths, dim=0).to(torch.int32)
        s_lp_cpu = torch.ones(bs, dtype=torch.int32)
        total_shared = int(s_kv_cpu[-1].item())

        cg["s_qo_indptr"][: bs + 1].copy_(s_qo_cpu, non_blocking=True)
        cg["s_kv_indptr"][: bs + 1].copy_(s_kv_cpu, non_blocking=True)
        if total_shared > 0:
            # 单个 triton kernel（无同步）构建共享 KV indices：每程序处理一个请求，
            # 从 req_to_token 复制第一个 beam 的 prompt token slot [0:shared_len]。
            # 填充槽位 shared_len==0（kernel 无操作），且其 start_beam 可能越界
            # （等于 bs），故将 pool index 钳制到有效范围以避免越界 gather。
            start_beams_dev = start_beams.clamp(max=bs - 1).to(device)
            first_beam_pool = req_pool_indices[start_beams_dev]
            shared_lengths_dev = shared_lengths.to(device, torch.int32)
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                first_beam_pool,
                shared_lengths_dev,
                cg["s_kv_indptr"],
                None,  # kv_start_idx = 0（共享前缀从 token 0 开始）
                cg["s_kv_indices"],
                self.req_to_token.shape[1],
            )

        # 对 (qo, kv) 长度结构指纹化：plan 仅取决于长度结构（每请求 beam 数 + 共享
        # 长度），不依赖具体 indices 值，故相同结构的 batch 可复用 plan。对原始 int
        # 字节做 hash（C 实现 O(bs)），避免构建 bs 大小的 Python tuple；作为缓存键
        # 碰撞安全性足够（碰撞仅导致一次冗余 re-plan，plan 始终会重新校验）。
        plan_tag = (
            int(s_qo_cpu[-1].item()),
            total_shared,
            hash(n_beams_per_req.numpy().tobytes()),
            hash(shared_lengths.numpy().tobytes()),
        )
        return shared_cpu, s_qo_cpu, s_kv_cpu, s_lp_cpu, total_shared, plan_tag

    def _plan_shared_wrapper(
        self,
        shared_wrapper: BatchPrefillWithPagedKVCacheWrapper,
        bs: int,
        s_qo_cpu: torch.Tensor,
        s_kv_cpu: torch.Tensor,
        s_lp_cpu: torch.Tensor,
        total_shared: int,
        plan_tag: tuple,
    ):
        """规划共享阶段 prefill wrapper，跨 batch 复用调度计划。

        prefill 调度计划（``_plan_info`` + ``_int_workspace_buffer`` 内容）仅取决于
        plan_tag 捕获的 (qo, kv) 长度结构，不依赖具体 kv indices（后者每 batch 在
        ``_build_shared_indices`` 中重新填充）。故按 (bs, plan_tag) 做 LRU 缓存：
        命中时恢复到该共享 wrapper 并跳过宿主调度器，未命中时运行 plan() 并快照。
        调度计划在同一 batch 的多次 run() 间持久有效（见 shared_unchanged 快路径）。
        """
        cg = self._cuda_graph_kv_buffers
        lru = self._plan_lru
        cache_enabled = self._plan_lru_cap > 0
        lru_key = (bs, plan_tag)
        cached = lru.get(lru_key) if cache_enabled else None

        if cached is not None:
            nvtx.range_push("cascade:plan_shared:cache_hit")
            cached_plan_info, cached_ws = cached
            shared_wrapper._int_workspace_buffer.copy_(cached_ws)
            shared_wrapper._plan_info = cached_plan_info
            lru.move_to_end(lru_key)
            logger.debug(
                "beam cascade plan cache HIT (bs=%d, plan_tag=%s, lru_size=%d) "
                "-> skipped prefill scheduler",
                bs,
                plan_tag,
                len(lru),
            )
            nvtx.range_pop()
            return

        nvtx.range_push("cascade:plan_shared")
        shared_wrapper.plan(
            s_qo_cpu,
            s_kv_cpu,
            cg["s_kv_indices"][:total_shared],
            s_lp_cpu,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.b.page_size,
            causal=False,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            # 跳过 split-K：K 个 beam 行已提供充足并行度，split-K 收益甚微却主导 plan() 开销。
            disable_split_kv=True,
        )
        # 为后续相同长度结构的 batch 快照调度计划（_plan_lru_cap <= 0 时跳过）。
        if cache_enabled:
            lru[lru_key] = (
                shared_wrapper._plan_info,
                shared_wrapper._int_workspace_buffer.clone(),
            )
            lru.move_to_end(lru_key)
            if len(lru) > self._plan_lru_cap:
                lru.popitem(last=False)
        logger.debug(
            "beam cascade plan cache MISS (bs=%d, plan_tag=%s, lru_size=%d, "
            "cache_enabled=%s) -> ran prefill scheduler",
            bs,
            plan_tag,
            len(lru),
            cache_enabled,
        )
        nvtx.range_pop()

    def _plan_private_stage(
        self,
        private_wrapper: BatchDecodeWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        beam_shared_lens: torch.Tensor,
        shared_cpu: torch.Tensor,
        bs: int,
        device: torch.device,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """规划私有阶段 decode wrapper（每 beam 行仅关注其私有后缀 [shared:seq]）。"""
        cg = self._cuda_graph_kv_buffers
        nvtx.range_push("cascade:plan_private")
        private_lens = seq_lens - beam_shared_lens.to(device, seq_lens.dtype)
        private_lens_cpu = (
            (seq_lens_cpu[:bs].reshape(-1) - shared_cpu).clamp(min=0)
            if seq_lens_cpu is not None
            else None
        )
        p_total = (
            int(private_lens_cpu.sum().item())
            if private_lens_cpu is not None
            else int(private_lens.sum().item())
        )
        self.b.indices_updater_decode.call_begin_forward(
            private_wrapper,
            req_pool_indices,
            private_lens,
            p_total,
            cg["p_kv_indptr"],
            beam_shared_lens.to(seq_lens.dtype),
            None,
            private_lens_cpu.to(torch.int32) if private_lens_cpu is not None else None,
        )
        nvtx.range_pop()

    # ══════════════════════════════════════════════════════════════════════
    # Decode 时执行
    # ══════════════════════════════════════════════════════════════════════

    def run_decode(self, layer, q_reshaped: torch.Tensor, kv_buffer):
        """执行混合级联 decode 并合并结果。

        级联一旦开启，所有 decode 批次均走此路径（非 beam 批次使用退化的 shared_len=0）。
        """
        md = self.b.forward_metadata
        shared_w = md.cascade_shared_wrapper
        private_w = md.cascade_private_wrapper

        # 每层 sm_scale / logits_soft_cap（对 CUDA graph 安全：仅写 Python 属性）。
        sm_scale = layer.scaling
        if layer.k_scale_float is not None:
            sm_scale *= layer.k_scale_float
        logit_cap = layer.logit_cap or 0.0
        shared_w._sm_scale = private_w._sm_scale = sm_scale
        shared_w._logits_soft_cap = private_w._logits_soft_cap = logit_cap

        o_shared, lse_shared = shared_w.run(q_reshaped, kv_buffer, return_lse=True)
        o_priv, lse_priv = private_w.run(
            q_reshaped, kv_buffer, v_scale=layer.v_scale_float, return_lse=True
        )
        o, _ = merge_state(o_shared, lse_shared, o_priv, lse_priv)
        return o
