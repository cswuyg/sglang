# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mixin classes for beam search operations in ScheduleBatch and Req."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import torch.cuda.nvtx as nvtx

from sglang.srt.managers.beam_search_type import BeamSearchList
from sglang.srt.mem_cache.allocation import alloc_for_decode
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class ScheduleBatchBeamSearchMixin:
    """Mixin class for beam search related operations in ScheduleBatch."""

    def _init_beam_cascade_config(self) -> None:
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
        self._beam_cascade_attention_enabled = server_args is not None and getattr(
            server_args, "enable_beam_cascade_attention", False
        )
        self._beam_cascade_shared_step = server_args.beam_cascade_shared_step

    def _fork_beam_mamba_states(
        self: ScheduleBatch,
        new_reqs: List["Req"],
        beam_req_pool_indices: torch.Tensor,
    ) -> None:
        """Allocate Mamba slots for new beam branches using copy-on-write reads."""
        nvtx.range_push("beam_search:fork_mamba_states")
        mapping_dtype = self.req_to_token_pool.req_index_to_mamba_index_mapping.dtype
        src_mamba_idx = torch.cat(
            [req.mamba_pool_idx.repeat(req.beam_width) for req in new_reqs]
        ).to(dtype=mapping_dtype)

        self.req_to_token_pool.set_beam_cow_read_mapping(
            beam_req_pool_indices, src_mamba_idx
        )

        total_slots = beam_req_pool_indices.numel()
        dst_mamba_idx = self.req_to_token_pool.mamba_allocator.alloc(total_slots)
        if dst_mamba_idx is None:
            raise RuntimeError(
                "Out of mamba cache space for beam search. Please increase "
                "--max-mamba-cache-size / --mamba-full-memory-ratio, reduce "
                "--max-running-requests / beam width, or disable "
                "--enable-beam-mamba-double-buffer."
            )
        self.req_to_token_pool.req_index_to_mamba_index_mapping[
            beam_req_pool_indices
        ] = dst_mamba_idx.to(device=self.device, dtype=mapping_dtype)

        nvtx.range_pop()

    def prepare_for_beam_search_decode(self: ScheduleBatch):
        """Prepare batch for beam search decode phase.

        This method sets up the batch for beam search decoding by:
        1. Collecting last tokens from all beam branches
        2. Initializing new beam search requests (allocating KV cache slots)
        3. Allocating output cache locations
        4. Updating sequence lengths
        """
        nvtx.range_push("beam_search:prepare_decode")
        self.forward_mode = ForwardMode.DECODE

        beam_ids = torch.cat(
            [req.beam_list.last_tokens.to(torch.int32) for req in self.reqs]
        )

        # beam search not need self.output_ids
        self.input_ids = beam_ids
        self.output_ids = None

        self._prepare_for_new_beam_search()

        # alloc_for_decode updates req.kv.kv_allocated_len for the normal decode
        # ownership model. Beam decode allocates into separate per-beam pool rows;
        # the logical request's original row retains only its prefill KV and is
        # released later by release_kv_cache(). Do not charge beam allocations to
        # that original row, otherwise finish cleanup either asserts or double-frees.
        original_kv_allocated_lens = [req.kv.kv_allocated_len for req in self.reqs]
        self.out_cache_loc = alloc_for_decode(self, token_per_req=1)
        for req, allocated_len in zip(self.reqs, original_kv_allocated_lens):
            req.kv.kv_allocated_len = allocated_len

        self.seq_lens.add_(1)
        self.seq_lens_cpu.add_(1)
        self.orig_seq_lens.add_(1)
        self.seq_lens_sum = self.seq_lens.sum().item()

        self._compute_beam_shared_lens()
        nvtx.range_pop()

    def _compute_beam_shared_lens(self: ScheduleBatch) -> None:
        """计算每个 beam 行的共享前缀长度，用于 cascade attention。"""
        if not self._beam_cascade_attention_enabled:
            self.beam_shared_lens = None
            self.beam_cascade_enabled = False
            return

        # 搜索推荐场景下 beam search 生成 token 数很少（prompt >> decode），
        # 因此直接用 prompt_len 作为共享前缀长度即可覆盖带宽节省的大头
        nvtx.range_push("beam_cascade:compute_shared_lens")
        prompt_lens = torch.cat([req.beam_list.prompt_lens for req in self.reqs]).to(
            dtype=torch.int32, device=self.device
        )

        # 向下取整到 step 的整数倍，以复用跨 batch 的 plan 缓存
        step = self._beam_cascade_shared_step
        if step > 1:
            shared_lens = (prompt_lens // step) * step
        else:
            shared_lens = prompt_lens

        self.beam_shared_lens = shared_lens
        self.beam_cascade_enabled = True
        logger.info("use beam cascade attention")
        nvtx.range_pop()

    def filter_beam_search_batch(
        self: ScheduleBatch,
        chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        """Filter beam search batch to keep only specified requests.

        This method handles the special filtering logic for beam search batches,
        where each request occupies beam_width slots in the batch tensors.

        Args:
            chunked_req_to_exclude: Requests to exclude from the batch
            keep_indices: Indices of requests to keep (if None, computed from chunked_req_to_exclude)
        """
        nvtx.range_push("beam_search:filter_batch")
        if keep_indices is None:
            if chunked_req_to_exclude is not None and not isinstance(
                chunked_req_to_exclude, list
            ):
                chunked_req_to_exclude = [chunked_req_to_exclude]
            elif chunked_req_to_exclude is None:
                chunked_req_to_exclude = []
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] not in chunked_req_to_exclude
            ]

        if keep_indices is None or len(keep_indices) == 0:
            self.reqs = []
            nvtx.range_pop()
            return

        if len(keep_indices) == len(self.reqs):
            nvtx.range_pop()
            return

        # Filter penalizer states before filtering self.reqs
        self._filter_penalizer_states(keep_indices)

        old_pool_indices = []
        # old_pool_indices 的 CPU 端镜像（每个请求的 beam 槽位索引列表），
        # 用于过滤 Python 侧的列表（如 multimodal_inputs），避免把张量从设备拷回 CPU。
        old_pool_slots: List[List[int]] = []
        extend_idx = 0
        for req in self.reqs:
            if req.beam_list.batch_slot_start_idx != -1:
                slots = [
                    req.beam_list.batch_slot_start_idx + i
                    for i in range(req.beam_width)
                ]
                old_pool_indices.append(
                    torch.tensor(slots, dtype=torch.int64, device=self.device)
                )
                old_pool_slots.append(slots)
                extend_idx += req.beam_width
            else:
                old_pool_indices.append(
                    torch.tensor([extend_idx], dtype=torch.int64, device=self.device)
                )
                old_pool_slots.append([extend_idx])
                extend_idx += 1
        keep_pool_indices = torch.concat([old_pool_indices[i] for i in keep_indices])
        self.reqs = [self.reqs[i] for i in keep_indices]

        old_pool_indices_for_debug = torch.concat(old_pool_indices)
        assert len(self.req_pool_indices) == len(old_pool_indices_for_debug)
        old_to_new_pool_indices = torch.arange(
            len(self.req_pool_indices), dtype=torch.int64, device=self.device
        )
        new_pool_indices = torch.arange(
            len(keep_pool_indices), dtype=torch.int64, device=self.device
        )
        old_to_new_pool_indices[keep_pool_indices] = new_pool_indices
        for req in self.reqs:
            if req.beam_list.batch_slot_start_idx != -1:
                req.beam_list.batch_slot_start_idx = old_to_new_pool_indices[
                    req.beam_list.batch_slot_start_idx
                ].item()

        self.req_pool_indices = self.req_pool_indices[keep_pool_indices]
        self.seq_lens = self.seq_lens[keep_pool_indices]
        self.seq_lens_cpu = self.seq_lens.cpu()
        self.seq_lens_sum = self.seq_lens.sum().item()
        self.orig_seq_lens = self.orig_seq_lens[keep_pool_indices]

        # multimodal_inputs 与扩展后的 beam 槽位维度对齐（见 _prepare_for_new_beam_search），
        # 因此必须用相同的 beam 槽位索引进行过滤，才能与 seq_lens 保持对齐。
        # 复用 CPU 端的 old_pool_slots，避免把 keep_pool_indices 从设备拷回 CPU。
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = [
                self.multimodal_inputs[slot]
                for i in keep_indices
                for slot in old_pool_slots[i]
            ]

        # filter 后 out_cache_loc 失效，下次 prepare_for_beam_search_decode 会重新分配
        self.out_cache_loc = None
        # beam search 用 beam_list.last_tokens 而非 output_ids
        self.output_ids = None
        # beam search 强制 return_logprob=False（见 scheduler 创建 Req 时），
        # 故 token_ids_logprobs 恒为 None，下面的过滤是与通用 filter 对齐的防御性写法
        self.return_logprob = False
        if self.token_ids_logprobs is not None:
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]

        self.has_stream = any(req.stream for req in self.reqs)
        self.has_grammar = any(req.grammar for req in self.reqs)
        # TODO(cswuyg) beam search 不支持 spec info
        nvtx.range_pop()

    def _filter_penalizer_states(self: ScheduleBatch, keep_indices: List[int]):
        """
        Filter penalizer states to match the filtered beam search requests.

        This method calculates which beam indices to keep in the penalizer state
        based on which requests are being kept. Penalizer states are organized by beam:
        [req0_beam0, req0_beam1, ..., req1_beam0, req1_beam1, ...]

        Args:
            keep_indices: List of request indices to keep (before filtering self.reqs)
        """
        if not self.sampling_info or not self.sampling_info.penalizer_orchestrator:
            return

        # Calculate penalizer keep indices BEFORE filtering self.reqs
        penalizer_keep_indices_list = []
        current_offset = 0

        for original_idx, req in enumerate(self.reqs):
            if original_idx in keep_indices:
                penalizer_keep_indices_list.extend(
                    range(current_offset, current_offset + req.beam_width)
                )
            current_offset += req.beam_width

        penalizer_keep_indices = torch.tensor(
            penalizer_keep_indices_list, dtype=torch.int64, device=self.device
        )

        self.sampling_info.penalizer_orchestrator.filter(penalizer_keep_indices)

    def _prepare_for_new_beam_search(self: ScheduleBatch):
        """
        Initialize beam search inference for new requests.

        Features:
            1. Allocate req_to_token slots for all beam branches (beam_width slots per request)
            2. Copy KV cache from normal slot to all beam branches in parallel
            3. Record the beam request slots in req_to_token to the local req_pool_indices
            4. Extend seq_lens and orig_seq_lens, replicating the same sequence length for each beam branch
            5. Set batch_slot_start_idx for each request (pointing to the request's starting position in the batch)

        Batch req_pool_indices layout for a single beam search request:
            [beam0_pool_idx | beam1_pool_idx | beam2_pool_idx | ... | beam(K-1)_pool_idx]
             └─────────────────────── beam_width slots ───────────────────────┘

            Each pool_idx points to a row in req_to_token_pool.req_to_token that stores
            the KV cache token indices for that beam branch.

        Memory management details:
            - Before extend: batch.req_pool_indices[i] points to the prefill slot (normal_idx)
            - After extend: batch.req_pool_indices[i:i+beam_width] are replaced with new beam slots
            - Important: The original prefill slot in batch.req_pool_indices is OVERWRITTEN and no
              longer accessible through batch.req_pool_indices
            - However: req.req_pool_idx still preserves the original prefill slot index, which is
              used by release_kv_cache() to free the prefill KV cache when the request finishes
            - This design allows both beam search and non-beam search requests to use the same
              cache_finished_req mechanism

        Notes:
            - Supports different beam_width for each request
            - All beam branches share the same prefill KV cache (implemented through parallel copying)
        """
        nvtx.range_push("beam_search:prepare_new_beams")
        new_reqs = []
        old_reqs = []
        for req in self.reqs:
            if req.beam_list.batch_slot_start_idx == -1:
                new_reqs.append(req)
            else:
                old_reqs.append(req)

        if not new_reqs:
            nvtx.range_pop()
            return

        new_pool_slot_list = [req.beam_width for req in new_reqs]
        total_slots = sum(new_pool_slot_list)

        beam_req_pool_indices = self.req_to_token_pool.alloc_by_count(total_slots)
        if beam_req_pool_indices is None:
            raise RuntimeError(
                "Out of memory. Please set a smaller number for `--max-running-requests` or `--beam-width`."
            )
        beam_req_pool_indices = torch.tensor(
            beam_req_pool_indices, dtype=torch.int64, device=self.device
        )

        skip_idx = sum(req.beam_width for req in old_reqs)

        new_req_pool_indices_list = []
        new_seq_lens_list = []
        new_orig_seq_lens_list = []
        new_multimodal_inputs_list = []

        beam_offset = 0
        req_pool = self.req_to_token_pool.req_to_token
        for i, req in enumerate(new_reqs):
            normal_idx = self.req_pool_indices[skip_idx + i : skip_idx + i + 1]
            seq_len_tensor = self.seq_lens[skip_idx + i : skip_idx + i + 1]
            seq_len = seq_len_tensor.squeeze()

            beam_start = beam_offset
            beam_end = beam_offset + req.beam_width

            normal_kvcache = req_pool[normal_idx, :seq_len].squeeze(0)
            beam_indices = beam_req_pool_indices[beam_start:beam_end]
            req_pool[beam_indices, :seq_len] = normal_kvcache

            beam_offset = beam_end

            # Use all beam indices (normal_idx is preserved in req.req_pool_idx for release_kv_cache)
            new_req_pool_indices_list.append(beam_indices)

            expanded_seq_lens = seq_len_tensor.repeat(req.beam_width)
            new_seq_lens_list.append(expanded_seq_lens)

            orig_seq_len = self.orig_seq_lens[skip_idx + i : skip_idx + i + 1]
            expanded_orig_seq_lens = orig_seq_len.repeat(req.beam_width)
            new_orig_seq_lens_list.append(expanded_orig_seq_lens)

            # 为每个 beam 分支复制 multimodal_inputs，使下游（如 mrope 位置计算）
            # 能与扩展后的 batch 维度保持对齐。此处 MultimodalInputs 对象是只读的，
            # 因此在各 beam 之间共享同一引用是安全的。
            if self.multimodal_inputs is not None:
                new_multimodal_inputs_list.extend(
                    [self.multimodal_inputs[skip_idx + i]] * req.beam_width
                )

        new_req_pool_indices = torch.cat(new_req_pool_indices_list)

        if isinstance(self.req_to_token_pool, HybridReqToTokenPool):
            self._fork_beam_mamba_states(new_reqs, new_req_pool_indices)

        self.req_pool_indices = torch.cat(
            [self.req_pool_indices[:skip_idx], new_req_pool_indices]
        )

        new_seq_lens = torch.cat(new_seq_lens_list)
        self.seq_lens = torch.cat([self.seq_lens[:skip_idx], new_seq_lens])
        self.seq_lens_cpu = self.seq_lens.cpu()

        new_orig_seq_lens = torch.cat(new_orig_seq_lens_list)
        self.orig_seq_lens = torch.cat(
            [self.orig_seq_lens[:skip_idx], new_orig_seq_lens]
        )

        if self.multimodal_inputs is not None:
            self.multimodal_inputs = (
                self.multimodal_inputs[:skip_idx] + new_multimodal_inputs_list
            )

        current_idx = skip_idx
        for req in new_reqs:
            req.beam_list.batch_slot_start_idx = current_idx
            current_idx += req.beam_width
        nvtx.range_pop()


class ReqBeamSearchMixin:
    """Mixin class for beam search related operations in Req.

    This mixin provides beam search specific attributes and initialization logic
    that can be mixed into the Req class.
    """

    def _init_beam_search_attributes(self, is_beam_search, sampling_params):
        """Initialize beam search related attributes.

        This method should be called from Req.__init__() to set up beam search state.
        """
        self.is_beam_search = is_beam_search
        if self.is_beam_search:
            # sampling_params.n has already been validated in tokenizermanager
            self.beam_width = sampling_params.n
            self.beam_list = BeamSearchList()
            # Path expansion candidate count. Use a wider candidate pool to better
            # match transformers' full-vocabulary beam expansion on near-tie cases.
            self.beam_candidates = self.beam_width * 2

        self._stop_token_ids_cache: Optional[set] = None

    @property
    def stop_token_ids(self):
        """Get the stop token ids (cached).

        This property is only used in beam search scenarios.
        """
        if self._stop_token_ids_cache is None:
            stop_token_ids = set()
            if self.sampling_params.stop_token_ids:
                stop_token_ids.update(self.sampling_params.stop_token_ids)
            if self.eos_token_ids:
                stop_token_ids.update(self.eos_token_ids)
            if self.tokenizer is not None:
                if self.tokenizer.eos_token_id is not None:
                    stop_token_ids.add(self.tokenizer.eos_token_id)
                if self.tokenizer.additional_stop_token_ids:
                    stop_token_ids.update(self.tokenizer.additional_stop_token_ids)
            self._stop_token_ids_cache = stop_token_ids
        return self._stop_token_ids_cache
