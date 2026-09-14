# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical input coordinates over native cache storage; no cache ownership."""

from copy import copy
from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.core.pic import PICLayout
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, async_copy_to_gpu


@dataclass
class PICInputBuffers:
    positions: torch.Tensor
    ranges: dict[str, torch.Tensor]
    # Range descriptors contain pointers into these tensors.
    entry_positions: dict[tuple[int, str], torch.Tensor]

    @classmethod
    def from_layout(cls, layout: PICLayout, max_tokens: int, device: torch.device):
        positions = np.zeros(layout.storage_length + max_tokens, dtype=np.int64)
        inline = list(layout.inline_ranges)
        inline.append((layout.storage_length, layout.canonical_length, max_tokens))
        for storage_start, logical_start, length in inline:
            positions[storage_start : storage_start + length] = np.arange(
                logical_start, logical_start + length, dtype=np.int64
            )

        ranges = {}
        entry_positions = {
            (index, name): torch.tensor(values, dtype=torch.float32, device=device)
            for index, content in enumerate(layout.contents)
            for name, values in content.entry_positions.items()
        }
        contiguous_ranges = None
        for layer_name in layout.layer_names:
            contiguous = all(
                layer_name not in content.entry_positions for content in layout.contents
            )
            if contiguous and contiguous_ranges is not None:
                ranges[layer_name] = contiguous_ranges
                continue
            descriptors = [
                (logical, storage, length, 0)
                for storage, logical, length in inline
                if length
            ]
            for occurrence in layout.occurrences:
                content = layout.contents[occurrence.content_index]
                entries = entry_positions.get((occurrence.content_index, layer_name))
                num_entries = content.length if entries is None else entries.numel()
                for storage, offset, capacity in occurrence.segments:
                    length = min(capacity, num_entries - offset)
                    if length > 0:
                        descriptors.append(
                            (
                                occurrence.start + offset
                                if entries is None
                                else occurrence.start,
                                storage,
                                length,
                                0
                                if entries is None
                                else entries.data_ptr()
                                + offset * entries.element_size(),
                            )
                        )
            descriptors.sort(key=lambda item: item[0])
            cumulative = 0
            rows = []
            for logical, storage, length, pointer in descriptors:
                cumulative += length
                rows.append((cumulative, storage, logical, pointer))
            ranges[layer_name] = torch.tensor(rows, dtype=torch.int64, device=device)
            if contiguous:
                contiguous_ranges = ranges[layer_name]
        return cls(torch.as_tensor(positions, device=device), ranges, entry_positions)


class PICModelState:
    def __init__(self, max_num_reqs: int, max_num_tokens: int, device: torch.device):
        self.device = device
        self.max_num_reqs = max_num_reqs
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.position_map_ptrs = StagedWriteTensor(
            max_num_reqs, dtype=torch.int64, device=device
        )
        self.requests: dict[str, PICInputBuffers] = {}
        self.pointer_tables: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.rotary_parameters: dict[str, tuple | None] = {}

    def add_request(self, req_index, request):
        pointer = 0
        if request.pic_layout is not None:
            buffers = PICInputBuffers.from_layout(
                request.pic_layout, request.sampling_params.max_tokens, self.device
            )
            self.requests[request.req_id] = buffers
            pointer = buffers.positions.data_ptr()
        self.position_map_ptrs.stage_write_elem(req_index, pointer)

    def remove_request(self, req_id):
        self.requests.pop(req_id, None)

    def apply_staged_writes(self):
        self.position_map_ptrs.apply_write()

    def prepare_metadata(self, metadata, batch, model):
        result = metadata.copy()
        requests = [
            (index, self.requests[req_id])
            for index, req_id in enumerate(batch.req_ids)
            if req_id in self.requests
        ]
        for layer_name, layer_metadata in metadata.items():
            if not isinstance(layer_metadata, TritonAttentionMetadata):
                continue
            if layer_name not in self.rotary_parameters:
                attention = model.get_submodule(layer_name.rsplit(".", 1)[0])
                rotary = getattr(attention, "rotary_emb", None)
                self.rotary_parameters[layer_name] = (
                    (
                        rotary.inv_freq.to(device=self.device, dtype=torch.float32),
                        rotary.is_neox_style,
                        rotary.attention_scaling,
                        rotary.cos_sin_cache,
                    )
                    if rotary is not None
                    and hasattr(rotary, "inv_freq")
                    and rotary.position_free_kv
                    else None
                )
            rotary_parameters = self.rotary_parameters[layer_name]
            if rotary_parameters is None:
                if requests:
                    raise NotImplementedError(
                        f"PIC requires resident rotary frequencies for {layer_name}"
                    )
                continue
            if layer_name not in self.pointer_tables:
                self.pointer_tables[layer_name] = (
                    torch.zeros(
                        self.max_num_reqs, dtype=torch.int64, device=self.device
                    ),
                    torch.zeros(
                        self.max_num_reqs, dtype=torch.int32, device=self.device
                    ),
                )
            pointers = np.zeros(self.max_num_reqs, dtype=np.int64)
            counts = np.zeros(self.max_num_reqs, dtype=np.int32)
            for index, buffers in requests:
                ranges = buffers.ranges[layer_name]
                pointers[index] = ranges.data_ptr()
                counts[index] = ranges.shape[0]
            pointer_table, count_table = self.pointer_tables[layer_name]
            async_copy_to_gpu(pointers, out=pointer_table)
            async_copy_to_gpu(counts, out=count_table)
            layer_metadata = copy(layer_metadata)
            num_reqs = layer_metadata.seq_lens.shape[0]
            layer_metadata.logical_ranges = pointer_table[:num_reqs]
            layer_metadata.logical_range_counts = count_table[:num_reqs]
            layer_metadata.query_positions = self.positions[
                : batch.num_tokens_after_padding
            ]
            (
                layer_metadata.rope_inv_freq,
                layer_metadata.rope_is_neox_style,
                layer_metadata.rope_attention_scaling,
                layer_metadata.cos_sin_cache,
            ) = rotary_parameters
            result[layer_name] = layer_metadata
        return result
