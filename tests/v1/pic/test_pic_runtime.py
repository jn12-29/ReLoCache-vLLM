# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Logical attention coordinates stay independent of native storage order."""

import torch

from vllm.v1.core.pic import PICContentLayout, PICLayout, PICOccurrence
from vllm.v1.worker.gpu.pic import PICInputBuffers


def test_contiguous_mapping_shares_layers_and_trims_reordered_tail_pages():
    layout = PICLayout(
        canonical_length=13,
        storage_length=11,
        dynamic_content_ranges=[(8, 0, 1), (9, 6, 1), (10, 12, 1)],
        occurrences=[
            PICOccurrence(start, 0, [(4, 0, 4), (0, 4, 4)]) for start in (1, 7)
        ],
        contents=[PICContentLayout(5, {})],
        layer_names=["layer.0", "layer.1"],
    )
    buffers = PICInputBuffers.from_layout(layout, 2, torch.device("cpu"))
    assert not buffers.entry_positions
    assert buffers.ranges["layer.0"] is buffers.ranges["layer.1"]
    assert buffers.ranges["layer.0"].tolist() == [
        [1, 8, 0, 0],
        [5, 4, 1, 0],
        [6, 0, 5, 0],
        [7, 9, 6, 0],
        [11, 4, 7, 0],
        [12, 0, 11, 0],
        [13, 10, 12, 0],
        [15, 11, 13, 0],
    ]


def test_reordered_storage_reuses_pages_with_distinct_logical_placements():
    content = PICContentLayout(
        2,
        {
            "layer.0": [0.5],
            "layer.1": [0.0, 0.5, 1.0, 1.5],
        },
    )
    layout = PICLayout(
        contents=[content],
        layer_names=["layer.0", "layer.1"],
        canonical_length=9,
        storage_length=9,
        dynamic_content_ranges=[(0, 0, 1), (1, 3, 2), (3, 7, 1), (8, 8, 1)],
        occurrences=[PICOccurrence(start, 0, [(4, 0, 4)]) for start in (1, 5)],
    )
    buffers = PICInputBuffers.from_layout(layout, 3, torch.device("cpu"))
    assert buffers.positions[[0, 1, 2, 3, 8, 9, 10, 11]].tolist() == [
        0,
        3,
        4,
        7,
        8,
        9,
        10,
        11,
    ]
    assert len(buffers.entry_positions) == 2
    short, long = buffers.ranges["layer.0"], buffers.ranges["layer.1"]
    assert short[:, 0].tolist() == [1, 2, 4, 5, 6, 7, 10]
    assert long[:, 0].tolist() == [1, 5, 7, 11, 12, 13, 16]
    assert long[:, 1].tolist() == [0, 4, 1, 4, 3, 8, 9]
    assert long[:, 2].tolist() == [0, 1, 3, 5, 7, 8, 9]
    assert long[1, 3] == long[3, 3] != 0
    assert (long[[0, 2, 4, 5, 6], 3] == 0).all()


def test_pic_segments_trim_layer_padding_and_offset_fractional_entries():
    content = PICContentLayout(2, {"layer.0": [0.0, 0.25, 0.5, 1.0, 1.5]})
    layout = PICLayout(
        contents=[content],
        layer_names=["layer.0"],
        canonical_length=4,
        storage_length=10,
        dynamic_content_ranges=[(8, 0, 1), (9, 3, 1)],
        occurrences=[PICOccurrence(1, 0, [(4, 0, 4), (0, 4, 4)])],
    )
    buffers = PICInputBuffers.from_layout(layout, 2, torch.device("cpu"))
    rows = buffers.ranges["layer.0"]
    assert rows[:, :3].tolist() == [
        [1, 8, 0],
        [5, 4, 1],
        [6, 0, 1],
        [7, 9, 3],
        [9, 10, 4],
    ]
    assert rows[2, 3].item() - rows[1, 3].item() == 4 * 4
    assert buffers.positions[[8, 9, 10, 11]].tolist() == [0, 3, 4, 5]


def test_resident_frequencies_match_standard_and_llama3_rotary_outputs():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    from vllm.model_executor.layers.rotary_embedding.llama3_rope import (
        Llama3RotaryEmbedding,
    )

    options = {
        "head_size": 8,
        "rotary_dim": 8,
        "max_position_embeddings": 64,
        "base": 10000,
        "is_neox_style": True,
        "dtype": torch.float32,
    }
    with set_current_vllm_config(VllmConfig()):
        modules = [
            RotaryEmbedding(**options),
            Llama3RotaryEmbedding(
                **options,
                scaling_factor=8,
                low_freq_factor=1,
                high_freq_factor=4,
                orig_max_position=16,
            ),
        ]
    positions = torch.tensor([0, 3, 17])
    query = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24
    for rotary in modules:
        actual, _ = rotary.forward_native(positions, query.clone(), query.clone())
        phase = positions.float()[:, None] * rotary.inv_freq
        cos, sin = phase.cos(), phase.sin()
        first, second = query.chunk(2, dim=-1)
        expected = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )
        torch.testing.assert_close(actual, expected * rotary.attention_scaling)
