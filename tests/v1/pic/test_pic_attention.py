# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native range attention agrees with dense mathematical attention."""

import pytest
import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.core.pic import PICContentLayout, PICLayout, PICOccurrence
from vllm.v1.worker.gpu.pic import PICInputBuffers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA attention kernel")
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize(
    "rotary_dim,neox,amplitude,use_table",
    [(64, True, 1.0, False), (32, False, 1.4, False), (64, True, 1.0, True)],
)
def test_ordered_ranges_skip_pad_and_future_pic_in_captured_attention(
    rotary_dim, neox, amplitude, use_table, contiguous
):
    torch.manual_seed(18)
    device, dtype = "cuda", torch.bfloat16
    heads, kv_heads, head_size = 4, 2, 64
    # Canonical A5 B3 C2 D2 E4; storage puts compressed B/D before Dynamic Content.
    # Both PICs are already resident when the first Q tile uses only A.
    q_positions = torch.tensor([0, 1, 2, 3, 4, 8, 9, 12, 13, 14, 15], device=device)
    dynamic_content_slots = torch.arange(32, 43, device=device)
    positions = torch.full((48,), -1.0, device=device)
    positions[dynamic_content_slots] = q_positions.float()
    local_b = torch.tensor(
        [0.0, 1.0] if contiguous else [0.0 if use_table else 0.5, 2.0], device=device
    )
    local_d = torch.tensor([0.0 if use_table or contiguous else 0.75], device=device)
    positions[:2] = local_b + 5
    positions[16] = local_d + 10
    contents = [
        PICContentLayout(
            len(values), {} if contiguous else {"layer.0": values.tolist()}
        )
        for values in (local_b, local_d)
    ]
    buffers = PICInputBuffers.from_layout(
        PICLayout(
            canonical_length=16,
            storage_length=43,
            dynamic_content_ranges=[(32, 0, 5), (37, 8, 2), (39, 12, 4)],
            occurrences=[
                PICOccurrence(5, 0, [(0, 0, 16)]),
                PICOccurrence(10, 1, [(16, 0, 16)]),
            ],
            contents=contents,
            layer_names=["layer.0"],
        ),
        0,
        torch.device(device),
    )
    ranges = buffers.ranges["layer.0"]
    frequencies = 10000 ** (
        -torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim
    )

    table_angles = torch.arange(128, device=device)[:, None] * frequencies[None]
    table = (
        torch.cat((table_angles.cos(), table_angles.sin()), -1).to(dtype)
        if use_table
        else None
    )

    def rotate(tensor, pos):
        angles = pos[:, None] * frequencies[None]
        active = tensor.float()[..., :rotary_dim]
        if neox:
            cosine = torch.cat((angles.cos(), angles.cos()), -1)[:, None]
            sine = torch.cat((angles.sin(), angles.sin()), -1)[:, None]
            left, right = active.chunk(2, -1)
            partner = torch.cat((-right, left), -1)
        else:
            cosine = angles.cos().repeat_interleave(2, -1)[:, None]
            sine = angles.sin().repeat_interleave(2, -1)[:, None]
            partner = torch.stack((-active[..., 1::2], active[..., ::2]), -1).flatten(
                -2
            )
        if use_table:
            cosine, sine = cosine.to(dtype).float(), sine.to(dtype).float()
        rotated = (active * cosine + partner * sine) * amplitude
        return torch.cat((rotated, tensor.float()[..., rotary_dim:]), -1).to(dtype)

    raw_query = torch.randn((11, heads, head_size), device=device, dtype=dtype)
    query = rotate(raw_query, q_positions)
    key = torch.randn((48, kv_heads, head_size), device=device, dtype=dtype)
    value = torch.randn_like(key)
    rotated_key = rotate(key, positions)
    cached_key = key.view(3, 16, kv_heads, head_size)
    values = value.view_as(cached_key)
    output = torch.empty_like(query)
    starts = torch.tensor([0, 11], dtype=torch.int32, device=device)
    lengths = torch.tensor([14], dtype=torch.int32, device=device)
    blocks = torch.arange(3, dtype=torch.int32, device=device)[None]
    pointers = torch.zeros(1, dtype=torch.int64, device=device)
    counts = torch.tensor([5], dtype=torch.int32, device=device)
    one = torch.tensor(1.0, device=device)

    def run_attention():
        unified_attention(
            query,
            cached_key,
            values,
            output,
            starts,
            11,
            lengths,
            14,
            head_size**-0.5,
            True,
            (-1, -1),
            blocks,
            0.0,
            one,
            one,
            one,
            seq_threshold_3D=0,
            logical_ranges=pointers,
            logical_range_counts=counts,
            query_positions=q_positions,
            cos_sin_cache=table,
            rope_inv_freq=frequencies,
            rope_is_neox_style=neox,
            rope_attention_scaling=amplitude,
        )

    def oracle(keys, visible):
        keys = keys.repeat_interleave(heads // kv_heads, 1)
        scores = (
            torch.einsum("qhd,khd->hqk", query.float(), keys.float()) * head_size**-0.5
        )
        scores.masked_fill_(~visible[None], -torch.inf)
        return torch.einsum(
            "hqk,khd->qhd",
            scores.softmax(-1),
            value.repeat_interleave(heads // kv_heads, 1).float(),
        )

    run_attention()
    ordinary_visible = (
        torch.arange(48, device=device)[None]
        <= torch.arange(3, 14, device=device)[:, None]
    )
    torch.testing.assert_close(
        output.float(),
        oracle(rotate(key, torch.arange(48, device=device)), ordinary_visible),
        atol=0.025,
        rtol=0.025,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_attention()
    pointers.fill_(ranges.data_ptr())
    graph.replay()
    visible = (positions[None] >= 0) & (positions[None] <= q_positions[:, None])
    torch.testing.assert_close(
        output.float(), oracle(rotated_key, visible), atol=0.025, rtol=0.025
    )

    # Repeated placements share two physical entries, but contribute separately
    # to attention. Logical visits (75) exceed native storage length (43).
    repeated_buffers = PICInputBuffers.from_layout(
        PICLayout(
            canonical_length=107,
            storage_length=43,
            dynamic_content_ranges=[(32, 96, 11)],
            occurrences=[PICOccurrence(3 * i, 0, [(0, 0, 16)]) for i in range(32)],
            contents=contents[:1],
            layer_names=["layer.0"],
        ),
        0,
        torch.device(device),
    )
    repeated_ranges = repeated_buffers.ranges["layer.0"]
    q_positions.copy_(torch.arange(96, 107, device=device))
    new_dynamic_content = rotate(key[dynamic_content_slots], q_positions)
    keys = torch.cat(
        [rotate(key[:2], local_b + 3 * i) for i in range(32)] + [new_dynamic_content]
    )
    expanded_values = torch.cat([value[:2]] * 32 + [value[dynamic_content_slots]])
    expanded_positions = torch.cat([local_b + 3 * i for i in range(32)] + [q_positions])
    pointers.fill_(repeated_ranges.data_ptr())
    counts.fill_(33)
    lengths.fill_(43)
    graph.replay()
    scores = (
        torch.einsum(
            "qhd,khd->hqk", query.float(), keys.repeat_interleave(2, 1).float()
        )
        / 8
    )
    scores.masked_fill_(
        expanded_positions[None, None] > q_positions[None, :, None], -torch.inf
    )
    expected = torch.einsum(
        "hqk,khd->qhd",
        scores.softmax(-1),
        expanded_values.repeat_interleave(2, 1).float(),
    )
    torch.testing.assert_close(output.float(), expected, atol=0.025, rtol=0.025)

    # Decode uses the original segmented attention/reduction path as well.
    partial_output = torch.empty((1, heads, 16, head_size), device=device)
    partial_max = torch.empty((1, heads, 16), device=device)
    partial_sum = torch.empty_like(partial_max)
    decode_output = torch.empty_like(query[-1:])
    unified_attention(
        query[-1:],
        cached_key,
        values,
        decode_output,
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        1,
        lengths,
        43,
        head_size**-0.5,
        True,
        (-1, -1),
        blocks,
        0.0,
        one,
        one,
        one,
        seq_threshold_3D=128,
        num_par_softmax_segments=16,
        softmax_segm_output=partial_output,
        softmax_segm_max=partial_max,
        softmax_segm_expsum=partial_sum,
        logical_ranges=pointers,
        logical_range_counts=counts,
        query_positions=q_positions[-1:],
        cos_sin_cache=table,
        rope_inv_freq=frequencies,
        rope_is_neox_style=neox,
        rope_attention_scaling=amplitude,
    )
    torch.testing.assert_close(
        decode_output.float(), expected[-1:], atol=0.025, rtol=0.025
    )
