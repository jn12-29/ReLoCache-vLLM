# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native PIC request construction and metadata, without KV ownership."""

from typing import Any, cast

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheGroupSpec


def build_reusable_metadata(
    metadata: dict[str, Any], groups: list[KVCacheGroupSpec]
) -> dict[str, Any]:
    """Derive native page keys from persisted metadata without reading KV."""
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.v1.core.kv_cache_utils import BlockHash, hash_block_tokens

    block_size = groups[0].kv_cache_spec.block_size
    num_slots = round_up(
        max(layer["physical_length"] for layer in metadata["layers"].values()),
        block_size,
    )
    metadata = dict(metadata)
    metadata["compiled_id"] = sha256(
        (metadata["method_id"], metadata["input_ids"])
    ).hex()
    # Restart the causal prefix at the PIC boundary, in the Writer namespace.
    parent_hash = sha256(
        (
            "reusable_content",
            metadata["method_id"],
            block_size,
            tuple(
                (index, layer["dtype"])
                for index, layer in sorted(metadata["layers"].items())
            ),
        )
    )
    hashes = []
    for start in range(0, num_slots, block_size):
        # Keep the true tail length: alignment zeros are not document tokens.
        parent_hash = hash_block_tokens(
            sha256,
            cast(BlockHash, parent_hash),
            metadata["input_ids"][start : start + block_size],
        )
        hashes.append(parent_hash)
    metadata["groups"] = [
        {
            "block_hashes": hashes,
            "num_slots": num_slots,
            "block_size": block_size,
            "layers": {
                name: metadata["layers"][str(extract_layer_index(name))]
                for name in group.layer_names
            },
        }
        for group in groups
    ]
    return metadata


def compile_inputs(content, *, internal=False):
    writer = content["writer"]
    return (
        content["input_ids"],
        SamplingParams(
            max_tokens=1,
            temperature=0,
            detokenize=False,
            extra_args={"pic_compile": content, "pic_internal": internal},
        ),
        LoRARequest(**writer) if writer is not None else None,
    )


def compile_token_ids(ids: list[int], block_size: int) -> list[int]:
    # A final observation row lets native prefix lookup reuse every PIC
    # page, including the last full page. It is not part of the PIC hash.
    return ids + [0] * (round_up(len(ids), block_size) - len(ids) + 1)


def prepare_content(payload, model, groups):
    """Map a service-owned compilation descriptor to native page metadata."""
    ids = payload["input_ids"]
    metadata = {
        "content_id": payload["content_id"],
        "input_ids": ids,
        "logical_length": len(ids),
        "method_id": payload["method_id"],
        "writer": payload["writer"],
        "layers": {
            str(index): {
                "entry_positions": None,
                "physical_length": len(ids),
                "dtype": str(model.dtype),
            }
            for index in range(model.hf_text_config.num_hidden_layers)
        },
    }
    return build_reusable_metadata(metadata, groups)
