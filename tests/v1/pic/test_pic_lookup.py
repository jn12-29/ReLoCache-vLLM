# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reusable hits enter the real native prefix allocation lifecycle."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.pic import (
    ReusableSpan,
    prepare_reusable_request,
    require_reusable_prefix,
)
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.kv_offload.base import OffloadingKVEventsConfig
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus


@pytest.fixture
def manager():
    init_none_hash(sha256)
    config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer.0"],
                FullAttentionSpec(
                    block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
        ],
    )
    return KVCacheManager(
        config, max_model_len=64, scheduler_block_size=4, hash_block_size=4
    )


def request(name, tokens, spans=None, **kwargs):
    return Request(
        name,
        tokens,
        SamplingParams(max_tokens=2),
        None,
        reusable_spans=spans,
        block_hasher=get_request_block_hasher(4, sha256),
        **kwargs,
    )


def cache(manager, req):
    manager.allocate_slots(req, req.num_tokens)
    manager.free(req)


def content(block_hash, compiled_id="pic"):
    return {
        "compiled_id": compiled_id,
        "method_id": "identity",
        "logical_length": 4,
        "groups": [
            {
                "block_hashes": [block_hash],
                "num_slots": 4,
                "layers": {"layer.0": {"entry_positions": None}},
            }
        ],
    }


def native_content(manager, name, tokens, method="writer", *, dtype=torch.float32):
    from vllm.v1.engine.pic_compiler import prepare_content

    model = SimpleNamespace(
        dtype=dtype, hf_text_config=SimpleNamespace(num_hidden_layers=1)
    )
    return prepare_content(
        {"content_id": name, "input_ids": tokens, "method_id": method, "writer": None},
        model,
        manager.kv_cache_config.kv_cache_groups,
    )


@pytest.fixture
def input_processor(tmp_path):
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.v1.engine.input_processor import InputProcessor

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 128,
                "max_position_embeddings": 64,
            }
        )
    )
    return InputProcessor(
        VllmConfig(
            model_config=ModelConfig(
                model=str(tmp_path),
                skip_tokenizer_init=True,
                dtype="float32",
                max_model_len=64,
                enable_prompt_embeds=True,
            ),
            cache_config=CacheConfig(block_size=4),
        )
    )


@pytest.mark.parametrize("num_tokens", [3, 4, 5])
@pytest.mark.parametrize("soft_lengths", [(0, 0), (2, 3)])
def test_engine_builds_pages_from_sources_and_transports_only_layout(
    manager,
    input_processor,
    num_tokens,
    soft_lengths,
):
    from vllm.v1.core.pic import PICLayout, ReusableInput
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.engine.pic_compiler import compile_inputs
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    engine = EngineCore.__new__(EngineCore)
    engine.mm_receiver_cache = None
    engine.vllm_config = input_processor.vllm_config
    engine.scheduler = SimpleNamespace(kv_cache_config=manager.kv_cache_config)
    engine.request_block_hasher = get_request_block_hasher(4, sha256)
    source: dict[str, Any] = {
        "content_id": "doc",
        "input_ids": list(range(20, 20 + num_tokens)),
        "method_id": "identity",
        "writer": None,
    }
    pre, post = soft_lengths
    if pre or post:
        source["writer_soft_tokens"] = {
            "pre": torch.full((pre, 64), 0.25),
            "post": torch.full((post, 64), 0.75),
        }
    ids, params, _ = compile_inputs(source)
    params.extra_args["custom_sampler"] = "preserved"
    wire = input_processor.process_inputs(
        "writer",
        {"type": "token", "prompt_token_ids": ids},
        params,
        ("generate",),
    )
    length = pre + num_tokens + post
    capacity = (length + 3) // 4 * 4
    assert wire.prompt_token_ids == [0] * pre + ids + [0] * (
        capacity - pre - num_tokens + 1
    )
    assert ids == source["input_ids"]
    decoder = MsgpackDecoder(EngineCoreRequest)
    writer, _ = engine.preprocess_add_request(
        decoder.decode(MsgpackEncoder().encode(wire))
    )
    assert writer.pic_compile["groups"][0]["num_slots"] == capacity
    assert writer.sampling_params.extra_args == {"custom_sampler": "preserved"}
    if pre or post:
        assert writer.prompt_is_token_ids == (
            [False] * pre
            + [True] * num_tokens
            + [False] * post
            + [True] * (capacity - length + 1)
        )
        torch.testing.assert_close(
            writer.prompt_embeds[:pre], source["writer_soft_tokens"]["pre"]
        )
        torch.testing.assert_close(
            writer.prompt_embeds[pre + num_tokens : length],
            source["writer_soft_tokens"]["post"],
        )
    cache(manager, writer)

    wrapped = [0] * pre + ids + [0] * post
    tokens = [1, *wrapped, 2, *wrapped, 3]
    reusable = ReusableInput([source], [(1, length, 0), (length + 2, length, 0)])
    wire = input_processor.process_inputs(
        "reader",
        {"type": "token", "prompt_token_ids": tokens, "reusable_input": reusable},
        SamplingParams(max_tokens=2),
        ("generate",),
    )
    reader, _ = engine.preprocess_add_request(
        decoder.decode(MsgpackEncoder().encode(wire))
    )
    assert reader.reusable_spans[0].content is reader.reusable_spans[1].content
    prepare_reusable_request(reader, manager)
    _, count, _ = manager.get_computed_blocks(reader)
    require_reusable_prefix(reader, count)
    assert count == capacity
    layout = MsgpackDecoder(PICLayout).decode(
        MsgpackEncoder().encode(reader.pic_layout)
    )
    assert len(layout.contents) == 1
    assert layout.contents[0].length == length
    assert layout.contents[0].entry_positions == {}
    assert layout.occurrences[0].segments == layout.occurrences[1].segments
    assert "groups" not in source


def test_compile_padding_is_checked_against_native_prompt_limit(input_processor):
    from vllm.exceptions import VLLMValidationError
    from vllm.inputs import tokens_input
    from vllm.v1.engine.pic_compiler import compile_inputs

    ids = [1] * 63
    source = {
        "content_id": "full",
        "input_ids": ids,
        "method_id": "identity",
        "writer": None,
    }
    prompt = tokens_input(ids)
    input_processor.process_inputs(
        "ordinary", prompt, SamplingParams(max_tokens=1), ("generate",)
    )
    _, params, _ = compile_inputs(source)
    with pytest.raises(VLLMValidationError, match="maximum model length"):
        input_processor.process_inputs("writer", prompt, params, ("generate",))
    assert len(prompt["prompt_token_ids"]) == 63


def test_writer_embeddings_and_their_placement_separate_native_pages(manager):
    from vllm.v1.engine.pic_compiler import prepare_content

    model = SimpleNamespace(
        dtype=torch.float32, hf_text_config=SimpleNamespace(num_hidden_layers=1)
    )
    source = {
        "content_id": "doc",
        "input_ids": [0, 0, 0],
        "method_id": "writer",
        "writer": None,
    }

    def compile_metadata(ids, soft):
        return prepare_content(
            {**source, "input_ids": ids, "writer_soft_tokens": soft},
            model,
            manager.kv_cache_config.kv_cache_groups,
        )

    pre = torch.ones(1, 64)
    original = compile_metadata([0, 0, 0], {"pre": pre})
    alias = compile_metadata([0, 0, 0], {"pre": pre.clone()})
    assert original["compiled_id"] == alias["compiled_id"]
    assert original["groups"] == alias["groups"]
    for soft in ({"pre": pre * 2}, {"post": pre}, {}):
        changed = compile_metadata([0, 0, 0], soft)
        assert original["compiled_id"] != changed["compiled_id"]
        assert (
            original["groups"][0]["block_hashes"]
            != changed["groups"][0]["block_hashes"]
        )
    # Zero-valued document tokens cannot impersonate Post embedding rows.
    short = compile_metadata([0, 0, 0], {"post": pre})
    long = compile_metadata([0, 0, 0, 0], {"post": pre})
    assert short["groups"][0]["block_hashes"][0] != long["groups"][0]["block_hashes"][0]


@pytest.mark.parametrize(
    "tokens,method,shared_pages",
    [
        ([1, 2, 3, 4, 9, 8, 7, 6], "writer", 1),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], "writer", 2),
        ([0, 2, 3, 4, 5, 6, 7, 8], "writer", 0),
        ([1, 2, 3, 4, 5, 6, 7, 8], "other-writer", 0),
    ],
)
def test_pic_pages_reuse_only_their_writer_local_prefix(
    manager, tokens, method, shared_pages
):
    original = native_content(manager, "original", list(range(1, 9)))
    candidate = native_content(manager, "candidate", tokens, method)
    assert original["compiled_id"] != candidate["compiled_id"]
    left = original["groups"][0]["block_hashes"]
    right = candidate["groups"][0]["block_hashes"]
    assert left[:shared_pages] == right[:shared_pages]
    assert all(a != b for a, b in zip(left[shared_pages:], right[shared_pages:]))


def test_pic_identity_ignores_alias_and_dtype_while_pages_keep_dtype_isolation(manager):
    original = native_content(manager, "original", [1, 2, 3, 4])
    candidate = native_content(manager, "alias", [1, 2, 3, 4], dtype=torch.bfloat16)
    assert original["compiled_id"] == candidate["compiled_id"]
    assert (
        original["groups"][0]["block_hashes"] != candidate["groups"][0]["block_hashes"]
    )


def test_repeated_pic_blocks_and_tail_padding_do_not_alias(manager):
    repeated = native_content(manager, "repeated", [1, 2, 3, 4] * 2)
    hashes = repeated["groups"][0]["block_hashes"]
    assert hashes[0] != hashes[1]
    short = native_content(manager, "short", [1, 2, 3, 4, 5])
    longer = native_content(manager, "longer", [1, 2, 3, 4, 5, 0])
    short_hashes = short["groups"][0]["block_hashes"]
    longer_hashes = longer["groups"][0]["block_hashes"]
    assert short_hashes[0] == longer_hashes[0]
    assert short_hashes[1] != longer_hashes[1]


def test_native_prefix_and_repeated_pic_share_storage(manager):
    ordinary = request("ordinary", list(range(5)))
    cache(manager, ordinary)
    compiled = request("compiled", [40, 41, 42, 43, 44])
    cache(manager, compiled)
    metadata = content(compiled.block_hashes[0])
    tokens = [0, 1, 2, 3, 4, 40, 41, 42, 8, 9, 40, 41, 42, 12, 13, 14]
    req = request(
        "reader", tokens, [ReusableSpan(5, 3, metadata), ReusableSpan(10, 3, metadata)]
    )
    assert not req.block_hashes
    assert req._pic_dynamic_content_hashes[0] == ordinary.block_hashes[0]
    prepare_reusable_request(req, manager)
    blocks, computed, _ = manager.get_computed_blocks(req)
    require_reusable_prefix(req, computed)
    assert computed == 8
    assert len(req.pic_layout.contents) == 1
    assert req.pic_layout.contents[0].entry_positions == {}
    assert req.pic_layout.occurrences[0].segments == [(4, 0, 4)]
    assert req.pic_layout.occurrences[1].segments == [(4, 0, 4)]
    assert list(req.all_token_ids[computed:]) == [4, 8, 9, 12, 13, 14]
    assert req.canonical_prompt_token_ids == tokens
    manager.allocate_slots(
        req,
        req.num_tokens - computed,
        num_new_computed_tokens=computed,
        new_computed_blocks=blocks,
    )
    owned = list(manager.get_blocks(req.request_id).blocks[0])
    assert all(block.ref_cnt == 1 for block in owned)
    req.append_output_token_ids([51, 52])
    assert len(req.block_hashes) == req.num_tokens // 4
    manager.free(req)
    assert all(block.ref_cnt == 0 for block in owned)


def test_native_writer_hashes_and_reader_use_the_same_pages(manager):
    from vllm.v1.engine.pic_compiler import compile_inputs, compile_token_ids

    metadata = native_content(manager, "independent", [20, 21, 22, 23, 24])
    hashes = metadata["groups"][0]["block_hashes"]
    ids, params, lora = compile_inputs(metadata)
    ids = compile_token_ids(ids, 4)
    writer = Request(
        "writer",
        ids,
        params,
        None,
        lora_request=lora,
        block_hasher=get_request_block_hasher(4, sha256),
    )
    assert ids == [20, 21, 22, 23, 24, 0, 0, 0, 0]
    assert writer.block_hashes == hashes
    assert writer.sampling_params.extra_args == {}
    assert params.extra_args["pic_compile"] is metadata
    cache(manager, writer)
    blocks, count, _ = manager.get_computed_blocks(writer)
    assert count == 8  # A cache hit does not recompute the last document page.
    writer.append_output_token_ids([1])
    assert writer.block_hashes == hashes
    reader = request(
        "reader", [1, *metadata["input_ids"], 2], [ReusableSpan(1, 5, metadata)]
    )
    prepare_reusable_request(reader, manager)
    reader_blocks, count, _ = manager.get_computed_blocks(reader)
    require_reusable_prefix(reader, count)
    assert reader_blocks.blocks == blocks.blocks

    extension = native_content(manager, "extension", [20, 21, 22, 23, 25, 26])
    ids, params, lora = compile_inputs(extension)
    ids = compile_token_ids(ids, 4)
    next_writer = Request(
        "next-writer",
        ids,
        params,
        None,
        lora_request=lora,
        block_hasher=get_request_block_hasher(4, sha256),
    )
    prefix_blocks, count, _ = manager.get_computed_blocks(next_writer)
    assert count == 4
    assert prefix_blocks.blocks[0] == blocks.blocks[0][:1]
    manager.allocate_slots(
        next_writer,
        next_writer.num_tokens - count,
        num_new_computed_tokens=count,
        new_computed_blocks=prefix_blocks,
    )
    extension_blocks = manager.get_blocks(next_writer.request_id)
    assert extension_blocks.blocks[0][0] is blocks.blocks[0][0]
    assert extension_blocks.blocks[0][1] is not blocks.blocks[0][1]
    manager.free(next_writer)

    combined = request(
        "combined",
        [1, *metadata["input_ids"], 2, *extension["input_ids"], 3],
        [ReusableSpan(1, 5, metadata), ReusableSpan(7, 6, extension)],
    )
    prepare_reusable_request(combined, manager)
    shared_blocks, count, _ = manager.get_computed_blocks(combined)
    require_reusable_prefix(combined, count)
    assert count == 12  # Three unique pages for two two-page PICs.
    assert len(shared_blocks.blocks[0]) == 3


def test_missing_pic_and_terminal_pic_cannot_execute_placeholders(manager):
    metadata = content(b"missing")
    req = request("missing", list(range(8)), [ReusableSpan(0, 4, metadata)])
    prepare_reusable_request(req, manager)
    _, computed, _ = manager.get_computed_blocks(req)
    with pytest.raises(RuntimeError, match="Required reusable KV"):
        require_reusable_prefix(req, computed)
    terminal = request("terminal", list(range(8)), [ReusableSpan(4, 4, metadata)])
    with pytest.raises(NotImplementedError, match="final observation"):
        prepare_reusable_request(terminal, manager)


def test_mixed_embeddings_follow_storage_order_without_changing_prefix_identity(
    manager,
):
    from vllm.v1.core.sched.output import NewRequestData

    embeds = torch.zeros(16, 3)
    embeds[[2, 8, 9, 12]] = torch.arange(12).reshape(4, 3).float() + 1
    mask = [index not in (2, 8, 9, 12) for index in range(16)]
    tokens = [index if is_token else 99 for index, is_token in enumerate(mask)]
    ordinary = request(
        "prefix", tokens[:5], prompt_embeds=embeds[:5], prompt_is_token_ids=mask[:5]
    )
    cache(manager, ordinary)
    compiled = request("compiled", [40, 41, 42, 43, 44])
    cache(manager, compiled)
    metadata = content(compiled.block_hashes[0])
    spans = [ReusableSpan(5, 3, metadata), ReusableSpan(10, 2, metadata)]
    req = request(
        "reader", tokens, spans, prompt_embeds=embeds, prompt_is_token_ids=mask
    )
    assert req._pic_dynamic_content_hashes[0] == ordinary.block_hashes[0]
    prepare_reusable_request(req, manager)
    data = NewRequestData.from_request(req, ([],))
    order = [0, 1, 2, 3, None, None, None, None, 4, 8, 9, 12, 13, 14, 15]
    expected = torch.stack(
        [torch.zeros(3) if index is None else embeds[index] for index in order]
    )
    torch.testing.assert_close(data.prompt_embeds, expected)
    assert data.prompt_is_token_ids == [
        True if index is None else mask[index] for index in order
    ]
    assert all(0 <= token < 99 for token in req.all_token_ids)
    req.append_output_token_ids([20, 21, 22, 23])
    fresh = request(
        "fresh", tokens, spans, prompt_embeds=embeds, prompt_is_token_ids=mask
    )
    fresh.append_output_token_ids([20, 21, 22, 23])
    assert req._pic_dynamic_content_hashes == fresh._pic_dynamic_content_hashes
    torch.testing.assert_close(req.prompt_embeds, embeds)


def test_dynamic_content_hashes_exclude_future_pic_identity(manager):
    tokens = list(range(24))
    early = ReusableSpan(4, 4, content(b"early"))
    left = request(
        "left", tokens, [early, ReusableSpan(16, 4, content(b"a", "future-a"))]
    )
    right = request(
        "right", tokens, [early, ReusableSpan(16, 4, content(b"b", "future-b"))]
    )
    assert left._pic_dynamic_content_hashes[:3] == right._pic_dynamic_content_hashes[:3]
    assert left._pic_dynamic_content_hashes[3] != right._pic_dynamic_content_hashes[3]


def test_dynamic_content_hashes_depend_on_the_pic_writer_identity(manager):
    tokens = list(range(12))
    original = native_content(manager, "original", tokens[4:8])
    alias = native_content(manager, "alias", tokens[4:8])
    changed = native_content(manager, "changed", tokens[4:8], method="other-writer")
    left, same, right = [
        request(name, tokens, [ReusableSpan(4, 4, metadata)])
        for name, metadata in (("left", original), ("same", alias), ("right", changed))
    ]
    assert left._pic_dynamic_content_hashes == same._pic_dynamic_content_hashes
    assert left._pic_dynamic_content_hashes[0] == right._pic_dynamic_content_hashes[0]
    assert left._pic_dynamic_content_hashes[1] != right._pic_dynamic_content_hashes[1]


def test_native_preemption_waits_for_store_before_replanning(manager):
    cpu = CPUOffloadingManager(32)
    spec = SimpleNamespace(
        tokens_per_block=[4],
        tokens_per_hash=4,
        blocks_per_chunk=1,
        offload_prompt_only=False,
        get_manager=lambda: cpu,
        kv_events_config=OffloadingKVEventsConfig(False, False),
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, prefix_cache_retention_interval=None
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(world_size=1, decode_context_parallel_size=1),
    )
    offload = OffloadingConnectorScheduler(spec, config, manager.kv_cache_config)
    # Assemble the actual scheduler's metadata/preemption methods without a
    # model executor. Cache managers, request queues and transfer jobs are real.
    connector = OffloadingConnector.__new__(OffloadingConnector)
    connector.connector_scheduler = offload
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.connector = connector
    scheduler.kv_cache_manager = manager
    scheduler.defer_block_free = False
    scheduler.encoder_cache_manager = EncoderCacheManager(0)
    scheduler._inflight_prefills = set()
    scheduler.log_stats = False
    scheduler.waiting = FCFSRequestQueue()
    scheduler.reset_preempted_req_ids = set()

    ordinary = request("ordinary", list(range(5)))
    cache(manager, ordinary)
    compiled = request("compiled", [40, 41, 42, 43, 44])
    cache(manager, compiled)
    pic_hash = compiled.block_hashes[0]
    req = request("reader", list(range(13)), [ReusableSpan(5, 3, content(pic_hash))])
    offload.on_new_request(req)
    blocks, computed, _, _ = scheduler._get_local_prefix_cache_hit(req)
    assert offload.get_num_new_matched_tokens(req, computed) == (0, False)
    manager.allocate_slots(
        req,
        req.num_tokens - computed,
        num_new_computed_tokens=computed,
        new_computed_blocks=blocks,
    )
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [
        NewRequestData.from_request(req, manager.get_block_ids(req.request_id))
    ]
    output.num_scheduled_tokens = {req.request_id: req.num_tokens}
    output.total_num_scheduled_tokens = req.num_tokens
    jobs = offload.build_connector_meta(output).store_jobs
    assert jobs
    state = offload._req_status[req.request_id]
    previous_layout = req.pic_layout
    previous_keys = list(state.group_states[0].offload_keys)

    req.status = RequestStatus.RUNNING
    scheduler._preempt_request(req, timestamp=0)
    pic_block = manager.block_pool.get_cached_block(pic_hash, [0])[0]
    manager.evict_blocks({pic_block.block_id})
    _, pending_count, _, _ = scheduler._get_local_prefix_cache_hit(req)
    assert req.pic_layout is previous_layout
    assert offload.get_num_new_matched_tokens(req, pending_count) == (None, False)
    flush = SchedulerOutput.make_empty()
    flush.preempted_req_ids = {req.request_id}
    assert offload.build_connector_meta(flush).jobs_to_flush == set(jobs)

    # CPU-only test supplies the worker's completion message; the native
    # connector performs complete_store and releases the pending request jobs.
    offload.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={job: 1 for job in jobs}
            )
        )
    )
    _, local, _, _ = scheduler._get_local_prefix_cache_hit(req)
    assert req.pic_layout is not previous_layout
    assert local == 8
    state.update_offload_keys()
    assert state.group_states[0].offload_keys != previous_keys
    assert state.group_states[0].next_stored_chunk_idx == 0
    external, asynchronous = offload.get_num_new_matched_tokens(req, local)
    assert (external, asynchronous) == (4, True)
    require_reusable_prefix(req, local + external)
    state.group_states[0].next_stored_chunk_idx = 3
    req.append_output_token_ids([51, 52])
    state.update_offload_keys()
    assert state.group_states[0].next_stored_chunk_idx == 3


@pytest.mark.parametrize("already_compiling", [False, True])
def test_missing_documents_enqueue_together_but_hits_and_pending_loads_do_not(
    manager,
    already_compiling,
):
    from vllm.v1.core.pic import defer_missing_content
    from vllm.v1.engine.pic_compiler import compile_inputs, compile_token_ids
    from vllm.v1.kv_offload.base import ReqContext, make_offload_key

    documents: list[dict[str, Any]] = [
        {
            "compiled_id": name,
            "method_id": "writer",
            "writer": None,
            "input_ids": [20, 21, 22],
            "logical_length": 3,
            "groups": [
                {
                    "block_hashes": [name.encode()],
                    "num_slots": 4,
                    "block_size": 4,
                    "layers": {"layer.0": {"entry_positions": None}},
                }
            ],
        }
        for name in ("missing-a", "gpu", "cpu", "pending", "missing-b")
    ]
    documents[0]["input_ids"] = [21]
    documents[0]["writer_soft_tokens"] = {
        "pre": torch.full((1, 64), 0.25),
        "post": torch.full((1, 64), 0.75),
    }
    ids, params, lora = compile_inputs(documents[1])
    ids = compile_token_ids(ids, 4)
    cache(
        manager,
        Request(
            "writer",
            ids,
            params,
            None,
            lora_request=lora,
            block_hasher=get_request_block_hasher(4, sha256),
        ),
    )
    cpu = CPUOffloadingManager(32)
    context = ReqContext("reader", require_load=True)
    cpu.on_new_request(context)
    keys = [make_offload_key(name.encode(), 0) for name in ("cpu", "pending")]
    cpu.prepare_store(keys, context)
    cpu.complete_store(keys[:1], context)
    spans = [ReusableSpan(index * 4, 3, item) for index, item in enumerate(documents)]
    spans.append(ReusableSpan(20, 3, documents[0]))
    reader = request("reader", list(range(25)), spans)
    prepare_reusable_request(reader, manager)
    queued = []
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config={"pic": {"auto_compile": True}}),
        kv_cache_manager=manager,
        pic_compiles={},
        connector=SimpleNamespace(
            connector_scheduler=SimpleNamespace(
                manager=cpu,
                _req_status={"reader": SimpleNamespace(req_context=context)},
            )
        ),
    )

    def enqueue(child):
        queued.append(child)
        scheduler.pic_compiles[child.pic_compile["compiled_id"]] = {child.request_id}

    scheduler.add_request = enqueue
    if already_compiling:
        scheduler.pic_compiles["missing-a"] = {"existing-writer"}
    expected = ["missing-b"] if already_compiling else ["missing-a", "missing-b"]
    assert defer_missing_content(scheduler, reader, 4)
    assert [child.pic_compile["compiled_id"] for child in queued] == expected
    assert all(child.pic_internal for child in queued)
    if not already_compiling:
        child = queued[0]
        assert child.prompt_token_ids == [0, 21, 0, 0, 0]
        assert child.prompt_is_token_ids == [False, True, False, True, True]
        torch.testing.assert_close(child.prompt_embeds[0], torch.full((64,), 0.25))
        torch.testing.assert_close(child.prompt_embeds[2], torch.full((64,), 0.75))
    assert defer_missing_content(scheduler, reader, 4)
    assert len(queued) == len(expected)
    # Once writes complete, normal connector loading can admit the Reader.
    more = [make_offload_key(name.encode(), 0) for name in ("missing-a", "missing-b")]
    cpu.prepare_store(more, context)
    cpu.complete_store([keys[1], *more], context)
    scheduler.pic_compiles.clear()
    assert not defer_missing_content(scheduler, reader, 4)
