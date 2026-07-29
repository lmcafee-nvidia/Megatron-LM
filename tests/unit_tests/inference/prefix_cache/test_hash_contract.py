# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Boundary, ancestry, and request-lifecycle stress for prefix-cache hashes."""

import hashlib

import msgpack
import pytest
import torch

from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    compute_block_hashes_batched,
)
from megatron.core.inference.sampling_params import SamplingParams

_CASES = (
    pytest.param(4, 0, id="block4-mutate-first"),
    pytest.param(16, 1, id="block16-mutate-second"),
    pytest.param(32, 8, id="block32-mutate-interior"),
    pytest.param(128, 22, id="block128-mutate-penultimate"),
    pytest.param(256, 23, id="block256-mutate-last"),
)


def _scalar_hashes(tokens, block_size, namespace=None):
    """Independent block-by-block SHA chain used as the test oracle."""
    parent = (
        bytes(32)
        if namespace is None
        else hashlib.sha256(f"mcore-prefix-cache:{namespace}".encode()).digest()
    )
    result = []
    complete_token_count = len(tokens) // block_size * block_size
    token_bytes = torch.tensor(tokens[:complete_token_count], dtype=torch.int64).numpy().tobytes()
    block_bytes = block_size * 8
    for offset in range(0, len(token_bytes), block_bytes):
        parent = hashlib.sha256(parent + token_bytes[offset : offset + block_bytes]).digest()
        raw = int.from_bytes(parent[:8], byteorder="little", signed=False)
        result.append(raw % (2**63 - 1) + 1)
    return result


@pytest.mark.internal
@pytest.mark.parametrize("block_size,mutation_block", _CASES)
def test_hash_chain_boundary_and_request_lifecycle_stress(block_size, mutation_block):
    """Each row checks seven boundaries and a 24-block serialized hash generation."""
    namespace = 1_000 + block_size
    chain_block_count = 24
    tokens = list(range(1, chain_block_count * block_size + block_size))

    for token_count in (
        0,
        block_size - 1,
        block_size,
        block_size + 1,
        2 * block_size - 1,
        2 * block_size,
        2 * block_size + 1,
    ):
        prompt = torch.tensor(tokens[:token_count], dtype=torch.int32)
        actual = compute_block_hashes_batched(prompt, block_size, namespace=namespace)
        assert actual == _scalar_hashes(tokens[:token_count], block_size, namespace)
        assert len(actual) == token_count // block_size

    prompt_tokens = torch.tensor(tokens[: chain_block_count * block_size], dtype=torch.int64)
    original = compute_block_hashes_batched(prompt_tokens, block_size, namespace=namespace)
    assert original == _scalar_hashes(prompt_tokens.tolist(), block_size, namespace)
    assert len(original) == chain_block_count

    mutated_tokens = prompt_tokens.clone()
    mutation_offset = mutation_block * block_size
    mutated_tokens[mutation_offset] += 31
    mutated_tokens[mutation_offset + 1] -= 1
    mutated = compute_block_hashes_batched(mutated_tokens, block_size, namespace=namespace)
    assert mutated[:mutation_block] == original[:mutation_block]
    assert all(
        changed != baseline
        for changed, baseline in zip(mutated[mutation_block:], original[mutation_block:])
    )

    request = DynamicInferenceRequest(
        request_id=block_size,
        prompt_tokens=prompt_tokens,
        sampling_params=SamplingParams(
            num_tokens_to_generate=block_size + 2, termination_id=-1, return_prompt_tokens=True
        ),
        block_size_tokens=block_size,
        enable_prefix_caching=True,
        prefix_cache_namespace=namespace,
    )
    assert request.precomputed_block_hashes == original

    serialized_request = msgpack.unpackb(
        msgpack.packb(request.serialize(), use_bin_type=True), raw=False
    )
    deserialized = DynamicInferenceRequest.deserialize(serialized_request)
    assert deserialized.precomputed_block_hashes == original
    assert deserialized.block_size_tokens == block_size
    assert deserialized.enable_prefix_caching is True
    assert deserialized.prefix_cache_namespace is None

    request.generated_tokens = list(range(block_size))
    record = DynamicInferenceRequestRecord.from_request(request)
    record.checkpoint()
    checkpointed = record[-1]
    expected_checkpoint_hashes = compute_block_hashes_batched(
        checkpointed.prompt_tokens, block_size, namespace=namespace
    )
    assert checkpointed.precomputed_block_hashes == expected_checkpoint_hashes
    assert len(expected_checkpoint_hashes) == chain_block_count + 1

    serialized_record = msgpack.unpackb(
        msgpack.packb(record.serialize(), use_bin_type=True), raw=False
    )
    restored_record = DynamicInferenceRequestRecord.deserialize(serialized_record)
    assert restored_record[-1].precomputed_block_hashes == expected_checkpoint_hashes
    assert restored_record[-1].prefix_cache_namespace is None
    restored_record[-1].set_prefix_cache_namespace(namespace + 1)
    assert restored_record[-1].precomputed_block_hashes != expected_checkpoint_hashes
