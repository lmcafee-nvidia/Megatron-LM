# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Managed KV allocation must survive real transfer and model continuation."""

from dataclasses import replace

import pytest

from megatron.core.inference import unified_memory
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from tests.unit_tests.inference.engines import test_disagg_pairwise as core

transport_world = core.transport_world


@pytest.mark.parametrize("backend", ["nccl", "nixl"])
def test_managed_memory_handoff(monkeypatch, transport_world, backend):
    initialized, initialize = [], DynamicInferenceContext.__init__

    def managed_context(context, *args, **kwargs):
        kwargs["inference_config"] = replace(kwargs["inference_config"], unified_memory_level=1)
        initialize(context, *args, **kwargs)
        pool, buffer = getattr(context, "unified_memory_mempool", None), context.memory_buffer
        assert context.unified_memory_level == 1 and pool.allocator is unified_memory._alloc
        start, size = buffer.data_ptr(), buffer.numel() * buffer.element_size()
        assert any(
            0 <= start - segment["address"] <= segment["total_size"] - size
            for segment in pool.snapshot()
        ), "KV storage is not backed by the actual cudaMallocManaged pool"
        initialized.append(start)

    monkeypatch.setattr(DynamicInferenceContext, "__init__", managed_context)
    core.test_disagg_real_engine_parity(
        transport_world=transport_world,
        backend=backend,
        length=33,
        changes={"hidden_size": 256, "flash_attention_version": 4},
        count=7,
    )
    assert len(initialized) == 2, "Both reference and stressed engine must own managed KV"
