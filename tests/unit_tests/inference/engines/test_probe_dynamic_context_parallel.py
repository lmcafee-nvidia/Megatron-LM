# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validation-only probe: what dynamic inference does with context_parallel_size=2 on main.

Runs the same small GPT engine workload with CP=1 and then with a real CP=2 group and
model configuration, and prints PROBE lines comparing weights and generated tokens.
"""

import traceback

import pytest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.engines import test_dynamic_engine as harness
from tests.unit_tests.test_utilities import Utils


class TestProbeDynamicContextParallel(harness.DynamicInferenceEngineTestBase):

    @classmethod
    def teardown_class(cls):
        harness.delete_cuda_graphs()
        Utils.destroy_model_parallel()

    def _generate(self, cp_size, monkeypatch):
        Utils.initialize_model_parallel(context_parallel_size=cp_size)
        try:
            monkeypatch.setattr(
                harness,
                "TransformerConfig",
                lambda **kwargs: TransformerConfig(**{**kwargs, "context_parallel_size": cp_size}),
            )
            env = self._run_test(
                num_requests=4,
                num_tokens_to_generate=8,
                num_gap_steps=0,
                transformer_impl="transformer_engine",
            )
            model = env.engine.controller.inference_wrapped_model.model
            checksum = sum(p.double().sum().item() for p in model.parameters())
            return [list(r.generated_tokens) for r in env.requests], checksum
        finally:
            harness.delete_cuda_graphs()
            Utils.destroy_model_parallel()

    def test_probe_context_parallel_generation(self, monkeypatch):
        if Utils.world_size != 2:
            pytest.skip("probe runs on exactly 2 GPUs")
        reference, reference_checksum = self._generate(1, monkeypatch)
        try:
            tokens, checksum = self._generate(2, monkeypatch)
        except Exception:  # noqa: BLE001
            print(f"PROBE cp2-error rank={Utils.rank}\n{traceback.format_exc()}")
            return
        mismatched = [i for i, (a, b) in enumerate(zip(tokens, reference)) if a != b]
        print(
            f"PROBE rank={Utils.rank} same_weights={checksum == reference_checksum} "
            f"mismatched_requests={mismatched} of {len(reference)} "
            f"cp1={reference} cp2={tokens}"
        )
