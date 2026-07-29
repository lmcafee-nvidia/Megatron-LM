# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Stress prefix caching with real CUDA-graph execution.

Each case runs the same three-cycle reuse workload with cache-off/graphs-on,
cache-on/graphs-off, and cache-on/graphs-on. This separately verifies cache
correctness and graph correctness while proving that cache hits really occur.
"""

import random
import types

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.config import (
    InferenceConfig,
    MambaInferenceStateConfig,
    PrefixCachingEvictionPolicy,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars

BLOCK_SIZE = 256
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 2048
NUM_TOKENS_TO_GENERATE = 8


def set_rounder(value):
    DynamicInferenceContext.ROUNDER = value
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


@pytest.mark.internal
@pytest.mark.skipif(not is_fa_min_version("2.7.3"), reason="need flash attn")
class TestPrefixCachingCudaGraphs:
    """Verify prefix caching + CUDA graph interaction across model types and batch structures."""

    def setup_method(self, method):
        Utils.initialize_model_parallel()
        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _create_model(self, model_type, num_cuda_graphs=None):
        """Create a model with optional CUDA graph support.

        Returns (model, mamba_config_or_none).
        """
        cuda_graph_impl = "local" if num_cuda_graphs else "none"

        if model_type == "transformer":
            config = TransformerConfig(
                params_dtype=torch.bfloat16,
                num_layers=4,
                hidden_size=32,
                num_attention_heads=4,
                use_cpu_initialization=True,
                cuda_graph_impl=cuda_graph_impl,
                inference_rng_tracker=True,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                pipeline_dtype=torch.bfloat16,
                add_bias_linear=True,
            )
            model = GPTModel(
                config=config,
                transformer_layer_spec=get_gpt_layer_local_spec(),
                vocab_size=VOCAB_SIZE,
                max_sequence_length=MAX_SEQ_LEN,
                parallel_output=True,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
            ).cuda()
            mamba_config = None
        else:  # hybrid
            config = TransformerConfig(
                params_dtype=torch.bfloat16,
                num_layers=3,
                hidden_size=256,
                mamba_num_heads=16,
                num_attention_heads=16,
                use_cpu_initialization=True,
                cuda_graph_impl=cuda_graph_impl,
                inference_rng_tracker=True,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                pipeline_dtype=torch.bfloat16,
                add_bias_linear=True,
                is_hybrid_model=True,
            )
            model = HybridModel(
                config=config,
                hybrid_stack_spec=hybrid_stack_spec,
                vocab_size=VOCAB_SIZE,
                max_sequence_length=MAX_SEQ_LEN,
                parallel_output=True,
                hybrid_layer_pattern="M*-",
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
            ).cuda()
            mamba_config = MambaInferenceStateConfig.from_model(model)

        for param in model.parameters():
            param.data = param.data.to(config.params_dtype)
        model.eval()
        return model, mamba_config

    def _reset_cuda_graph_state(self, model):
        """Reset all CUDA graph global and per-module state."""
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None
        for module in model.modules():
            if isinstance(module, CudaGraphManager):
                module.cudagraph_runners.clear()
                module.custom_cudagraphs_lookup_table.clear()

    def _build_engine(self, model, mamba_config, enable_prefix_caching, num_cuda_graphs):
        """Build an engine with independently controlled prefix caching and CUDA graphs."""
        set_rounder(4)
        inference_config_kwargs = dict(
            max_sequence_length=MAX_SEQ_LEN,
            buffer_size_gb=0.5,
            block_size_tokens=BLOCK_SIZE,
            materialize_only_last_token_logits=False,
            enable_prefix_caching=enable_prefix_caching,
            unified_memory_level=0,
            num_cuda_graphs=num_cuda_graphs,
            use_cuda_graphs_for_non_decode_steps=True,
        )
        if mamba_config is not None:
            # max_requests is not capped here, so it auto-derives from the KV buffer
            # size. The Mamba cache budget must cover the per-step extraction scratch
            # (which scales with max_requests) on top of the durable cache.
            # max_requests is left uncapped to preserve this test's CUDA-graph buckets.
            inference_config_kwargs["mamba_inference_state_config"] = mamba_config
        if enable_prefix_caching:
            inference_config_kwargs["prefix_caching_eviction_policy"] = (
                PrefixCachingEvictionPolicy.LRU
            )
            if mamba_config is not None:
                inference_config_kwargs["prefix_caching_mamba_gb"] = 2.0
        context = DynamicInferenceContext(
            model_config=model.config, inference_config=InferenceConfig(**inference_config_kwargs)
        )
        wrapper = GPTInferenceWrapper(model, context)
        wrapper.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )
        controller = TextGenerationController(
            inference_wrapped_model=wrapper,
            tokenizer=types.SimpleNamespace(
                vocab_size=VOCAB_SIZE, detokenize=lambda tokens: "tokenized_prompt"
            ),
        )
        self._reset_cuda_graph_state(model)
        return DynamicInferenceEngine(controller, context)

    def _create_prompts(self):
        """Build one seed and six requests that reuse its first block."""
        device = torch.cuda.current_device()
        base = torch.arange(0, 256, dtype=torch.int64, device=device)
        extra = torch.arange(256, 300, dtype=torch.int64, device=device)
        prompts = [torch.cat([base, extra])]
        for cycle in range(6):
            unique_length = 144 + 50 * cycle
            unique_start = 1000 + 500 * cycle
            unique = torch.arange(
                unique_start, unique_start + unique_length, dtype=torch.int64, device=device
            )
            prompts.append(torch.cat([base, unique]))
        return prompts

    def _make_request(self, req_id, prompt, enable_prefix_caching):
        return DynamicInferenceRequest(
            request_id=req_id,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(
                num_tokens_to_generate=NUM_TOKENS_TO_GENERATE, termination_id=-1, top_k=1
            ),
            block_size_tokens=BLOCK_SIZE if enable_prefix_caching else None,
            enable_prefix_caching=enable_prefix_caching,
        )

    def _run_scenario(self, engine, schedule, prompts, enable_prefix_caching):
        """Run three seed-reuse cycles with staged or mixed batching.

        Returns outputs, executed requests, step dimensions, and cache metrics.
        """
        ctx = engine.context
        finished = {}
        requests = []
        step_log = []

        def _step_and_log():
            result = engine.step_modern()
            step_log.append(
                (
                    ctx.batch_dimensions.prefill_req_count,
                    ctx.batch_dimensions.decode_req_count,
                    ctx.using_cuda_graph_this_step(),
                )
            )
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

        def _add_request(prompt):
            request = self._make_request(
                len(requests), prompt, enable_prefix_caching=enable_prefix_caching
            )
            requests.append(request)
            engine._add_request(request)

        def _drain():
            while engine.has_unfinished_requests():
                _step_and_log()

        _add_request(prompts[0])
        _drain()

        if schedule == "staged":
            for prompt in prompts[1:4]:
                _add_request(prompt)
                _drain()
        else:
            for cycle in range(3):
                _add_request(prompts[1 + 2 * cycle])
                while engine.has_unfinished_requests():
                    _step_and_log()
                    prefill_count, decode_count, _ = step_log[-1]
                    if prefill_count == 0 and decode_count > 0:
                        break
                _add_request(prompts[2 + 2 * cycle])
                _drain()

        return finished, requests, step_log, engine.get_prefix_cache_metrics()

    @pytest.mark.parametrize("model_type", ["transformer", "hybrid"])
    @pytest.mark.parametrize("schedule", ["staged", "mixed"])
    @torch.inference_mode()
    def test_prefix_caching_cuda_graphs(self, model_type, schedule):
        """Verify three reuse cycles against cache-off and graph-off references."""
        if model_type == "hybrid":
            sequence_packing_available, reason = _check_mamba_sequence_packing_support()
            if not sequence_packing_available:
                pytest.skip(reason)

        # Create model with CUDA graph support (cuda_graph_impl="local").
        model, mamba_config = self._create_model(model_type, num_cuda_graphs=2)
        prompts = self._create_prompts()

        off_graph_engine = self._build_engine(
            model, mamba_config, enable_prefix_caching=False, num_cuda_graphs=2
        )
        off_outputs, _, _, off_metrics = self._run_scenario(
            off_graph_engine, schedule, prompts, enable_prefix_caching=False
        )

        on_eager_engine = self._build_engine(
            model, mamba_config, enable_prefix_caching=True, num_cuda_graphs=None
        )
        on_eager_outputs, on_eager_requests, _, on_eager_metrics = self._run_scenario(
            on_eager_engine, schedule, prompts, enable_prefix_caching=True
        )

        on_graph_engine = self._build_engine(
            model, mamba_config, enable_prefix_caching=True, num_cuda_graphs=2
        )
        on_graph_outputs, on_graph_requests, step_log, on_graph_metrics = self._run_scenario(
            on_graph_engine, schedule, prompts, enable_prefix_caching=True
        )

        assert off_outputs == on_eager_outputs == on_graph_outputs
        assert off_metrics["hits"] == off_metrics["blocks_matched"] == 0

        expected_hits = 3 if schedule == "staged" else 6
        for metrics in (on_eager_metrics, on_graph_metrics):
            assert metrics["hits"] >= expected_hits
            assert metrics["blocks_matched"] >= expected_hits
            assert metrics["prefill_tokens_skipped"] >= expected_hits * BLOCK_SIZE
            if model_type == "hybrid":
                assert metrics["mamba_restore_hits"] >= expected_hits

        for requests in (on_eager_requests, on_graph_requests):
            for request in requests[1:]:
                assert request.num_cached_tokens >= BLOCK_SIZE
                if model_type == "hybrid":
                    assert request._mamba_num_matched_blocks >= 1

        assert any(p > 0 and d == 0 and cg for p, d, cg in step_log), step_log
        decode_only = [(p, d, cg) for p, d, cg in step_log if p == 0 and d > 0]
        assert decode_only and all(cg for _, _, cg in decode_only), step_log
        if schedule == "mixed":
            assert any(
                p > 0 and d > 0 and cg for p, d, cg in step_log
            ), f"no mixed CG step found in {step_log}"


@pytest.mark.internal
@pytest.mark.skipif(not is_fa_min_version("2.7.3"), reason="need flash attn")
class TestHybridChunkedPrefillIntermediateState:
    """Verify hybrid chunked prefill with concurrent Mamba state extraction and restoration.

    Scenario: one request is mid-chunk (Mamba intermediate state being extracted during
    forward pass) while another request has its Mamba state restored from the prefix cache.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()
        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    @classmethod
    def teardown_class(cls):
        set_rounder(64)
        Utils.destroy_model_parallel()

    def _create_hybrid_model(self, num_cuda_graphs=None):
        """Create a hybrid (Mamba + attention) model."""
        cuda_graph_impl = "local" if num_cuda_graphs else "none"
        config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=3,
            hidden_size=256,
            mamba_num_heads=16,
            num_attention_heads=16,
            use_cpu_initialization=True,
            cuda_graph_impl=cuda_graph_impl,
            inference_rng_tracker=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            add_bias_linear=True,
            is_hybrid_model=True,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=VOCAB_SIZE,
            max_sequence_length=MAX_SEQ_LEN,
            parallel_output=True,
            hybrid_layer_pattern="M*-",
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        ).cuda()
        for param in model.parameters():
            param.data = param.data.to(config.params_dtype)
        model.eval()
        return model

    def _reset_cuda_graph_state(self, model):
        """Reset all CUDA graph global and per-module state."""
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None
        for module in model.modules():
            if isinstance(module, CudaGraphManager):
                module.cudagraph_runners.clear()
                module.custom_cudagraphs_lookup_table.clear()

    def _build_engine(
        self,
        model,
        mamba_config,
        enable_prefix_caching,
        enable_chunked_prefill,
        max_tokens=None,
        num_cuda_graphs=None,
        max_requests=128,
        cuda_graph_max_tokens=None,
    ):
        """Build an engine with the given prefix caching / chunked prefill config."""
        set_rounder(4)
        inference_config_kwargs = dict(
            max_sequence_length=MAX_SEQ_LEN,
            buffer_size_gb=0.5,
            block_size_tokens=BLOCK_SIZE,
            mamba_inference_state_config=mamba_config,
            materialize_only_last_token_logits=False,
            unified_memory_level=0,
            num_cuda_graphs=num_cuda_graphs,
            use_cuda_graphs_for_non_decode_steps=True,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=enable_chunked_prefill,
            max_requests=max_requests,
        )
        if enable_prefix_caching:
            # The Mamba cache budget must cover both the durable cache and the
            # per-step extraction scratch (which scales with max_requests), so it
            # needs enough headroom to fit the scratch and still leave durable slots.
            inference_config_kwargs.update(
                prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
                prefix_caching_mamba_gb=0.2,
            )
        if max_tokens is not None:
            inference_config_kwargs["max_tokens"] = max_tokens
        if cuda_graph_max_tokens is not None:
            inference_config_kwargs["cuda_graph_max_tokens"] = cuda_graph_max_tokens
        context = DynamicInferenceContext(
            model_config=model.config, inference_config=InferenceConfig(**inference_config_kwargs)
        )
        wrapper = GPTInferenceWrapper(model, context)
        wrapper.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )
        controller = TextGenerationController(
            inference_wrapped_model=wrapper,
            tokenizer=types.SimpleNamespace(
                vocab_size=VOCAB_SIZE, detokenize=lambda tokens: "tokenized_prompt"
            ),
        )
        self._reset_cuda_graph_state(model)
        return DynamicInferenceEngine(controller, context)

    def _make_request(self, req_id, prompt, enable_pc):
        return DynamicInferenceRequest(
            request_id=req_id,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(
                num_tokens_to_generate=NUM_TOKENS_TO_GENERATE, termination_id=-1, top_k=1
            ),
            block_size_tokens=BLOCK_SIZE if enable_pc else None,
            enable_prefix_caching=enable_pc,
        )

    @torch.inference_mode()
    def test_hybrid_chunked_prefill_intermediate_state(self):
        """Stress concurrent Mamba extraction and restoration for three cycles.

        Each cycle co-schedules a long request whose unique suffix is chunked with
        a short request that restores the seed state. Cache-off and cache-on use
        the same chunking configuration and external admission order.
        """
        sequence_packing_available, reason = _check_mamba_sequence_packing_support()
        if not sequence_packing_available:
            pytest.skip(reason)

        clear_nvte_env_vars()  # conftest's set_env fixture re-sets these per test

        model = self._create_hybrid_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)

        device = torch.cuda.current_device()
        base = torch.arange(0, BLOCK_SIZE, dtype=torch.int64, device=device)
        seed_prompt = torch.arange(0, 300, dtype=torch.int64, device=device)
        long_prompts = [
            torch.cat(
                [
                    base,
                    torch.arange(
                        5000 + cycle * 600, 5544 + cycle * 600, dtype=torch.int64, device=device
                    ),
                ]
            )
            for cycle in range(3)
        ]

        def run(enable_prefix_caching):
            engine = self._build_engine(
                model,
                mamba_config,
                enable_prefix_caching=enable_prefix_caching,
                enable_chunked_prefill=True,
                max_tokens=400,
            )
            outputs = {}
            reuse_requests = []
            concurrent_prefills = []

            def drain():
                max_prefill_requests = 0
                while engine.has_unfinished_requests():
                    result = engine.step_modern()
                    max_prefill_requests = max(
                        max_prefill_requests, engine.context.batch_dimensions.prefill_req_count
                    )
                    for record in result["finished_request_records"]:
                        merged = record.merge()
                        outputs[merged.request_id] = list(merged.generated_tokens)
                return max_prefill_requests

            seed = self._make_request(0, seed_prompt, enable_pc=enable_prefix_caching)
            engine._add_request(seed)
            drain()

            for cycle, long_prompt in enumerate(long_prompts):
                long_request = self._make_request(
                    1 + 2 * cycle, long_prompt, enable_pc=enable_prefix_caching
                )
                short_request = self._make_request(
                    2 + 2 * cycle, seed_prompt, enable_pc=enable_prefix_caching
                )
                reuse_requests.extend((long_request, short_request))
                engine._add_request(long_request)
                engine._add_request(short_request)
                concurrent_prefills.append(drain())

            return (
                outputs,
                reuse_requests,
                concurrent_prefills,
                engine.get_prefix_cache_metrics(),
                engine.context,
            )

        off_outputs, _, _, off_metrics, off_context = run(False)
        on_outputs, on_requests, concurrent_prefills, on_metrics, on_context = run(True)

        assert off_outputs == on_outputs
        assert off_metrics["hits"] == off_metrics["blocks_matched"] == 0
        assert on_metrics["hits"] >= 6
        assert on_metrics["mamba_restore_hits"] >= 6
        assert on_metrics["prefill_tokens_skipped"] >= 6 * BLOCK_SIZE
        assert on_context.lifetime_prefill_token_count < off_context.lifetime_prefill_token_count
        assert all(prefill_count >= 2 for prefill_count in concurrent_prefills)

        for request in on_requests:
            assert request.num_cached_tokens >= BLOCK_SIZE
            assert request._mamba_num_matched_blocks >= 1
        for request in on_requests[::2]:
            assert any(
                block_hash in on_context.mamba_slot_allocator.hash_to_block_id
                for block_hash in request.precomputed_block_hashes[1:]
            )

    @torch.inference_mode()
    def test_prefill_shorter_than_conv_window(self):
        """Restore a prefix before graphing a suffix shorter than the Mamba conv window.

        Conv-state extraction gathers d_conv positions per slot, and unused slots use
        abs_position == d_conv (gather indices up to d_conv-1). The CUDA-graph bucket
        list always includes a size-1 (tp_size) graph. Each reuse request restores one
        full block, leaving fewer than d_conv fresh prompt tokens in the captured graph.
        """
        sequence_packing_available, reason = _check_mamba_sequence_packing_support()
        if not sequence_packing_available:
            pytest.skip(reason)

        clear_nvte_env_vars()  # conftest's set_env fixture re-sets these per test

        model = self._create_hybrid_model(num_cuda_graphs=2)
        mamba_config = MambaInferenceStateConfig.from_model(model)
        device = torch.cuda.current_device()

        d_conv = mamba_config.conv_states_shape[-1]
        if d_conv < 3:
            pytest.skip(f"d_conv={d_conv} too small to exercise a sub-window prefill")

        base = torch.arange(0, BLOCK_SIZE, dtype=torch.int64, device=device)
        fresh_suffix_tokens = 2
        reuse_prompts = [
            torch.cat(
                [
                    base,
                    torch.arange(
                        5000 + cycle * d_conv,
                        5000 + cycle * d_conv + fresh_suffix_tokens,
                        dtype=torch.int64,
                        device=device,
                    ),
                ]
            )
            for cycle in range(3)
        ]

        def run(enable_prefix_caching):
            engine = self._build_engine(
                model,
                mamba_config,
                enable_prefix_caching=enable_prefix_caching,
                enable_chunked_prefill=True,
                num_cuda_graphs=2,
                max_requests=2,
                cuda_graph_max_tokens=fresh_suffix_tokens,
            )
            outputs = {}
            requests = []
            step_log = []

            def drain():
                while engine.has_unfinished_requests():
                    result = engine.step_modern()
                    step_log.append(
                        (
                            engine.context.batch_dimensions.prefill_req_count,
                            engine.context.using_cuda_graph_this_step(),
                        )
                    )
                    for record in result["finished_request_records"]:
                        merged = record.merge()
                        outputs[merged.request_id] = list(merged.generated_tokens)

            seed = self._make_request(0, base, enable_pc=enable_prefix_caching)
            engine._add_request(seed)
            drain()
            for req_id, prompt in enumerate(reuse_prompts, start=1):
                request = self._make_request(req_id, prompt, enable_pc=enable_prefix_caching)
                requests.append(request)
                engine._add_request(request)
                drain()

            return outputs, requests, step_log, engine.get_prefix_cache_metrics()

        off_outputs, _, _, off_metrics = run(False)
        on_outputs, on_requests, on_step_log, on_metrics = run(True)

        assert off_outputs == on_outputs
        assert off_metrics["hits"] == off_metrics["blocks_matched"] == 0
        assert on_metrics["hits"] >= 3
        assert on_metrics["blocks_matched"] >= 3
        assert on_metrics["mamba_restore_hits"] >= 3
        assert on_metrics["prefill_tokens_skipped"] >= 3 * BLOCK_SIZE
        assert any(prefill_count > 0 and used_graph for prefill_count, used_graph in on_step_log)
        for request in on_requests:
            assert request.num_cached_tokens == BLOCK_SIZE
            assert request._mamba_num_matched_blocks == 1
