# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bitwise cache-hit parity, including chunked prefill, graph replay and refit."""

import gc
import math
from unittest import mock

import msgpack
import pytest
import torch

from megatron.core.inference.config import (
    InferenceConfig,
    KVCacheManagementMode,
    MambaInferenceStateConfig,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import dynamic_engine
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.headers import Headers
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.resharding.refit import (
    clear_all_caches,
    prepare_swap_model_weights,
    swap_model_weights,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import HAVE_FA3, HAVE_FA4
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    set_batch_invariant_mode,
    te_supports_batch_invariant_attention,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.engines.ssm_test_helpers import (
    skip_if_sequence_packing_not_available,
)
from tests.unit_tests.test_utilities import Utils


class DummyTokenizer:
    """Minimal tokenizer for the self-contained inference lifecycle fixture."""

    def __init__(self, vocab_size: int, bos: int | None = None, eod: int = 0, pad: int = 0):
        self.vocab_size = vocab_size
        self.bos = bos
        self.eod = eod
        self.pad = pad

    def tokenize(self, prompt):
        if isinstance(prompt, str):
            return [int(token) % self.vocab_size for token in prompt.strip().split()]
        return list(prompt)

    def detokenize(self, tokens, skip_special_tokens: bool = False):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if skip_special_tokens:
            tokens = [token for token in tokens if token != self.eod]
        return " ".join(str(token) for token in tokens)

    def offsets(self, tokens, text):
        del text
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        offsets, cursor = [], 0
        for token in tokens:
            offsets.append(cursor)
            cursor += len(str(token)) + 1
        return offsets


@pytest.mark.internal
@pytest.mark.skipif(
    not te_supports_batch_invariant_attention() or not (HAVE_FA3 or HAVE_FA4),
    reason="Requires batch-invariant TE and FA3/4",
)
class TestBatchInvariantMambaPrefixCaching:
    def setup_method(self):
        skip_if_sequence_packing_not_available()
        Utils.initialize_model_parallel(1, 1)
        self._previous_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)

    def teardown_method(self):
        delete_cuda_graphs()
        Utils.destroy_model_parallel()
        torch.use_deterministic_algorithms(self._previous_deterministic)

    @staticmethod
    def _run(
        *,
        prefix,
        graphs,
        chunked,
        kv_cache_management_mode=KVCacheManagementMode.PERSIST,
        sampling_backend="torch",
        exercise_real_refit=True,
    ):
        model_parallel_cuda_manual_seed(321, inference_rng_tracker=True, force_reset_rng=True)
        torch.manual_seed(321)
        cfg = TransformerConfig(
            num_layers=3,
            hidden_size=128,
            num_attention_heads=4,
            is_hybrid_model=True,
            mamba_num_heads=8,
            mamba_head_dim=32,
            mamba_num_groups=1,
            mamba_state_dim=16,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            batch_invariant_mode=True,
            batch_invariant_backend="te_native",
            flash_attention_version=4 if HAVE_FA4 else 3,
            normalization="RMSNorm",
            params_dtype=torch.bfloat16,
            bf16=True,
            attention_backend=AttnBackend.flash,
            cuda_graph_impl="local" if graphs else "none",
            inference_rng_tracker=True,
        )

        def build_model(seed):
            model_parallel_cuda_manual_seed(seed, inference_rng_tracker=True, force_reset_rng=True)
            torch.manual_seed(seed)
            return (
                HybridModel(
                    config=cfg,
                    hybrid_stack_spec=hybrid_stack_spec,
                    hybrid_layer_pattern="M*-",
                    vocab_size=128,
                    max_sequence_length=1024,
                )
                .cuda()
                .eval()
            )

        target_core = build_model(321)
        source_core = build_model(654) if exercise_real_refit else None
        mamba_config = MambaInferenceStateConfig.from_model(target_core)
        assert mamba_config.ssm_states_dtype == torch.float32
        model = Float16Module(cfg, target_core).cuda().eval()
        target_parameters = dict(model.named_parameters())
        source_model = None
        source_parameters = {}
        target_storage = {}
        if exercise_real_refit:
            source_model = Float16Module(cfg, source_core).cuda().eval()
            source_parameters = dict(source_model.named_parameters())
            assert source_parameters.keys() == target_parameters.keys()
            assert any(
                not torch.equal(source_parameters[name], target_parameters[name])
                for name in source_parameters
            )
            target_storage = {
                name: parameter.data_ptr() for name, parameter in target_parameters.items()
            }
            prepare_swap_model_weights([source_model], [model])
        context = DynamicInferenceContext(
            cfg,
            InferenceConfig(
                mamba_inference_state_config=mamba_config,
                prefix_caching_mamba_gb=0.125,
                max_sequence_length=1024,
                buffer_size_gb=0.125,
                block_size_tokens=256,
                max_requests=4,
                max_tokens=512 if chunked else 2048,
                num_cuda_graphs=4 if graphs else None,
                cuda_graph_max_tokens=256,
                materialize_only_last_token_logits=True,
                use_cuda_graphs_for_non_decode_steps=True,
                enable_chunked_prefill=chunked,
                enable_prefix_caching=prefix,
                kv_cache_management_mode=kv_cache_management_mode,
                # PERSIST naturally retains graph addresses. OFFLOAD and
                # RECOMPUTE deliberately exercise deallocation/reallocation
                # without requiring the optional torch-memory-saver runtime.
                static_kv_memory_pointers=False,
                use_flashinfer_fused_rope=False,
                unified_memory_level=0,
                sampling_backend=sampling_backend,
            ),
        )
        engine = DynamicInferenceEngine(
            TextGenerationController(
                GPTInferenceWrapper(model, context), DummyTokenizer(vocab_size=128, eod=127)
            ),
            context,
        )
        flashinfer_sampling_calls = 0
        if sampling_backend == "flashinfer":
            sample_kernel = engine.controller._sampling.sample_kernel

            def observe_flashinfer_sampling(*args, **kwargs):
                nonlocal flashinfer_sampling_calls
                sampled = sample_kernel(*args, **kwargs)
                flashinfer_sampling_calls += 1
                return sampled

            engine.controller._sampling.sample_kernel = observe_flashinfer_sampling
        graph_recaptures = 0
        if graphs and kv_cache_management_mode != KVCacheManagementMode.PERSIST:
            create_cuda_graphs = engine.create_cuda_graphs

            def observe_graph_recapture():
                nonlocal graph_recaptures
                graph_recaptures += 1
                return create_cuda_graphs()

            engine.create_cuda_graphs = observe_graph_recapture
        prompts = [[i % 120 + 1 for i in range(n)] for n in (600, 650, 512, 513, 768, 769)]
        outputs = {}
        graph_steps = 0
        post_refit_graph_steps = 0
        post_refit_chunked_steps = 0
        post_refit_sampling_start = None
        torch_top_k = int(exercise_real_refit and sampling_backend == "torch")
        for wave, order in enumerate(([0, 1], [1, 0], [2], [3], [4], [5], [0])):
            if wave == 6:
                if exercise_real_refit:
                    # First create resident KV under the old weights. PERSIST/OFFLOAD
                    # intentionally retain this mixed-generation request, while
                    # RECOMPUTE re-admits it from its checkpoint on resume.
                    engine.add_request(
                        5999,
                        [125] * 577,
                        SamplingParams(
                            seed=99 if sampling_backend == "torch" else None,
                            num_tokens_to_generate=8,
                            temperature=1.0,
                            # Torch top-k=1 is exact greedy sampling, so token identity
                            # remains a cache/refit oracle rather than a stochastic-
                            # trajectory assertion. FlashInfer retains its support-
                            # invariant stochastic owner below.
                            top_k=torch_top_k,
                            top_p=1.0,
                            return_log_probs=True,
                            skip_prompt_log_probs=True,
                            termination_id=-1,
                        ),
                    )
                    # The first async step primes old-weight logits. The second
                    # consumes that result into resident state and launches the
                    # next old-weight forward that must be invalidated by refit.
                    for _ in range(2):
                        pre_refit_result = engine.step_modern()
                        assert not pre_refit_result["finished_requests"]
                    assert engine.controller._async_sched_forward.is_valid
                    resident = engine.get_request(5999)
                    assert resident.finished_chunk_token_count > 0 or resident.generated_tokens
                    resident_salt = resident.block_hash_salt

                    # This request is admitted under the old weight epoch but has
                    # no KV yet. It must execute wholly under the new generation.
                    engine.add_request(
                        6000,
                        [126] * 577,
                        SamplingParams(
                            seed=100 if sampling_backend == "torch" else None,
                            num_tokens_to_generate=8,
                            temperature=1.0,
                            top_k=torch_top_k,
                            top_p=1.0,
                            return_log_probs=True,
                            skip_prompt_log_probs=True,
                            termination_id=-1,
                        ),
                    )
                    queued = engine.get_request(6000)
                    assert queued.finished_chunk_token_count == 0 and not queued.generated_tokens
                    old_salt = queued.block_hash_salt
                    # Old cached states must not be reused after a weight update.
                    engine.suspend()
                    pending_survived_suspend = engine.controller._async_sched_forward.is_valid
                    swap_model_weights([source_model], [model], refit_method="nccl")
                    assert all(
                        torch.equal(source_parameters[name], target_parameters[name])
                        for name in source_parameters
                    )
                    assert {
                        name: parameter.data_ptr() for name, parameter in target_parameters.items()
                    } == target_storage
                    assert pending_survived_suspend is (
                        kv_cache_management_mode != KVCacheManagementMode.RECOMPUTE
                    )

                    # Drive the same epoch message used by the RL coordinator.  The
                    # pending logits above came from the old weights and must not be
                    # sampled after the real transfer. Production sends this while
                    # suspended and only then resumes the inference engine.
                    engine.is_mp_coordinator = True
                    engine.socket_for_receiving_requests = mock.Mock()
                    engine.model_parallel_publisher_socket = mock.Mock()
                    engine.socket_for_receiving_requests.recv_multipart.side_effect = [
                        [msgpack.packb([Headers.SET_GENERATION_EPOCH.value, 1])],
                        dynamic_engine.zmq.Again,
                    ]
                    assert engine.schedule_requests() == 1
                    assert not engine.controller._async_sched_forward.is_valid
                    assert engine._generation_epoch == 1
                    assert all(
                        engine.get_request(request_id).policy_epoch[-1][1] == 1
                        for request_id in (5999, 6000)
                    )
                    engine.resume()
                    assert not engine.controller._async_sched_forward.is_valid
                    assert queued.block_hash_salt != old_salt
                    assert engine.get_request(5999).block_hash_salt == resident_salt
                else:
                    # Preserve the legacy cache-equivalence owner unchanged: it
                    # exercises deterministic request-local sampling around the
                    # original in-place weight mutation without adding a second
                    # model whose CUDA initialization perturbs that strict test.
                    engine.suspend()
                    for parameter in model.parameters():
                        parameter.mul_(1.03)
                    engine.resume()
                skipped_before_refit_request = engine._prefill_tokens_skipped
                post_refit_sampling_start = flashinfer_sampling_calls
            for i in order:
                engine.add_request(
                    wave * 10 + i + 1,
                    prompts[i],
                    SamplingParams(
                        seed=100 + i if sampling_backend == "torch" else None,
                        num_tokens_to_generate=8,
                        temperature=1.0,
                        top_k=torch_top_k,
                        top_p=1.0,
                        return_log_probs=True,
                        skip_prompt_log_probs=True,
                        termination_id=-1,
                    ),
                )
            for _ in range(200):
                if not engine.has_unfinished_requests():
                    break
                result = engine.step_modern()
                used_graph = int(context.using_cuda_graph_this_step())
                graph_steps += used_graph
                if wave == 6:
                    post_refit_graph_steps += used_graph
                    post_refit_chunked_steps += int(context.chunked_prefill_request_id != -1)
                for request in result["finished_requests"]:
                    outputs[request.request_id] = (
                        list(request.generated_tokens),
                        list(request.generated_log_probs),
                    )
            assert not engine.has_unfinished_requests(), "Cached suffix made no scheduling progress"
            if wave == 6:
                assert engine._prefill_tokens_skipped == skipped_before_refit_request
                if graphs:
                    assert post_refit_graph_steps > 0
                if chunked:
                    assert post_refit_chunked_steps > 0
        if sampling_backend == "torch":
            if not exercise_real_refit:
                for first_id, repeated_id in ((1, 11), (2, 12)):
                    first_tokens, first_log_probs = outputs[first_id]
                    repeated_tokens, repeated_log_probs = outputs[repeated_id]
                    assert first_tokens == repeated_tokens
                    assert len(first_log_probs) == len(repeated_log_probs)
                    for first, repeated in zip(first_log_probs, repeated_log_probs):
                        if math.isnan(first) or math.isnan(repeated):
                            assert math.isnan(first) and math.isnan(repeated)
                        else:
                            assert first == repeated

            before_tokens, before_log_probs = outputs[1]
            after_tokens, after_log_probs = outputs[61]
            assert len(before_log_probs) == len(after_log_probs)
            finite_log_probs_differ = any(
                (math.isnan(before) != math.isnan(after))
                or (not math.isnan(before) and before != after)
                for before, after in zip(before_log_probs, after_log_probs)
            )
            assert before_tokens != after_tokens or finite_log_probs_differ
        else:
            # FlashInfer owns one global Philox stream and intentionally does
            # not promise request-local seeded trajectories. Prove that its
            # real kernel consumed post-refit logits, then assert support and
            # alignment invariants instead of cross-RNG token identity.
            assert post_refit_sampling_start is not None
            assert flashinfer_sampling_calls > post_refit_sampling_start
            for request_id in (61, 5999, 6000):
                tokens, log_probs = outputs[request_id]
                assert len(tokens) == len(log_probs) == 8
                assert all(0 <= token < 128 for token in tokens)
                assert all(math.isfinite(float(log_prob)) for log_prob in log_probs)
        if exercise_real_refit:
            assert len(outputs[5999][0]) == 8
            assert len(outputs[6000][0]) == 8
        if prefix:
            assert engine._prefill_tokens_skipped > 0
        if graphs:
            assert graph_steps > 0
            if kv_cache_management_mode != KVCacheManagementMode.PERSIST:
                assert graph_recaptures == 1
        del engine, context, model, source_model, target_core, source_core
        clear_all_caches()
        delete_cuda_graphs()
        gc.collect()
        return outputs

    @pytest.mark.parametrize("graphs", [False, True])
    @pytest.mark.parametrize("chunked", [False, True])
    @torch.inference_mode()
    def test_cache_matches_uncached_eager(self, graphs, chunked, monkeypatch):
        monkeypatch.setenv("MAMBA_DETERMINISTIC", "1")
        monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
        monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
        with set_batch_invariant_mode(True, backend="te_native"):
            baseline = self._run(
                prefix=False, graphs=False, chunked=False, exercise_real_refit=False
            )
            cached = self._run(
                prefix=True, graphs=graphs, chunked=chunked, exercise_real_refit=False
            )
        assert cached == baseline

    @pytest.mark.parametrize(
        "kv_cache_management_mode",
        [
            KVCacheManagementMode.PERSIST,
            KVCacheManagementMode.OFFLOAD,
            KVCacheManagementMode.RECOMPUTE,
        ],
    )
    @torch.inference_mode()
    def test_refit_across_reallocating_cache_lifecycle(self, kv_cache_management_mode, monkeypatch):
        """A real refit remains observable across every cache lifecycle mode."""
        monkeypatch.setenv("MAMBA_DETERMINISTIC", "1")
        monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
        monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
        with set_batch_invariant_mode(True, backend="te_native"):
            self._run(
                prefix=True,
                graphs=True,
                chunked=False,
                kv_cache_management_mode=kv_cache_management_mode,
            )

    @torch.inference_mode()
    def test_refit_executes_flashinfer_sampling(self, monkeypatch):
        """FlashInfer samples valid logits produced after the real refit."""
        pytest.importorskip("flashinfer", reason="FlashInfer sampling is not installed")
        monkeypatch.setenv("MAMBA_DETERMINISTIC", "1")
        monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
        monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
        with set_batch_invariant_mode(True, backend="te_native"):
            self._run(prefix=False, graphs=False, chunked=False, sampling_backend="flashinfer")
