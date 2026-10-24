# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import copy
import importlib
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode, PrefixCachingEvictionPolicy
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, EngineState
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)
from megatron.core.inference.quantization.utils import quantize_params_to_mxfp8
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.resharding.refit import (
    clear_all_caches,
    prepare_swap_model_weights,
    swap_model_weights,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import attention as attention_module
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.inference.engines import test_dynamic_engine as dynamic_engine_tests
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig,
    DynamicInferenceEngineTestBase,
    reset_rounder,
)
from tests.unit_tests.test_utilities import Utils

NEMO_RL_COMMIT = "b26733300f0f7f5421c43c8ba4cd9e37e5d5f65d"


class _CoordinatorTokenizer:
    vocab_size = 100
    eod, bos = 99, None

    def tokenize(self, prompt):
        return prompt if isinstance(prompt, list) else [int(token) for token in prompt.split()]

    def detokenize(self, tokens):
        return " ".join(map(str, tokens))


def _mount_nemo_rl_generation_packages(source):
    for name, relative_path in (
        ("nemo_rl.models.generation", "nemo_rl/models/generation"),
        ("nemo_rl.models.generation.megatron", "nemo_rl/models/generation/megatron"),
        ("nemo_rl.weight_sync", "nemo_rl/weight_sync"),
    ):
        package = ModuleType(name)
        package.__path__ = [str(source / relative_path)]
        package.__package__ = name
        sys.modules[name] = package


def _load_nemo_rl_adapter():
    source_value = os.environ.get("NEMO_RL_SOURCE")
    if source_value is None:
        pytest.skip("NEMO_RL_SOURCE is supplied by the NeMo-RL acceptance worker")
    assert os.environ.get("NEMO_RL_SOURCE_COMMIT") == NEMO_RL_COMMIT
    source = Path(source_value).resolve(strict=True)
    sys.path.insert(0, str(source))
    try:
        data_module = importlib.import_module("nemo_rl.distributed.batched_data_dict")
        _mount_nemo_rl_generation_packages(source)
        worker_module = importlib.import_module(
            "nemo_rl.models.generation.megatron.megatron_worker"
        )
    finally:
        sys.path.remove(str(source))
    assert Path(worker_module.__file__).resolve().is_relative_to(source)
    assert Path(data_module.__file__).resolve().is_relative_to(source)
    adapter_cls = type(
        "_TestMegatronPolicyAdapter",
        (worker_module.MegatronGenerationMixin, worker_module.MegatronGenerationRefitMixin),
        {},
    )
    return data_module.BatchedDataDict, adapter_cls


def _load_nemo_rl_token_capture():
    source_value = os.environ.get("NEMO_RL_SOURCE")
    if source_value is None:
        pytest.skip("NEMO_RL_SOURCE is supplied by the NeMo-RL acceptance worker")
    assert os.environ.get("NEMO_RL_SOURCE_COMMIT") == NEMO_RL_COMMIT
    source = Path(source_value).resolve(strict=True)
    gym_source = Path(
        os.environ.get("NEMO_GYM_SOURCE", source / "3rdparty" / "Gym-workspace" / "Gym")
    ).resolve(strict=True)
    sys.path[:0] = [str(source), str(gym_source)]
    try:
        _mount_nemo_rl_generation_packages(source)
        capture_module = importlib.import_module("nemo_rl.models.generation.megatron.token_capture")
        gym_module = importlib.import_module("nemo_gym.token_id_capture")
        records_module = importlib.import_module("nemo_gym.token_id_capture.staging.records")
    finally:
        sys.path.remove(str(gym_source))
        sys.path.remove(str(source))
    assert Path(capture_module.__file__).resolve().is_relative_to(source)
    assert Path(gym_module.__file__).resolve().is_relative_to(gym_source)
    return capture_module, gym_module, records_module


def _protocol_adapter(mixin_cls, *, stop_token_ids=None, ignore_eos=False):
    adapter = object.__new__(mixin_cls)
    adapter.cfg = {
        "generation": {
            "temperature": 0.7,
            "top_k": None,
            "top_p": None,
            "max_new_tokens": 3,
            "ignore_eos": ignore_eos,
            "stop_token_ids": stop_token_ids,
            "stop_strings": None,
        }
    }
    adapter.tokenizer = SimpleNamespace(pad_token_id=0)
    adapter.megatron_tokenizer = SimpleNamespace(eod=99)
    return adapter


def _reply(tokens, scores, reason, prompt=None):
    return SimpleNamespace(
        prompt_tokens=prompt,
        generated_tokens=tokens,
        generated_log_probs=scores,
        finish_reason=reason,
    )


def _assert_exact_scores(actual, expected):
    torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=0)


async def _wait_for_epoch(engine, epoch):
    for _ in range(100):
        if engine._generation_epoch == epoch:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"generation epoch did not reach {epoch}")


@pytest.mark.internal
def test_native_adapter_preserves_stop_tokens_and_finish_reason():
    BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
    adapter = _protocol_adapter(mixin_cls, stop_token_ids=[23, 17, 23])
    params = adapter._build_sampling_params(greedy=False, stop_words=None)
    assert params.stop_token_ids == [17, 23]
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[4, 5], [6, 0]], dtype=torch.long),
            "input_lengths": torch.tensor([2, 1], dtype=torch.long),
        }
    )
    replies = [_reply([7, 8, 9], [-0.7, -0.8, -0.9], "length"), _reply([17], [-0.2], "stop")]
    output = adapter._parse_result_to_batched_data_dict(data, replies)
    assert output["truncated"].tolist() == [True, False]


@pytest.mark.internal
@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (_reply([7], [-0.7], "stop", [8, 8]), "prompt tokens do not match"),
        (_reply([7, 8], [-0.7], "length"), "generated token/logprob lengths do not match"),
    ],
    ids=["prompt-identity", "score-alignment"],
)
def test_native_adapter_rejects_misaligned_reply(reply, message):
    BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
    adapter = _protocol_adapter(mixin_cls)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[4, 5]], dtype=torch.long),
            "input_lengths": torch.tensor([2], dtype=torch.long),
        }
    )
    with pytest.raises(RuntimeError, match=message):
        adapter._parse_result_to_batched_data_dict(data, [reply])


@pytest.mark.internal
@pytest.mark.parametrize("result_count", [0, 2])
def test_native_adapter_rejects_reply_count_mismatch(result_count):
    BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
    adapter = _protocol_adapter(mixin_cls)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[4, 5]], dtype=torch.long),
            "input_lengths": torch.tensor([2], dtype=torch.long),
        }
    )
    reply = _reply([7], [-0.7], "stop")
    with pytest.raises(RuntimeError, match="reply count does not match"):
        adapter._parse_result_to_batched_data_dict(data, [reply] * result_count)


async def _drive_engine(engine, futures, witness=None):
    step_count = 0
    while not all(future.done() for future in futures):
        with torch.inference_mode():
            step = await engine.async_step()
        if witness is not None and step["cuda_graph_request_count"] is not None:
            request_ids = tuple(map(int, step["active_request_ids"]))
            graph_ids = witness["cuda_graph_request_ids"]
            finished_ids = tuple(item.request_id for item in step["finished_requests"])
            graph_ids.append(request_ids + finished_ids)
        step_count += 1
        assert step_count < 100, "dynamic engine did not drain the adapter requests"
        await asyncio.sleep(0)
    return await asyncio.gather(*futures)


def _build_inference_mxfp8_env(test_case, config, *, quantize=True):
    config_class = dynamic_engine_tests.TransformerConfig

    def build_mxfp8_config(**kwargs):
        kwargs.update(fp8_recipe="mxfp8", fp8_param=True)
        return config_class(**kwargs)

    with (
        mock.patch.object(dynamic_engine_tests, "TransformerConfig", build_mxfp8_config),
        mock.patch.object(
            dynamic_engine_tests,
            "get_gpt_layer_with_transformer_engine_spec",
            lambda num_experts=None: get_gpt_layer_with_inference_spec(num_experts=num_experts),
        ),
    ):
        env = test_case._build_test_env(config)
    model = env.engine.controller.inference_wrapped_model.model
    assert any(type(module).__name__.startswith("Inference") for module in model.modules())
    buffers = quantize_params_to_mxfp8(model.decoder, backend="triton") if quantize else None
    assert not quantize or buffers
    return env


class _InProcessInferenceClient:
    def __init__(self, engine, expected_requests, first_request_id, witness):
        self.engine = engine
        self.expected_requests = expected_requests
        self.next_request_id = first_request_id
        self.witness = witness
        self.pending = []
        self.driver = None

    def add_request_with_id(self, prompt, sampling_params, *, multi_modal_data=None):
        assert multi_modal_data is None
        request_id = self.next_request_id
        self.next_request_id += 1
        self.witness["adapter_submissions"].append(
            {
                "request_id": request_id,
                "prompt": tuple(prompt),
                "temperature": sampling_params.temperature,
                "top_k": sampling_params.top_k,
                "top_p": sampling_params.top_p,
            }
        )
        future = self.engine.add_request(request_id, prompt, sampling_params)
        self.pending.append(future)
        if len(self.pending) == self.expected_requests:
            drive = _drive_engine(self.engine, self.pending, self.witness)
            self.driver = asyncio.create_task(drive)
        return request_id, future

    def abort_request(self, request_id):  # pragma: no cover - failure path assertion
        raise AssertionError(f"unexpected adapter abort for request {request_id}")


class TestNeMoRLAdapterContract(DynamicInferenceEngineTestBase):
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        reset_rounder()
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.parametrize("modality", ["image", "video"])
    def test_real_multimodal_adapter_runs_vision_and_language_models(self, modality, monkeypatch):
        monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
        monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
        BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
        source = Path(os.environ["NEMO_RL_SOURCE"]).resolve(strict=True)
        sys.path.insert(0, str(source))
        try:
            PackedTensor = importlib.import_module("nemo_rl.data.multimodal_utils").PackedTensor
        finally:
            sys.path.remove(str(source))
        hidden_size, patch_dim, image_size = 64, 14, 28
        language_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=False,
            bf16=True,
            attention_backend=AttnBackend.flash,
        )
        vision_config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            use_cpu_initialization=False,
            bf16=True,
            attention_backend=AttnBackend.flash,
        )
        projection_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            ffn_hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=False,
            bf16=True,
        )
        submodules = get_gpt_layer_local_submodules()
        vision_spec = get_gpt_layer_with_transformer_engine_spec()
        projection_spec = copy.deepcopy(get_submodules(submodules.mlp))
        assert isinstance(projection_spec, MLPSubmodules)
        language_config.language_model_type = "dummy"
        vision_config.vision_model_type = "radio"
        model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            language_vocab_size=128,
            language_max_sequence_length=64,
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_spec,
            drop_vision_class_token=True,
            vision_projection_config=projection_config,
            vision_projection_layer_spec=projection_spec,
            img_h=image_size,
            img_w=image_size,
            patch_dim=patch_dim,
            dynamic_resolution=True,
        ).cuda()
        model = Float16Module(model.config, model).eval()
        context = StaticInferenceContext(max_batch_size=1, max_sequence_length=64)
        wrapper = VLMInferenceWrapper(model, context)
        vision_calls = []
        vision_forward = wrapper._forward_vision_encoder

        def traced_vision_forward(*args, **kwargs):
            vision_calls.append(modality)
            return vision_forward(*args, **kwargs)

        wrapper._forward_vision_encoder = traced_vision_forward
        frame_count = 1 if modality == "image" else 2
        embedding_count = 4 * frame_count
        prompt = [10] + [-200] * embedding_count + [20]
        patches = torch.randn(1, embedding_count, 3 * patch_dim * patch_dim, dtype=torch.float32)
        sizes = torch.tensor([[image_size, image_size]] * frame_count)
        batch = {
            "input_ids": torch.tensor([prompt]),
            "input_lengths": torch.tensor([len(prompt)]),
            "pixel_values": PackedTensor(
                [patches],
                dim_to_pack=0,
                preprocess_mode="patchify",
                preprocess_kwargs={"patch_dim": patch_dim},
            ),
            "imgs_sizes": PackedTensor([sizes], dim_to_pack=0),
        }
        if modality == "video":
            batch["num_frames"] = PackedTensor([torch.tensor([frame_count])], dim_to_pack=0)
        data = BatchedDataDict(batch)
        adapter = _protocol_adapter(mixin_cls, ignore_eos=True)
        adapter.cfg["generation"]["max_new_tokens"] = 1
        adapter._get_megatron_inference_wrapper_cls = lambda: VLMInferenceWrapper
        prompts, media, params = adapter._prepare_data_for_generation(data, greedy=True)
        assert prompts == [prompt] and set(media[0]) == {modality, "media_tokens_preexpanded"}
        assert params[0].return_prompt_tokens
        engine = object.__new__(DynamicInferenceEngine)
        engine.controller = SimpleNamespace(
            inference_wrapped_model=wrapper, tokenizer=SimpleNamespace(eod=127), pp_group=None
        )
        engine.context = SimpleNamespace(
            block_size_tokens=16, enable_prefix_caching=False, add_vlm_request_data=mock.Mock()
        )
        engine.vision_embedding_cache_max_bytes = sys.maxsize
        engine.allow_stale_multimodal_embeddings = False
        engine._vision_embedding_cache = OrderedDict()
        engine._vision_embedding_cache_bytes = 0
        engine._weight_epoch = 0
        engine.requests = {}

        def build_request(request_id, media_data):
            payload = media_data[modality]
            return engine._build_vlm_request(
                request_id=request_id,
                prompt_str=None,
                tokens=torch.tensor(prompt),
                sampling_params=params[0],
                num_img_embeddings_per_tile=0,
                num_tiles=None,
                precomputed_block_hashes=None,
                media_tokens_preexpanded=True,
                media_cache_key=f"consumer-{modality}",
                **payload,
            )

        def next_token(request):
            context.reset()
            tokens = request.prompt_tokens.unsqueeze(0).cuda()
            with torch.inference_mode():
                logits = wrapper._forward_dynamic(
                    {
                        "tokens": tokens,
                        "position_ids": torch.arange(tokens.shape[1], device="cuda").unsqueeze(0),
                        "attention_mask": None,
                        "image_token_mask": request.image_token_mask.unsqueeze(0),
                        "image_embeddings": request.image_embeddings,
                    }
                )
            step_logits = logits[0, -1].float()
            token = int(step_logits.argmax())
            return token, float(torch.log_softmax(step_logits, dim=-1)[token])

        direct_request = build_request(1300, media[0])
        direct_token, direct_score = next_token(direct_request)
        adapter_request = build_request(1301, media[0])
        adapter_token, adapter_score = next_token(adapter_request)
        cache_key = f"consumer-{modality}"
        assert cache_key in engine._vision_embedding_cache
        assert vision_calls == [modality]
        assert torch.equal(direct_request.image_embeddings, adapter_request.image_embeddings)
        assert (adapter_token, adapter_score) == (direct_token, direct_score)
        reply = _reply([adapter_token], [adapter_score], "length", adapter_request.prompt_tokens)
        output = adapter._parse_result_to_batched_data_dict(data, [reply])
        assert output["output_ids"][0, len(prompt)].item() == direct_token
        assert output["logprobs"][0, len(prompt)].item() == direct_score
        assert output["truncated"].tolist() == [True]
        cached_embedding = engine._vision_embedding_cache[cache_key].embedding
        engine.allow_stale_multimodal_embeddings = True
        engine._invalidate_vision_state()
        assert engine._vision_embedding_cache[cache_key].embedding is cached_embedding
        stale_request = build_request(1302, media[0])
        assert stale_request.image_embeddings is cached_embedding
        assert vision_calls == [modality]
        engine.allow_stale_multimodal_embeddings = False
        engine._invalidate_vision_state()
        assert not engine._vision_embedding_cache
        rebuilt_request = build_request(1303, media[0])
        assert rebuilt_request.image_embeddings is not cached_embedding
        torch.testing.assert_close(
            rebuilt_request.image_embeddings, cached_embedding, rtol=0, atol=0
        )
        assert vision_calls == [modality, modality]

    @pytest.mark.internal
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "feature_config",
        [
            {},
            {
                "context_block_size_tokens": 256,
                "context_max_tokens": 256,
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
                "prefix_caching_eviction_policy": PrefixCachingEvictionPolicy.LRU,
            },
            {
                "context_block_size_tokens": 256,
                "context_max_tokens": 1024,
                "enable_prefix_caching": True,
                "prefix_caching_eviction_policy": PrefixCachingEvictionPolicy.REF_ZERO,
            },
            {"model_provider": "hybrid", "ssm_mixer": "mamba"},
            {"model_provider": "hybrid", "ssm_mixer": "gdn"},
            {"model_provider": "hybrid", "ssm_mixer": "gdp"},
            {"transformer_impl": "transformer_engine"},
            {"transformer_impl": "inference_optimized"},
            {"fp8": True},
            {"fp8": True, "mxfp8": True, "transformer_impl": "inference_optimized"},
            {"window_size": (4, 0), "softmax_type": "off-by-one"},
            {
                "num_cuda_graphs": 2,
                "force_build_cuda_graphs": True,
                "use_cuda_graphs_for_non_decode_steps": False,
            },
            {"num_speculative_tokens": 1, "mtp_use_repeated_layer": True},
            {
                "mla": True,
                "transformer_impl": "transformer_engine",
                "context_block_size_tokens": 64,
                "position_embedding_type": "rope",
            },
            {"mla": True, "context_block_size_tokens": 64, "position_embedding_type": "rope"},
            {"adapter_sampling": {"temperature": 0.73, "top_k": 1, "top_p": 0.91}},
            {"logprobs_mode": "processed_logprobs", "materialize_only_last_token_logits": False},
            dict(hidden_size=256, position_embedding_type="rope", use_flashinfer_fused_rope=True),
            {"position_embedding_type": "rope", "use_flashinfer_fused_rope": True},
        ],
        ids=[
            "dense",
            "prefix-cache-chunked-prefill",
            "prefix-ref-zero",
            "mamba-hybrid",
            "gdn-hybrid",
            "gdp-hybrid",
            "transformer-engine",
            "inference-optimized",
            "fp8",
            "mxfp8",
            "sliding-window-sink",
            "cuda-graphs",
            "mtp",
            "mla-te-latent-cache",
            "mla-local-latent-cache",
            "non-greedy-parameters",
            "processed-logprobs",
            "flashinfer-fused-rope",
            "flashinfer-fused-rope-small-head-fallback",
        ],
    )
    async def test_direct_and_real_adapter_greedy_outputs_match(self, feature_config, monkeypatch):
        BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
        stresses_chunking = feature_config.get("enable_chunked_prefill", False)
        stresses_prefix = feature_config.get("enable_prefix_caching", False)
        adapter_sampling = feature_config.get("adapter_sampling")
        uses_mla = feature_config.get("mla", False)
        uses_mxfp8 = feature_config.get("mxfp8", False)
        if uses_mla and not attention_module.HAVE_FMLA:
            pytest.skip("dynamic latent-cache MLA requires FlashMLA")
        if stresses_prefix:
            prefix_len, suffix_lens = (256, (258, 260)) if stresses_chunking else (256, (2, 4))
            shared_prefix = [index % 89 + 1 for index in range(prefix_len)]
            prompts = [
                shared_prefix + [(index + shift) % 89 + 1 for index in range(length)]
                for shift, length in zip((0, 41), suffix_lens, strict=True)
            ]
        else:
            prompts = [[3, 5, 7, 9], [2, 4, 6, 8, 10, 12]]
        config = DynamicEngineTestConfig(
            num_requests=0,
            max_prompt_length=max(map(len, prompts)),
            max_sequence_length=max(map(len, prompts)) + 8,
            context_max_requests=8,
            context_max_tokens=feature_config.get("context_max_tokens", 64),
            num_gap_steps=0,
            sampling_backend="flashinfer",
            async_sched_mode=AsyncScheduleMode.ASYNC,
            flash_attention_version=None if uses_mla else 2,
            return_log_probs=True,
            skip_prompt_log_probs=True,
            logprobs_mode=feature_config.get("logprobs_mode", "raw_logprobs"),
            **{
                key: value
                for key, value in feature_config.items()
                if key
                not in ("adapter_sampling", "context_max_tokens", "logprobs_mode", "mla", "mxfp8")
            },
        )
        if uses_mla:

            def build_mla_config(**kwargs):
                kwargs.update(
                    hidden_size=512,
                    num_attention_heads=8,
                    add_bias_linear=False,
                    normalization="RMSNorm",
                    qk_layernorm=True,
                    multi_latent_attention=True,
                    q_lora_rank=128,
                    kv_lora_rank=512,
                    qk_head_dim=128,
                    qk_pos_emb_head_dim=64,
                    v_head_dim=128,
                    rope_type="rope",
                    cache_mla_latents=True,
                )
                return MLATransformerConfig(**kwargs)

            with (
                mock.patch.object(dynamic_engine_tests, "TransformerConfig", build_mla_config),
                mock.patch.object(
                    dynamic_engine_tests,
                    "get_gpt_layer_with_transformer_engine_spec",
                    lambda num_experts=None: get_gpt_layer_with_transformer_engine_spec(
                        num_experts=num_experts, qk_layernorm=True, multi_latent_attention=True
                    ),
                ),
                mock.patch.object(
                    dynamic_engine_tests,
                    "get_gpt_layer_local_spec",
                    lambda num_experts=None: ModuleSpec(
                        module=TransformerLayer,
                        submodules=get_gpt_layer_local_submodules(
                            num_experts=num_experts,
                            qk_layernorm=True,
                            multi_latent_attention=True,
                            normalization="RMSNorm",
                        ),
                    ),
                ),
            ):
                env = self._build_test_env(config)
        elif uses_mxfp8:
            env = _build_inference_mxfp8_env(self, config)
        else:
            env = self._build_test_env(config)
        engine = env.engine
        assert not uses_mla or engine.context.cache_mla_latent
        if uses_mxfp8:
            model_config = engine.controller.inference_wrapped_model.model.config
            assert "mxfp8" in str(model_config.fp8_recipe).lower()
        generation_length = 4
        witness = {
            "adapter_submissions": [],
            "model_request_ids": [],
            "sampler_request_ids": [],
            "chunked_prefill_request_ids": [],
            "cuda_graph_request_ids": [],
            "mtp_request_ids": [],
            "flash_mla_request_ids": [],
            "fused_rope_request_ids": [],
        }

        def active_request_ids():
            return tuple(engine.context.request_ids[: engine.context.total_request_count].tolist())

        model = engine.controller.inference_wrapped_model.model
        mixer_calls = []
        mixer_name = {
            "mamba": "MambaMixer",
            "gdn": "GatedDeltaNet",
            "gdp": "GatedDeltaProductMixer",
        }.get(feature_config.get("ssm_mixer"))
        mixer_hooks = [
            module.register_forward_hook(lambda _module, _inputs, _output: mixer_calls.append(1))
            for module in model.modules()
            if type(module).__name__ == mixer_name
        ]
        original_forward = model.forward

        def traced_forward(*args, **kwargs):
            request_ids = active_request_ids()
            witness["model_request_ids"].append(request_ids)
            result = original_forward(*args, **kwargs)
            if not engine.context.config.materialize_only_last_token_logits:
                ctx = engine.context
                shape = result.shape[1], ctx.padded_active_token_count, ctx.num_last_token_logits
                witness.setdefault("all_logit_rows", []).append((request_ids, *shape))
            return result

        model.forward = traced_forward
        sampler = engine.controller._sampling
        original_sample_kernel = sampler.sample_kernel

        def traced_sample_kernel(*args, **kwargs):
            witness["sampler_request_ids"].append(active_request_ids())
            return original_sample_kernel(*args, **kwargs)

        sampler.sample_kernel = traced_sample_kernel
        if feature_config.get("use_flashinfer_fused_rope"):
            original_fused_rope = engine.context.apply_fused_qk_rotary_emb

            def traced_fused_rope(*args, **kwargs):
                witness["fused_rope_request_ids"].append(active_request_ids())
                return original_fused_rope(*args, **kwargs)

            engine.context.apply_fused_qk_rotary_emb = traced_fused_rope
        if uses_mla:
            original_flash_mla = attention_module.flash_mla_with_kvcache

            def traced_flash_mla(*args, **kwargs):
                witness["flash_mla_request_ids"].append(active_request_ids())
                return original_flash_mla(*args, **kwargs)

            monkeypatch.setattr(attention_module, "flash_mla_with_kvcache", traced_flash_mla)
        if feature_config.get("num_speculative_tokens", 0):
            original_mtp_step = engine.controller._run_async_sched_sample_mtp

            def traced_mtp_step(*args, **kwargs):
                witness["mtp_request_ids"].append(active_request_ids())
                return original_mtp_step(*args, **kwargs)

            engine.controller._run_async_sched_sample_mtp = traced_mtp_step
        direct_params = [
            SamplingParams(
                temperature=(adapter_sampling or {}).get("temperature", 0),
                top_k=(adapter_sampling or {}).get("top_k", 1),
                top_p=(adapter_sampling or {}).get("top_p", 0.0),
                skip_prompt_log_probs=True,
                return_log_probs=True,
                num_tokens_to_generate=generation_length,
                termination_id=-1,
                detokenize_stop_sequence=True,
            )
            for _ in prompts
        ]
        if stresses_chunking:
            warm_params = copy.deepcopy(direct_params[0])
            warm_params.num_tokens_to_generate = 1
            await _drive_engine(engine, [engine.add_request(90, shared_prefix, warm_params)])
        direct_futures = [
            engine.add_request(request_id, prompt, params)
            for request_id, prompt, params in zip(
                range(100, 100 + len(prompts)), prompts, direct_params, strict=True
            )
        ]
        direct = await _drive_engine(engine, direct_futures)
        if stresses_chunking:
            assert torch.is_inference(engine.context.request_ids)
            engine.reset()
            await _drive_engine(
                engine, [engine.add_request(190, shared_prefix, copy.deepcopy(warm_params))]
            )
        engine._prefill_tokens_skipped = 0
        witness["mtp_request_ids"].clear()
        witness["flash_mla_request_ids"].clear()
        witness["fused_rope_request_ids"].clear()
        original_schedule = engine.schedule_waiting_requests

        def traced_schedule():
            original_schedule()
            if (request_id := engine.context.chunked_prefill_request_id) >= 0:
                witness["chunked_prefill_request_ids"].append(int(request_id))

        engine.schedule_waiting_requests = traced_schedule
        if uses_mla:
            for layer in engine.controller.inference_wrapped_model.model.decoder.layers:
                assert layer.self_attention.kv_layernorm.weight.is_cuda
                assert layer.self_attention.linear_kv_up_proj_linear.weight.is_cuda
        adapter = object.__new__(mixin_cls)
        adapter.cfg = {
            "generation": {
                "temperature": (adapter_sampling or {}).get("temperature", 0.73),
                "top_k": (adapter_sampling or {}).get("top_k", 17),
                "top_p": (adapter_sampling or {}).get("top_p", 0.91),
                "max_new_tokens": generation_length,
                "ignore_eos": True,
                "stop_token_ids": None,
                "stop_strings": None,
            }
        }
        adapter.tokenizer = SimpleNamespace(pad_token_id=0)
        adapter.megatron_tokenizer = SimpleNamespace(eod=99)
        adapter.inference_client = _InProcessInferenceClient(
            engine, expected_requests=len(prompts), first_request_id=200, witness=witness
        )
        max_prompt_length = max(map(len, prompts))
        padded = torch.tensor(
            [prompt + [0] * (max_prompt_length - len(prompt)) for prompt in prompts],
            dtype=torch.long,
            device="cuda",
        )
        data = BatchedDataDict(
            {
                "input_ids": padded,
                "input_lengths": torch.tensor(
                    list(map(len, prompts)), dtype=torch.long, device="cuda"
                ),
            }
        )
        lowered_prompts, media, lowered_params = adapter._prepare_data_for_generation(
            data, greedy=adapter_sampling is None
        )
        assert lowered_prompts == prompts and media == [None, None]
        expected_sampling = (
            (0, 1, 0.0, -1)
            if adapter_sampling is None
            else (
                adapter_sampling["temperature"],
                adapter_sampling["top_k"],
                adapter_sampling["top_p"],
                -1,
            )
        )
        assert all(
            (params.temperature, params.top_k, params.top_p, params.termination_id)
            == expected_sampling
            for params in lowered_params
        )
        adapter_requests = await adapter._generate_with_persistent_engine(
            lowered_prompts, media, lowered_params
        )
        await adapter.inference_client.driver
        output = adapter._parse_result_to_batched_data_dict(data, adapter_requests)
        assert [entry["prompt"] for entry in witness["adapter_submissions"]] == [
            tuple(prompt) for prompt in prompts
        ]
        assert all(
            (entry["temperature"], entry["top_k"], entry["top_p"]) == expected_sampling[:3]
            for entry in witness["adapter_submissions"]
        )
        for index, expected in enumerate(direct):
            prompt_length = len(prompts[index])
            actual_length = int(output["generation_lengths"][index])
            assert actual_length == generation_length
            assert output["output_ids"][index, :prompt_length].tolist() == prompts[index]
            end = prompt_length + actual_length
            actual_tokens = output["output_ids"][index, prompt_length:end]
            assert actual_tokens.tolist() == expected.generated_tokens
            assert output["unpadded_sequence_lengths"][index].item() == end
            assert output["output_ids"][index, end:].eq(0).all()
            assert output["logprobs"][index, end:].eq(0).all()
            actual_scores = output["logprobs"][index, prompt_length:end]
            _assert_exact_scores(actual_scores, expected.generated_log_probs)
            if feature_config.get("logprobs_mode") == "processed_logprobs":
                assert actual_scores.eq(0).all()
        adapter_ids = {200, 201}
        assert adapter_ids.issubset(
            {request_id for step in witness["model_request_ids"] for request_id in step}
        )
        assert adapter_ids.issubset(
            {request_id for step in witness["sampler_request_ids"] for request_id in step}
        )
        if not engine.context.config.materialize_only_last_token_logits:
            rows = [row[1:] for row in witness["all_logit_rows"] if adapter_ids & set(row[0])]
            assert rows and all(actual == padded for actual, padded, _ in rows)
            assert any(padded > selected for _, padded, selected in rows)
        assert engine.context.async_sched_step_count > 0
        if stresses_chunking:
            assert witness["chunked_prefill_request_ids"]
            assert set(witness["chunked_prefill_request_ids"]) <= adapter_ids
            assert engine._prefill_tokens_skipped > 0
        if stresses_prefix:
            allocator = engine.context.kv_block_allocator
            assert engine._prefill_tokens_skipped > 0
            assert not torch.count_nonzero(allocator.block_ref_counts)
            if (
                feature_config["prefix_caching_eviction_policy"]
                == PrefixCachingEvictionPolicy.REF_ZERO
            ):
                assert not allocator.kv_hash_to_block_id
            else:
                assert allocator.kv_hash_to_block_id
        if feature_config.get("num_cuda_graphs") is not None:
            graph_ids = {
                request_id
                for step_request_ids in witness["cuda_graph_request_ids"]
                for request_id in step_request_ids
            }
            assert graph_ids and graph_ids <= adapter_ids
        if feature_config.get("num_speculative_tokens", 0):
            mtp_ids = {
                request_id
                for step_request_ids in witness["mtp_request_ids"]
                for request_id in step_request_ids
            }
            assert mtp_ids and mtp_ids <= adapter_ids
        if uses_mla:
            mla_ids = {
                request_id for step in witness["flash_mla_request_ids"] for request_id in step
            }
            assert mla_ids and mla_ids <= adapter_ids
        if feature_config.get("use_flashinfer_fused_rope"):
            rope_ids = {
                request_id for step in witness["fused_rope_request_ids"] for request_id in step
            }
            assert rope_ids and rope_ids <= adapter_ids
        if mixer_name is not None:
            assert mixer_hooks and mixer_calls
        for hook in mixer_hooks:
            hook.remove()
        assert not engine.requests and not engine.has_unfinished_requests()

    @pytest.mark.internal
    @pytest.mark.asyncio
    async def test_real_adapter_joint_sampling_matches_analytic_distribution(self):
        BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
        sample_count = 128
        prompt = [3, 5, 7, 9]
        engine = self._build_test_env(
            DynamicEngineTestConfig(
                num_requests=0,
                max_prompt_length=len(prompt),
                max_sequence_length=len(prompt) + 1,
                context_max_requests=sample_count,
                context_max_tokens=sample_count * len(prompt),
                sampling_backend="flashinfer",
                async_sched_mode=AsyncScheduleMode.ASYNC,
                return_log_probs=True,
                skip_prompt_log_probs=True,
                logprobs_mode="raw_logprobs",
            )
        ).engine
        observed = []
        sampler = engine.controller._sampling
        original_sample_kernel = sampler.sample_kernel

        def traced_sample_kernel(logits, n, context, **kwargs):
            result = original_sample_kernel(logits, n, context, **kwargs)
            observed.append(
                {
                    "logits": logits[:n].detach().float().cpu(),
                    "temperature": context.gpu_view.temperature[:n].cpu().clone(),
                    "top_k": context.gpu_view.top_k[:n].cpu().clone(),
                    "top_p": context.gpu_view.top_p[:n].cpu().clone(),
                    "tokens": result[:n].cpu().clone(),
                    "no_top_k": kwargs["no_top_k"],
                    "no_top_p": kwargs["no_top_p"],
                }
            )
            return result

        sampler.sample_kernel = traced_sample_kernel
        adapter = _protocol_adapter(mixin_cls, ignore_eos=True)
        adapter.cfg["generation"].update(temperature=0.73, top_k=4, top_p=0.9, max_new_tokens=1)
        adapter.inference_client = _InProcessInferenceClient(
            engine,
            expected_requests=sample_count,
            first_request_id=900,
            witness={"adapter_submissions": []},
        )
        prompts = [prompt] * sample_count
        data = BatchedDataDict(
            {
                "input_ids": torch.tensor(prompts, device="cuda"),
                "input_lengths": torch.full((sample_count,), len(prompt), device="cuda"),
            }
        )
        lowered_prompts, media, params = adapter._prepare_data_for_generation(data, greedy=False)
        replies = await adapter._generate_with_persistent_engine(lowered_prompts, media, params)
        await adapter.inference_client.driver
        output = adapter._parse_result_to_batched_data_dict(data, replies)
        assert len(observed) == 1
        step = observed[0]
        assert not step["no_top_k"] and not step["no_top_p"]
        assert step["temperature"].tolist() == pytest.approx([0.73] * sample_count)
        assert step["top_k"].tolist() == [4] * sample_count
        assert step["top_p"].tolist() == pytest.approx([0.9] * sample_count)
        torch.testing.assert_close(
            step["logits"], step["logits"][0].expand_as(step["logits"]), rtol=0, atol=0
        )
        raw_log_probs = torch.log_softmax(step["logits"][0], dim=-1)
        scaled_probs = torch.softmax(step["logits"][0] / 0.73, dim=-1)
        top_probs, top_tokens = scaled_probs.topk(4)
        sorted_probs, order = top_probs.sort(descending=True)
        sorted_tokens = top_tokens[order]
        keep = sorted_probs.cumsum(0) - sorted_probs < 0.9
        expected_probs = sorted_probs[keep]
        expected_probs /= expected_probs.sum()
        allowed_tokens = sorted_tokens[keep]
        assert len(allowed_tokens) > 1
        sampled = step["tokens"]
        assert torch.isin(sampled, allowed_tokens).all()
        assert output["output_ids"][:, len(prompt)].cpu().equal(sampled)
        torch.testing.assert_close(
            output["logprobs"][:, len(prompt)].cpu(), raw_log_probs[sampled], rtol=2e-4, atol=2e-4
        )
        for token, probability in zip(allowed_tokens, expected_probs, strict=True):
            observed_probability = (sampled == token).float().mean()
            standard_error = torch.sqrt(probability * (1 - probability) / sample_count)
            assert abs(observed_probability - probability) <= 5 * standard_error + (
                1 / sample_count
            )
        assert not engine.requests and not engine.has_unfinished_requests()

    @pytest.mark.internal
    @pytest.mark.asyncio
    async def test_public_async_adapter_preserves_row_attribution(self):
        BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
        engine = self._build_test_env(
            DynamicEngineTestConfig(
                num_requests=0,
                max_prompt_length=6,
                max_sequence_length=10,
                context_max_requests=4,
                context_max_tokens=32,
                sampling_backend="flashinfer",
                async_sched_mode=AsyncScheduleMode.ASYNC,
                return_log_probs=True,
                skip_prompt_log_probs=True,
                logprobs_mode="raw_logprobs",
            )
        ).engine
        prompts = [[3, 5, 7, 9], [2, 4, 6, 8, 10, 12]]
        direct = await _drive_engine(
            engine,
            [
                engine.add_request(
                    request_id,
                    prompt,
                    SamplingParams(
                        temperature=0,
                        top_k=1,
                        top_p=0,
                        return_log_probs=True,
                        skip_prompt_log_probs=True,
                        num_tokens_to_generate=3,
                        termination_id=-1,
                    ),
                )
                for request_id, prompt in enumerate(prompts, 1200)
            ],
        )
        adapter = _protocol_adapter(mixin_cls, ignore_eos=True)
        adapter.cfg["generation"]["max_new_tokens"] = 3
        adapter.inference_client = _InProcessInferenceClient(
            engine, expected_requests=2, first_request_id=1210, witness={"adapter_submissions": []}
        )
        adapter._inference_loop = asyncio.get_running_loop()
        data = BatchedDataDict(
            {
                "input_ids": torch.tensor([prompts[0] + [0, 0], prompts[1]], device="cuda"),
                "input_lengths": torch.tensor([4, 6], device="cuda"),
            }
        )
        completed = {}
        async for index, row in adapter.generate_async(data, greedy=True):
            completed[index] = row
        assert set(completed) == {0, 1}
        for index, expected in enumerate(direct):
            row = completed[index]
            start = len(prompts[index])
            assert row["output_ids"][0, start : start + 3].tolist() == (expected.generated_tokens)
            _assert_exact_scores(
                row["logprobs"][0, start : start + 3], expected.generated_log_probs
            )
        assert engine.context.async_sched_step_count > 0
        assert not engine.requests and not engine.has_unfinished_requests()

    @pytest.mark.internal
    @pytest.mark.asyncio
    async def test_real_stop_and_length_finish_contracts(self):
        BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
        engine = self._build_test_env(
            DynamicEngineTestConfig(
                num_requests=0,
                max_prompt_length=6,
                max_sequence_length=12,
                context_max_requests=8,
                context_max_tokens=64,
                sampling_backend="flashinfer",
                async_sched_mode=AsyncScheduleMode.ASYNC,
                return_log_probs=True,
                skip_prompt_log_probs=True,
                logprobs_mode="raw_logprobs",
            )
        ).engine
        engine.controller.tokenizer = _CoordinatorTokenizer()
        prompts = [[3, 5, 7, 9], [2, 4, 6, 8, 10]]
        generation_length = 4
        direct = await _drive_engine(
            engine,
            [
                engine.add_request(
                    request_id,
                    prompt,
                    SamplingParams(
                        temperature=0,
                        top_k=1,
                        top_p=0,
                        return_log_probs=True,
                        skip_prompt_log_probs=True,
                        num_tokens_to_generate=generation_length,
                        termination_id=-1,
                    ),
                )
                for request_id, prompt in enumerate(prompts, 700)
            ],
        )
        first_tokens = [reply.generated_tokens[0] for reply in direct]

        async def run_adapter_case(
            *,
            selected_prompts=prompts,
            stop_token_ids=None,
            stop_strings=None,
            row_stop_strings=None,
            ignore_eos=True,
            eod=99,
            first_request_id,
        ):
            adapter = _protocol_adapter(
                mixin_cls, stop_token_ids=stop_token_ids, ignore_eos=ignore_eos
            )
            adapter.cfg["generation"]["max_new_tokens"] = generation_length
            adapter.cfg["generation"]["stop_strings"] = stop_strings
            adapter.megatron_tokenizer.eod = eod
            adapter.inference_client = _InProcessInferenceClient(
                engine,
                expected_requests=len(selected_prompts),
                first_request_id=first_request_id,
                witness={"adapter_submissions": []},
            )
            max_prompt_length = max(map(len, selected_prompts))
            batch = {
                "input_ids": torch.tensor(
                    [
                        prompt + [0] * (max_prompt_length - len(prompt))
                        for prompt in selected_prompts
                    ],
                    device="cuda",
                ),
                "input_lengths": torch.tensor(list(map(len, selected_prompts)), device="cuda"),
            }
            if row_stop_strings is not None:
                batch["stop_strings"] = row_stop_strings
            data = BatchedDataDict(batch)
            lowered_prompts, media, params = adapter._prepare_data_for_generation(data, greedy=True)
            replies = await adapter._generate_with_persistent_engine(lowered_prompts, media, params)
            await adapter.inference_client.driver
            return replies, adapter._parse_result_to_batched_data_dict(data, replies)

        token_replies, token_output = await run_adapter_case(
            stop_token_ids=first_tokens, first_request_id=710
        )
        assert all(reply.finish_reason == "stop" for reply in token_replies)
        assert token_output["generation_lengths"].tolist() == [1, 1]
        assert token_output["truncated"].tolist() == [False, False]
        string_replies, string_output = await run_adapter_case(
            stop_strings=[str(first_tokens[0])],
            row_stop_strings=[None, [str(first_tokens[1])]],
            first_request_id=720,
        )
        assert all(reply.finish_reason == "stop" for reply in string_replies)
        assert string_output["generation_lengths"].tolist() == [1, 1]
        assert string_output["truncated"].tolist() == [False, False]
        eos_replies, eos_output = await run_adapter_case(
            selected_prompts=prompts[:1],
            ignore_eos=False,
            eod=first_tokens[0],
            first_request_id=730,
        )
        assert eos_replies[0].finish_reason == "stop"
        assert eos_output["generation_lengths"].tolist() == [1]
        assert eos_output["truncated"].tolist() == [False]
        length_replies, length_output = await run_adapter_case(
            selected_prompts=prompts[:1], ignore_eos=True, eod=first_tokens[0], first_request_id=740
        )
        assert length_replies[0].finish_reason == "length"
        assert length_output["generation_lengths"].tolist() == [generation_length]
        assert length_output["truncated"].tolist() == [True]
        for output in (token_output, string_output, eos_output, length_output):
            assert int(output["output_ids"][0, len(prompts[0])]) == first_tokens[0]
            _assert_exact_scores(
                output["logprobs"][0, len(prompts[0])], direct[0].generated_log_probs[0]
            )
        assert not engine.requests and not engine.has_unfinished_requests()

    @pytest.mark.internal
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("kv_cache_management_mode", "mxfp8"),
        [("persist", False), ("offload", False), ("recompute", False), ("recompute", True)],
        ids=["persist", "offload", "recompute", "recompute-mxfp8"],
    )
    async def test_real_client_coordinator_adapter_matches_direct(
        self, kv_cache_management_mode, mxfp8
    ):
        BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
        config = DynamicEngineTestConfig(
            num_requests=0,
            max_prompt_length=8,
            max_sequence_length=64,
            context_max_requests=8,
            context_max_tokens=64,
            sampling_backend="flashinfer",
            async_sched_mode=AsyncScheduleMode.ASYNC,
            return_log_probs=True,
            skip_prompt_log_probs=True,
            logprobs_mode="raw_logprobs",
            kv_cache_management_mode=kv_cache_management_mode,
            fp8=mxfp8,
            hidden_size=128 if mxfp8 else None,
            transformer_impl="inference_optimized" if mxfp8 else "local",
        )
        if mxfp8:
            engine = _build_inference_mxfp8_env(self, config, quantize=False).engine
            assert (
                "mxfp8"
                in str(engine.controller.inference_wrapped_model.model.config.fp8_recipe).lower()
            )
        else:
            engine = self._build_test_env(config).engine
        engine.controller.tokenizer = _CoordinatorTokenizer()
        prompts = [[3, 5, 7, 9], [2, 4, 6, 8, 10]]
        generation_length, refit_expected, refit_model = 3, None, None
        if kv_cache_management_mode == "recompute":
            source_config = copy.deepcopy(config)
            source_config.fp8 = False
            refit_engine = self._build_test_env(source_config).engine
            refit_model = refit_engine.controller.inference_wrapped_model.model
            base_state = copy.deepcopy(refit_model.state_dict())
            if mxfp8:
                inference_model = engine.controller.inference_wrapped_model.model
                prepare_swap_model_weights(refit_model, inference_model)
                swap_model_weights(refit_model, inference_model, refit_method="nccl")
        direct_futures = [
            engine.add_request(
                request_id,
                prompt,
                SamplingParams(
                    temperature=0,
                    top_k=1,
                    top_p=0.0,
                    skip_prompt_log_probs=True,
                    return_log_probs=True,
                    num_tokens_to_generate=generation_length,
                    termination_id=-1,
                ),
            )
            for request_id, prompt in enumerate(prompts, 300)
        ]
        direct = await _drive_engine(engine, direct_futures)
        if kv_cache_management_mode == "recompute":
            refit_expected = []
            refit_states = []
            for version_seed in (901, 902):
                torch.manual_seed(version_seed)
                with torch.inference_mode():
                    refit_model.load_state_dict(base_state)
                    for parameter in refit_model.parameters():
                        parameter.add_(0.05 * torch.randn_like(parameter))
                expected = await _drive_engine(
                    refit_engine,
                    [
                        refit_engine.add_request(
                            request_id,
                            prompt,
                            SamplingParams(
                                temperature=0,
                                top_k=1,
                                top_p=0,
                                skip_prompt_log_probs=True,
                                return_log_probs=True,
                                num_tokens_to_generate=generation_length,
                                termination_id=-1,
                            ),
                        )
                        for request_id, prompt in enumerate(prompts, 350 + version_seed)
                    ],
                )
                refit_expected.append(expected)
                refit_states.append(copy.deepcopy(refit_model.state_dict()))
            assert any(
                first.generated_tokens != second.generated_tokens
                or first.generated_log_probs != second.generated_log_probs
                for first, second in zip(*refit_expected, strict=True)
            )
            with torch.inference_mode():
                refit_model.load_state_dict(refit_states[0])
            del refit_engine, base_state

        client = None
        try:
            coordinator_address = await engine.start_listening_to_data_parallel_coordinator(
                hostname="127.0.0.1"
            )
            client = InferenceClient(coordinator_address, deserialize=True)
            client.start(loop=asyncio.get_running_loop(), connect_timeout_seconds=10)
            adapter = _protocol_adapter(mixin_cls, ignore_eos=True)
            adapter.inference_client = client
            adapter._token_capture_enabled = True
            adapter._inference_loop = asyncio.get_running_loop()
            await asyncio.to_thread(adapter.set_rollout_weight_version, 0)
            await _wait_for_epoch(engine, 0)
            padded = torch.tensor([prompts[0] + [0], prompts[1]], dtype=torch.long, device="cuda")
            data = BatchedDataDict(
                {"input_ids": padded, "input_lengths": torch.tensor([4, 5], device="cuda")}
            )
            lowered_prompts, media, params = adapter._prepare_data_for_generation(data, greedy=True)

            async def direct_target_reference(first_request_id):
                rng_state = engine.controller.sampling_rng.get_state()
                references = await asyncio.gather(
                    *(
                        engine.add_request(first_request_id + index, *request)
                        for index, request in enumerate(zip(lowered_prompts, params, strict=True))
                    )
                )
                engine.controller.sampling_rng.set_state(rng_state)
                return references

            def assert_mxfp8_refit_content():
                if not mxfp8:
                    return
                from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

                params, matched = dict(refit_model.named_parameters()), 0
                for module_name, module in adapter.inference_model.decoder.named_modules():
                    for name, dst in vars(module).items():
                        if not isinstance(dst, MXFP8Tensor):
                            continue
                        key = f"decoder.{module_name}.{name}".replace("..", ".")
                        src = MXFP8Tensor.from_bf16(params[key].to(dst.device), backend=dst.backend)
                        assert torch.equal(dst.data, src.data) and torch.equal(dst.scale, src.scale)
                        matched += 1
                assert matched

            replies = await asyncio.wait_for(
                adapter._generate_with_persistent_engine(lowered_prompts, media, params), timeout=30
            )
            output = adapter._parse_result_to_batched_data_dict(data, replies)
            assert all(reply.policy_epoch == [(0, 0)] for reply in replies)
            for index, expected in enumerate(direct):
                prompt_length = len(prompts[index])
                assert output["output_ids"][
                    index, prompt_length : prompt_length + generation_length
                ].tolist() == (expected.generated_tokens)
                _assert_exact_scores(
                    output["logprobs"][index, prompt_length : prompt_length + generation_length],
                    expected.generated_log_probs,
                )
            weight_epoch = engine._weight_epoch
            adapter.dynamic_inference_engine = engine
            await adapter._sleep_engine()
            assert engine.state == EngineState.SUSPENDED
            if refit_model is not None:
                adapter.model = refit_model
                adapter.inference_model = engine.controller.inference_wrapped_model.model
                adapter.cfg["generation"]["mcore_generation_config"] = {
                    "refit_backend": "nccl",
                    "refit_execution_batch_bytes": None,
                }
                adapter._inference_model_offloaded = False
                adapter._swap_weights_plan_prepared = False
                adapter.should_disable_forward_pre_hook = False
                adapter.move_model = lambda model, device, **_kwargs: model.to(device)
                adapter._reshard_into_inference_model()
                assert_mxfp8_refit_content()
            await asyncio.to_thread(adapter.set_rollout_weight_version, 1)
            await adapter._wake_engine()
            assert engine.state == EngineState.RUNNING
            assert engine._weight_epoch == weight_epoch + 1
            await _wait_for_epoch(engine, 1)
            expected_after_first_refit = refit_expected[0] if refit_expected else direct
            if mxfp8:
                expected_after_first_refit = await direct_target_reference(500)
            resumed_replies = await asyncio.wait_for(
                adapter._generate_with_persistent_engine(lowered_prompts, media, params), timeout=30
            )
            resumed_output = adapter._parse_result_to_batched_data_dict(data, resumed_replies)
            assert all(reply.policy_epoch == [(0, 1)] for reply in resumed_replies)
            for actual, expected in zip(resumed_replies, expected_after_first_refit, strict=True):
                assert actual.generated_tokens == expected.generated_tokens
                assert actual.generated_log_probs == expected.generated_log_probs
            spanning_params = [adapter._build_sampling_params(greedy=True, stop_words=None)]
            spanning_params[0].num_tokens_to_generate = 48
            spanning_task = asyncio.create_task(
                adapter._generate_with_persistent_engine([prompts[0]], [None], spanning_params)
            )
            for _ in range(10_000):
                if any(
                    len(engine.get_request(request_id).generated_tokens) >= 2
                    for request_id in engine.requests
                ):
                    break
                await asyncio.sleep(0.001)
            assert engine.requests, "spanning request was not observed in flight"
            if refit_model is not None:
                await adapter._sleep_engine()
                with torch.inference_mode():
                    refit_model.load_state_dict(refit_states[1])
                adapter._reshard_into_inference_model()
                assert_mxfp8_refit_content()
                await asyncio.to_thread(adapter.set_rollout_weight_version, 2)
                await adapter._wake_engine()
                assert engine._weight_epoch == weight_epoch + 2
            else:
                await asyncio.to_thread(adapter.set_rollout_weight_version, 2)
            await _wait_for_epoch(engine, 2)
            spanning_reply = (await asyncio.wait_for(spanning_task, timeout=30))[0]
            assert spanning_reply.policy_epoch[0] == (0, 1)
            assert spanning_reply.policy_epoch[-1][1] == 2
            assert spanning_reply.policy_epoch[-1][0] > len(prompts[0])

            if refit_expected is not None:
                final_expected = await direct_target_reference(600) if mxfp8 else refit_expected[1]
                final_replies = await asyncio.wait_for(
                    adapter._generate_with_persistent_engine(lowered_prompts, media, params),
                    timeout=30,
                )
                assert all(reply.policy_epoch == [(0, 2)] for reply in final_replies)
                for actual, expected in zip(final_replies, final_expected, strict=True):
                    assert actual.generated_tokens == expected.generated_tokens
                    assert actual.generated_log_probs == expected.generated_log_probs

            cancel_params = [adapter._build_sampling_params(greedy=True, stop_words=None)]
            cancel_params[0].num_tokens_to_generate = 48
            cancel_task = asyncio.create_task(
                adapter._generate_with_persistent_engine([prompts[1]], [None], cancel_params)
            )
            for _ in range(10_000):
                if engine.requests:
                    break
                await asyncio.sleep(0.001)
            cancelled_request_ids = set(engine.requests)
            assert cancelled_request_ids
            cancel_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancel_task
            for _ in range(200):
                if not engine.requests and not client.abort_futures:
                    break
                await asyncio.sleep(0.01)
            assert cancelled_request_ids.isdisjoint(engine.requests)
            assert not client.abort_futures
        finally:
            if client is not None:
                if engine.state == EngineState.RUNNING:
                    client.pause_engines()
                    await asyncio.wait_for(engine.wait_until(EngineState.PAUSED), timeout=10)
                client.stop_engines()
                await asyncio.wait_for(engine.engine_loop_task, timeout=10)
                client.shutdown_coordinator()
                coordinator_process = getattr(engine, "inference_coordinator_process", None)
                if coordinator_process is not None:
                    await asyncio.to_thread(coordinator_process.join, 10)
                    assert not coordinator_process.is_alive()
                client.stop()
            if refit_model is not None:
                clear_all_caches()

    @pytest.mark.internal
    @pytest.mark.asyncio
    async def test_real_token_capture_stages_exact_concurrent_versioned_payloads(self):
        _, mixin_cls = _load_nemo_rl_adapter()
        capture_module, gym_module, records_module = _load_nemo_rl_token_capture()

        class ConcurrentMemorySink:
            media_pixel_dtype = None

            def __init__(self):
                self.records = []
                self.lock = threading.Lock()
                self.barrier = threading.Barrier(2)
                self.active = 0
                self.peak = 0

            def stage(self, record, *, attachments=None):
                assert attachments is None
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                try:
                    if record.model_call_id in {"call-1", "call-2"}:
                        self.barrier.wait(timeout=10)
                    with self.lock:
                        self.records.append(record)
                finally:
                    with self.lock:
                        self.active -= 1
                return records_module.StageResult(ok=True, staging_key=record.staging_key)

        config = DynamicEngineTestConfig(
            num_requests=0,
            max_prompt_length=8,
            max_sequence_length=64,
            context_max_requests=8,
            context_max_tokens=64,
            sampling_backend="flashinfer",
            async_sched_mode=AsyncScheduleMode.ASYNC,
            return_log_probs=True,
            skip_prompt_log_probs=True,
            logprobs_mode="raw_logprobs",
        )
        engine = self._build_test_env(config).engine
        engine.controller.tokenizer = _CoordinatorTokenizer()
        prompts = [[3, 5, 7, 9], [2, 4, 6, 8]]

        def sampling_params(num_tokens):
            return SamplingParams(
                temperature=0,
                top_k=1,
                top_p=0,
                skip_prompt_log_probs=True,
                return_log_probs=True,
                num_tokens_to_generate=num_tokens,
                termination_id=-1,
            )

        direct = await _drive_engine(
            engine,
            [
                engine.add_request(request_id, prompt, sampling_params(3))
                for request_id, prompt in enumerate(prompts, 800)
            ],
        )
        sink = ConcurrentMemorySink()
        stager = capture_module.TQMegatronTokenStager(sink)
        engine.payload_stager = stager
        client = None
        try:
            address = await engine.start_listening_to_data_parallel_coordinator(
                hostname="127.0.0.1"
            )
            client = InferenceClient(address, deserialize=True)
            client.start(loop=asyncio.get_running_loop(), connect_timeout_seconds=10)
            adapter = _protocol_adapter(mixin_cls, ignore_eos=True)
            adapter.inference_client = client
            adapter._token_capture_enabled = True
            adapter._inference_loop = asyncio.get_running_loop()
            await asyncio.to_thread(adapter.set_rollout_weight_version, 7)
            await _wait_for_epoch(engine, 7)

            futures = []
            for index, prompt in enumerate(prompts, 1):
                admission = records_module.CaptureAdmission(
                    rollout_id="rollout-concurrent", model_call_id=f"call-{index}", mode="text"
                )
                futures.append(
                    client.add_request(
                        prompt,
                        sampling_params(3),
                        offload_params={
                            gym_module.NG_CAPTURE_FIELD: admission.model_dump(mode="json")
                        },
                    )
                )
            replies = await asyncio.wait_for(asyncio.gather(*futures), timeout=30)
            assert sink.peak == 2
            staged = {record.model_call_id: record for record in sink.records}
            assert set(staged) == {"call-1", "call-2"}
            for index, (prompt, expected, reply) in enumerate(
                zip(prompts, direct, replies, strict=True), 1
            ):
                record = staged[f"call-{index}"]
                assert reply.payload_offloaded
                assert reply.generated_tokens == expected.generated_tokens
                assert reply.generated_log_probs is None
                assert record.weight_version == 7
                assert record.token_ids_delta == prompt + expected.generated_tokens
                assert record.token_mask_delta == [0.0] * len(prompt) + [1.0] * 3
                assert record.generation_log_probs_delta == [0.0] * len(prompt) + (
                    expected.generated_log_probs
                )
                coords = reply.payload_stage_metadata[gym_module.NG_COMMIT_COORDS_FIELD]
                assert coords["weight_version"] == 7
                assert coords["staging_key"] == record.staging_key

            spanning_admission = records_module.CaptureAdmission(
                rollout_id="rollout-spanning", model_call_id="call-spanning", mode="text"
            )
            spanning_id, spanning_future = client.add_request_with_id(
                prompts[0],
                sampling_params(48),
                offload_params={
                    gym_module.NG_CAPTURE_FIELD: spanning_admission.model_dump(mode="json")
                },
            )
            for _ in range(10_000):
                request = engine.requests.get(spanning_id)
                if request is not None and len(request.record[-1].generated_tokens) >= 2:
                    break
                await asyncio.sleep(0.001)
            assert spanning_id in engine.requests
            await asyncio.to_thread(adapter.set_rollout_weight_version, 8)
            spanning_reply = await asyncio.wait_for(spanning_future, timeout=30)
            spanning_record = next(
                record for record in sink.records if record.model_call_id == "call-spanning"
            )
            assert spanning_reply.policy_epoch[0] == (0, 7)
            assert spanning_reply.policy_epoch[-1][1] == 8
            assert spanning_record.weight_version == 7
            assert spanning_record.token_ids_delta == (prompts[0] + spanning_reply.generated_tokens)
            assert stager.epoch_span_count == 1
        finally:
            if client is not None:
                client.pause_engines()
                await asyncio.wait_for(engine.wait_until(EngineState.PAUSED), timeout=10)
                client.stop_engines()
                await asyncio.wait_for(engine.engine_loop_task, timeout=10)
                client.shutdown_coordinator()
                coordinator_process = engine.inference_coordinator_process
                await asyncio.to_thread(coordinator_process.join, 10)
                assert not coordinator_process.is_alive()
                client.stop()


@pytest.mark.internal
@pytest.mark.asyncio
async def test_real_noncolocated_consumer_refit_then_adapter_output():
    if Utils.world_size != 4:
        pytest.skip("noncolocated source/destination contract requires four ranks")

    from megatron.core.inference.shards import build_inference_pg_collection
    from megatron.core.resharding.copy_services.gloo_copy_service import GlooCopyService
    from tests.unit_tests.resharding.test_model_swap import (
        _build_gpt,
        _destroy_pg_collection,
        _run_forward,
    )

    BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
    refit_cls = importlib.import_module(
        "nemo_rl.models.generation.megatron.megatron_worker"
    ).MegatronGenerationRefitMixin
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    rank = torch.distributed.get_rank()
    source_pgs = build_inference_pg_collection(world_size=2, tp_size=1, rank_offset=0)
    target_pgs = build_inference_pg_collection(world_size=2, tp_size=1, rank_offset=2)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.auto,
    )
    torch.manual_seed(1234)
    source_model = (
        _build_gpt(config, 128, 8, source_pgs, parallel_output=False).cuda().bfloat16().eval()
        if rank < 2
        else None
    )
    torch.manual_seed(1234)
    target_model = (
        _build_gpt(config, 128, 8, target_pgs, parallel_output=False).cuda().bfloat16().eval()
        if rank >= 2
        else None
    )
    tokens = torch.tensor([[3, 5, 7, 9]], device="cuda")
    positions = torch.arange(tokens.shape[1], device="cuda").unsqueeze(0)
    attention_mask = torch.triu(
        torch.ones((1, 1, 4, 4), dtype=torch.bool, device="cuda"), diagonal=1
    )
    refitter = object.__new__(refit_cls)
    refitter.model = source_model if rank < 2 else target_model
    refitter.refit_execution_batch_bytes = None
    refitter.refit_dst_rank_offset = 0
    with pytest.raises(RuntimeError, match="was not initialized"):
        refitter.swap_weights_via_reshard(is_source=rank < 2)
    refitter.refit_execution_batch_bytes = 1 << 20
    adapter = _protocol_adapter(mixin_cls, ignore_eos=True)
    adapter.cfg["generation"]["max_new_tokens"] = 1
    data = BatchedDataDict({"input_ids": tokens.cpu(), "input_lengths": torch.tensor([4])})
    if target_model is not None:
        inference_config = dynamic_engine_tests.InferenceConfig(
            max_sequence_length=8,
            buffer_size_gb=0.01,
            max_requests=1,
            max_tokens=8,
            sampling_backend="torch",
            pg_collection=target_pgs,
        )
        context = dynamic_engine_tests.DynamicInferenceContext(config, inference_config)
        wrapper = dynamic_engine_tests.GPTInferenceWrapper(target_model, context)
        wrapper.model_is_pipeline_parallel = False
        engine = DynamicInferenceEngine(
            dynamic_engine_tests.TextGenerationController(wrapper, _CoordinatorTokenizer()), context
        )
    previous_token = None
    refit_group = torch.distributed.new_group(backend="gloo")
    refitter.refit_pg = refit_group
    refitter.refit_copy_service = GlooCopyService(group=refit_group)
    try:
        for version, delta in enumerate((0.05, 0.11), 1):
            if source_model is not None:
                with torch.no_grad():
                    for parameter in source_model.parameters():
                        parameter.add_(delta)
                direct_logits = _run_forward(
                    source_model, tokens, positions, attention_mask, source_pgs
                )
            else:
                direct_logits = torch.empty(1, 4, 128, device="cuda", dtype=torch.bfloat16)
            torch.distributed.broadcast(direct_logits, src=0)
            assert refitter.swap_weights_via_reshard(is_source=rank < 2)
            torch.cuda.synchronize()
            if target_model is not None:
                with torch.inference_mode():
                    target_logits = target_model(
                        tokens, positions, attention_mask, runtime_gather_output=True
                    )
                torch.testing.assert_close(target_logits, direct_logits, rtol=0, atol=0)
                if rank == 3:
                    torch.distributed.barrier()
                    continue
                engine._generation_epoch = version
                prompts, media, sampling_params = adapter._prepare_data_for_generation(
                    data, greedy=True
                )
                direct = (
                    await _drive_engine(
                        engine, [engine.add_request(300 + version, prompts[0], sampling_params[0])]
                    )
                )[0]
                witness = {"adapter_submissions": [], "cuda_graph_request_ids": []}
                adapter.inference_client = _InProcessInferenceClient(
                    engine, expected_requests=1, first_request_id=400 + version, witness=witness
                )
                with mock.patch.object(torch.distributed, "get_rank", return_value=0):
                    replies = await adapter._generate_with_persistent_engine(
                        prompts, media, sampling_params
                    )
                await adapter.inference_client.driver
                output = adapter._parse_result_to_batched_data_dict(data, replies)
                assert direct.generated_tokens == replies[0].generated_tokens
                assert direct.generated_log_probs == replies[0].generated_log_probs
                assert replies[0].policy_epoch == [(0, version)]
                assert output["output_ids"][0, 4].item() == int(target_logits[0, -1].argmax())
                assert output["logprobs"][0, 4].item() == direct.generated_log_probs[0]
                assert witness["adapter_submissions"][0]["request_id"] == 400 + version
                assert not engine.requests and not engine.has_unfinished_requests()
                if previous_token is not None:
                    assert output["output_ids"][0, 4].item() != previous_token or not torch.equal(
                        target_logits, previous_logits
                    )
                previous_token, previous_logits = (
                    output["output_ids"][0, 4].item(),
                    target_logits.clone(),
                )
            torch.distributed.barrier()
    finally:
        refitter.refit_copy_service.close()
        clear_all_caches()
        torch.distributed.destroy_process_group(refit_group)
        _destroy_pg_collection(source_pgs)
        _destroy_pg_collection(target_pgs)
        Utils.destroy_model_parallel()


@pytest.fixture
def nemo_parallel(request, monkeypatch):
    topology = request.param
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=topology["tp"],
        pipeline_model_parallel_size=topology["pp"],
        expert_model_parallel_size=topology["ep"],
        expert_tensor_parallel_size=topology.get("etp", 1),
    )
    synchronization_group = torch.distributed.new_group(backend="gloo")
    try:
        yield topology, synchronization_group
    finally:
        torch.distributed.destroy_process_group(synchronization_group)
        delete_cuda_graphs()
        reset_rounder()
        Utils.destroy_model_parallel()


class TestNeMoRLAdapterParallelContract(DynamicInferenceEngineTestBase):
    @pytest.mark.internal
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "nemo_parallel",
        [
            pytest.param({"tp": 2, "pp": 1, "ep": 1}, id="tp2"),
            pytest.param(
                {"tp": 2, "pp": 1, "ep": 1, "sp": True, "transformer_impl": "inference_optimized"},
                id="tp2-sp-inference-optimized",
            ),
            pytest.param({"tp": 1, "pp": 2, "ep": 1}, id="pp2"),
            pytest.param({"tp": 1, "pp": 1, "ep": 2}, id="ep2-moe"),
            pytest.param({"tp": 2, "pp": 1, "ep": 1, "etp": 2, "moe": True}, id="etp2-moe-local"),
            pytest.param(
                {
                    "tp": 2,
                    "pp": 1,
                    "ep": 1,
                    "etp": 2,
                    "moe": True,
                    "transformer_impl": "transformer_engine",
                },
                id="etp2-moe-te",
            ),
            pytest.param({"tp": 1, "pp": 1, "ep": 1}, id="dp2"),
        ],
        indirect=True,
    )
    async def test_real_adapter_parallel_topology(self, nemo_parallel):
        BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
        topology, synchronization_group = nemo_parallel
        rank = torch.distributed.get_rank()
        assert torch.distributed.get_world_size() == 2
        uses_moe = topology["ep"] > 1 or topology.get("moe", False)
        config = DynamicEngineTestConfig(
            num_requests=0,
            max_prompt_length=8,
            max_sequence_length=16,
            context_max_requests=8,
            context_max_tokens=64,
            tensor_model_parallel_size=topology["tp"],
            pipeline_model_parallel_size=topology["pp"],
            expert_model_parallel_size=topology["ep"],
            use_moe_layer_spec=uses_moe,
            sequence_parallel=topology.get("sp", False),
            transformer_impl=topology.get("transformer_impl", "local"),
            sampling_backend="flashinfer",
            async_sched_mode=AsyncScheduleMode.ASYNC,
            return_log_probs=True,
            skip_prompt_log_probs=True,
            logprobs_mode="raw_logprobs",
        )
        transformer_config = dynamic_engine_tests.TransformerConfig

        def build_transformer_config(**kwargs):
            kwargs.update(
                expert_tensor_parallel_size=topology.get("etp", 1),
                num_moe_experts=2 if uses_moe else kwargs["num_moe_experts"],
                add_bias_linear=False if uses_moe else kwargs["add_bias_linear"],
            )
            return transformer_config(**kwargs)

        with mock.patch.object(dynamic_engine_tests, "TransformerConfig", build_transformer_config):
            engine = self._build_test_env(config).engine
        engine.controller.tokenizer = _CoordinatorTokenizer()
        is_dp = topology["tp"] == topology["pp"] == topology["ep"] == 1
        prompts = [[3, 5, 7, 9], [3, 5, 7, 9]] if is_dp else [[3, 5, 7, 9], [2, 4, 6, 8, 10]]
        params = [
            SamplingParams(
                temperature=0.7 if is_dp else 0,
                top_k=0 if is_dp else 1,
                top_p=0,
                skip_prompt_log_probs=True,
                return_log_probs=True,
                num_tokens_to_generate=3,
                termination_id=-1,
            )
            for _ in (prompts[:1] if is_dp else prompts)
        ]
        initial_seed = engine.controller.sampling_rng.initial_seed()
        direct = await _drive_engine(
            engine,
            [
                engine.add_request(request_id, prompt, request_params)
                for request_id, prompt, request_params in zip(
                    range(600, 600 + len(params)), prompts[: len(params)], params, strict=True
                )
            ],
        )
        direct_replies = [(reply.generated_tokens, reply.generated_log_probs) for reply in direct]
        expected_by_dp, seeds_by_dp = [None, None], [None, None]
        if is_dp:
            await asyncio.to_thread(
                torch.distributed.all_gather_object,
                expected_by_dp,
                (direct[0].generated_tokens, direct[0].generated_log_probs),
                group=synchronization_group,
            )
            await asyncio.to_thread(
                torch.distributed.all_gather_object,
                seeds_by_dp,
                initial_seed,
                group=synchronization_group,
            )
            assert seeds_by_dp[1] == seeds_by_dp[0] + 1
        engine.controller.sampling_rng.manual_seed(initial_seed)
        adapter_forward_batches, model = [], engine.controller.inference_wrapped_model.model
        original_forward = model.forward

        def traced_forward(*args, **kwargs):
            adapter_forward_batches.append(
                tuple(engine.context.request_ids[: engine.context.total_request_count].tolist())
            )
            return original_forward(*args, **kwargs)

        model.forward = traced_forward
        moe_forward_batches = []
        moe_hooks = [
            module.register_forward_hook(
                lambda _module, _inputs, _output: moe_forward_batches.append(
                    adapter_forward_batches[-1]
                )
            )
            for module in model.modules()
            if type(module).__name__ == "MoELayer"
        ]
        client, failure = None, None
        try:
            coordinator_address = await engine.start_listening_to_data_parallel_coordinator(
                hostname="127.0.0.1"
            )
            await asyncio.to_thread(torch.distributed.barrier, group=synchronization_group)
            if rank == 0:
                try:
                    client = InferenceClient(coordinator_address, deserialize=True)
                    client.start(loop=asyncio.get_running_loop(), connect_timeout_seconds=10)
                    adapter = _protocol_adapter(mixin_cls, ignore_eos=True)
                    adapter.inference_client = client
                    max_prompt_length = max(map(len, prompts))
                    data = BatchedDataDict(
                        {
                            "input_ids": torch.tensor(
                                [
                                    prompt + [0] * (max_prompt_length - len(prompt))
                                    for prompt in prompts
                                ],
                                device="cuda",
                            ),
                            "input_lengths": torch.tensor(list(map(len, prompts)), device="cuda"),
                        }
                    )
                    lowered_prompts, media, lowered_params = adapter._prepare_data_for_generation(
                        data, greedy=not is_dp
                    )
                    replies = await asyncio.wait_for(
                        adapter._generate_with_persistent_engine(
                            lowered_prompts, media, lowered_params
                        ),
                        timeout=30,
                    )
                    output = adapter._parse_result_to_batched_data_dict(data, replies)
                    expected_replies = expected_by_dp if is_dp else direct_replies
                    actual_replies = [
                        (reply.generated_tokens, reply.generated_log_probs) for reply in replies
                    ]
                    if is_dp:
                        assert sorted(actual_replies) == sorted(expected_replies)
                    else:
                        assert actual_replies == expected_replies
                    assert [reply.request_id for reply in replies] == list(range(len(prompts)))
                    for index, reply in enumerate(replies):
                        prompt_length = len(prompts[index])
                        assert (
                            output["output_ids"][index, prompt_length : prompt_length + 3].tolist()
                            == reply.generated_tokens
                        )
                        _assert_exact_scores(
                            output["logprobs"][index, prompt_length : prompt_length + 3],
                            reply.generated_log_probs,
                        )
                except BaseException as error:
                    failure = error
            await asyncio.to_thread(torch.distributed.barrier, group=synchronization_group)
        finally:
            if rank == 0 and client is not None:
                client.pause_engines()
            await asyncio.wait_for(engine.wait_until(EngineState.PAUSED), timeout=15)
            if rank == 0 and client is not None:
                client.stop_engines()
            await asyncio.wait_for(engine.engine_loop_task, timeout=15)
            if rank == 0 and client is not None:
                client.shutdown_coordinator()
                coordinator_process = engine.inference_coordinator_process
                await asyncio.to_thread(coordinator_process.join, 10)
                assert not coordinator_process.is_alive()
                client.stop()
        if failure is not None:
            raise failure
        assert adapter_forward_batches
        if topology["ep"] > 1:
            adapter_ids = set(range(len(prompts)))
            assert any(adapter_ids.issubset(batch) for batch in adapter_forward_batches)
        if uses_moe:
            assert moe_hooks and moe_forward_batches
            if topology["ep"] > 1:
                assert any(adapter_ids.issubset(batch) for batch in moe_forward_batches)
        for hook in moe_hooks:
            hook.remove()
