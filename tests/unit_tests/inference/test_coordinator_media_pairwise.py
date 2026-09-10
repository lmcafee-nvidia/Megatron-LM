# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Media affinity must reach a real vision encoder, projection, and language model."""

import asyncio
import copy

import pytest
import torch

from megatron.core.inference.config import (
    MediaCacheCoordinatorPolicy,
    PrefixCachingCoordinatorPolicy,
    PrefixCachingEvictionPolicy,
)
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.headers import Headers
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.inference.coordinator_pairwise_utils import greedy_params, routed_model, until


def _install_vlm(h, monkeypatch):
    """Reuse the dense context layout with a genuine tiny CLIP/LLaVA model."""
    language = h.engine.controller.model_config
    language.language_model_type = "dummy"
    vision = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
    )
    vision.vision_model_type = "clip"
    projection = TransformerConfig(
        num_layers=1,
        hidden_size=language.hidden_size,
        ffn_hidden_size=32,
        num_attention_heads=1,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
    )
    submodules = get_gpt_layer_with_transformer_engine_submodules()
    torch.manual_seed(887)
    model = (
        LLaVAModel(
            language_transformer_config=language,
            language_transformer_layer_spec=ModuleSpec(
                module=TransformerLayer, submodules=submodules
            ),
            language_vocab_size=h.config.vocab_size,
            language_max_sequence_length=h.config.max_sequence_length,
            vision_transformer_config=vision,
            vision_transformer_layer_spec=ModuleSpec(
                module=TransformerLayer, submodules=copy.deepcopy(submodules)
            ),
            drop_vision_class_token=True,
            vision_projection_config=projection,
            vision_projection_layer_spec=copy.deepcopy(get_submodules(submodules.mlp)),
            img_h=32,
            img_w=32,
            patch_dim=16,
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    context = h.engine.context
    context.config.vision_embedding_cache_max_bytes = 1024 * 1024
    wrapper = VLMInferenceWrapper(model, context)
    wrapper.prep_model_for_inference()
    monkeypatch.setattr(h.tokenizer, "convert_tokens_to_ids", lambda token: 99, raising=False)
    h.engine = DynamicInferenceEngine(TextGenerationController(wrapper, h.tokenizer), context)
    admissions, calls = [], {"vision": 0, "projection": 0}
    for name, module in (("vision", model.vision_model), ("projection", model.vision_projection)):
        forward = module.forward

        def observed(*args, forward=forward, name=name, **kwargs):
            result = forward(*args, **kwargs)
            calls[name] += 1
            return result

        monkeypatch.setattr(module, "forward", observed)
    build = h.engine._build_vlm_request

    def build_request(**kwargs):
        before = dict(calls)
        request = build(**kwargs)
        admissions.append(
            dict(
                id=request.request_id,
                salt=request.block_hash_salt,
                vision=calls["vision"] - before["vision"],
                projection=calls["projection"] - before["projection"],
            )
        )
        return request

    monkeypatch.setattr(h.engine, "_build_vlm_request", build_request)

    def observe_language():
        forward = model.language_model.forward

        def observed(*args, **kwargs):
            ids = context.request_ids[
                context.paused_request_count : context.total_request_count
            ].tolist()
            mask = context.current_image_token_mask() if context.has_vlm_data else None
            image_positions = 0 if mask is None else int((mask >= 0).sum().item())
            cached = {rid: h.engine.get_request(rid).num_cached_tokens for rid in ids}
            result = forward(*args, **kwargs)
            h.witnesses.append(dict(ids=ids, images=image_positions, cached=cached))
            return result

        monkeypatch.setattr(model.language_model, "forward", observed)

    # LLaVA dynamic execution uses forward_lm_only, not the outer model.forward.
    monkeypatch.setattr(h, "_observe_forwards", observe_language)
    return admissions


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("policy", list(MediaCacheCoordinatorPolicy))
async def test_routed_media_affinity_uses_real_cached_embeddings(monkeypatch, policy):
    """Warm rank one, then distinguish affinity from rank-zero load balancing."""
    prompt = [*range(4, 20), 99, 20, 21]
    params = greedy_params(return_prompt_tokens=True)
    image_a = dict(
        imgs=torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32) / 3072,
        num_tiles=torch.tensor([1]),
        num_img_embeddings_per_tile=4,
    )
    image_b = {**image_a, "imgs": image_a["imgs"].flip(-1).contiguous()}
    async with routed_model(
        monkeypatch,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
        coordinator_options={
            "prefix_caching_coordinator_policy": PrefixCachingCoordinatorPolicy.LOAD_BALANCED,
            "media_cache_coordinator_policy": policy,
            "vision_embedding_cache_enabled": True,
        },
    ) as h:
        assert h.dp_size == 2, "This owner-selection schedule is a two-replica row"
        admissions = _install_vlm(h, monkeypatch)
        references = []
        for media in (image_a, image_b):
            future = h.engine.add_request(10001, prompt, copy.deepcopy(params), **media)
            while h.engine.has_unfinished_requests():
                await h.engine.async_step()
            references.append(list((await future).merge().generated_tokens))
            h.engine.reset()
        admissions.clear()
        await h.start()
        await h.pause()
        if h.rank == 0:
            text = h.clients[0].add_request([4, 5, 6], copy.deepcopy(params))
            await until(lambda: len(h.service.coordinator.request_id_to_rank) == 1)
            warm = h.clients[1].add_request(
                prompt, copy.deepcopy(params), multi_modal_data={"image": image_a}
            )
            await until(lambda: len(h.service.coordinator.request_id_to_rank) == 2)
            assert h.service.coordinator.request_id_to_rank[1] == b"mp-coord-1"
        await h.barrier()
        await h.unpause()
        if h.rank == 0:
            _, result = await asyncio.wait_for(asyncio.gather(text, warm), timeout=60)
            assert result["generated_tokens"] == references[0]
            for media, expected in zip((image_a, image_b), references):
                result = await asyncio.wait_for(
                    h.clients[0].add_request(
                        prompt, copy.deepcopy(params), multi_modal_data={"image": media}
                    ),
                    timeout=60,
                )
                assert result["generated_tokens"] == expected
            submits = [e for e in h.service.events if e["header"] == Headers.SUBMIT_REQUEST]
            expected_owner = (
                b"mp-coord-1" if policy == MediaCacheCoordinatorPolicy.AFFINITY else b"mp-coord-0"
            )
            assert submits[2]["after"][2] == expected_owner
        await h.barrier()
        by_id = {row["id"]: row for row in admissions}
        if 1 in by_id:
            assert by_id[1]["vision"] == by_id[1]["projection"] == 1
        if 2 in by_id:
            expected_calls = 0 if policy == MediaCacheCoordinatorPolicy.AFFINITY else 1
            assert by_id[2]["vision"] == by_id[2]["projection"] == expected_calls
        if 3 in by_id:
            assert by_id[3]["vision"] == by_id[3]["projection"] == 1
            assert all(s["cached"].get(3, 0) == 0 for s in h.witnesses)
        if 1 in by_id and 2 in by_id:
            assert by_id[1]["salt"] == by_id[2]["salt"]
        if 2 in by_id and 3 in by_id:
            assert by_id[2]["salt"] != by_id[3]["salt"]
        for rid in by_id:
            assert any(rid in s["ids"] and s["images"] > 0 for s in h.witnesses)
        h.assert_retired()
