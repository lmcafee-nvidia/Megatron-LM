# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
    language = h.engine.controller.model_config
    language.language_model_type = "dummy"
    config = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
    )
    vision = TransformerConfig(**config)
    vision.vision_model_type = "clip"
    projection = TransformerConfig(
        **{
            **config,
            "hidden_size": language.hidden_size,
            "ffn_hidden_size": 32,
            "num_attention_heads": 1,
        }
    )
    submodules = get_gpt_layer_with_transformer_engine_submodules()
    layer = ModuleSpec(module=TransformerLayer, submodules=submodules)
    torch.manual_seed(887)
    model = (
        LLaVAModel(
            language_transformer_config=language,
            language_transformer_layer_spec=layer,
            language_vocab_size=h.config.vocab_size,
            language_max_sequence_length=h.config.max_sequence_length,
            vision_transformer_config=vision,
            vision_transformer_layer_spec=copy.deepcopy(layer),
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
        delta = tuple(calls[name] - before[name] for name in calls)
        admissions.append((request.request_id, h.rank, request.block_hash_salt, *delta))
        return request

    monkeypatch.setattr(h.engine, "_build_vlm_request", build_request)

    def observe_language():
        forward = model.language_model.forward

        def observed(*args, **kwargs):
            active = context.request_ids[
                context.paused_request_count : context.total_request_count
            ].tolist()
            ids = [rid for rid in active if rid in h.engine.requests]
            mask = context.current_image_token_mask() if context.has_vlm_data else None
            positions = 0 if mask is None else int((mask >= 0).sum().item())
            injected = False
            if positions:
                assert len(ids) == 1, "The schedule must have one live media target per owner"
                image_positions = mask[0] >= 0
                actual = kwargs["decoder_input"][image_positions, 0]
                embeddings = context.current_image_embeddings()
                expected = embeddings[mask[0, image_positions], 0].to(actual.dtype)
                injected = torch.equal(actual, expected)
            cached = {rid: h.engine.get_request(rid).num_cached_tokens for rid in ids}
            result = forward(*args, **kwargs)
            if ids:
                h.witnesses.append((h.rank, ids, positions, injected, cached))
            return result

        monkeypatch.setattr(model.language_model, "forward", observed)

    monkeypatch.setattr(h, "_observe_forwards", observe_language)
    return admissions


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("policy", list(MediaCacheCoordinatorPolicy))
async def test_routed_media_affinity_uses_real_cached_embeddings(monkeypatch, policy):
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
            assert len(submits) == 4
            routes = []
            for event in submits:
                new_ids = set(event["after"]) - set(event["before"])
                assert len(new_ids) == 1, event
                request_id = new_ids.pop()
                routes.append((request_id, event["after"][request_id]))
            text_route, warm_route, repeat_route, different_route = routes
            coordinator = h.service.coordinator
            assert text_route[1] != warm_route[1], "Competing text must put the warm on its peer"
            expected_repeat_owner = (
                warm_route[1] if policy == MediaCacheCoordinatorPolicy.AFFINITY else text_route[1]
            )
            assert repeat_route[1] == expected_repeat_owner
            assert different_route[1] == text_route[1]
            route_info = [(rid, coordinator.identity_to_rank_index[owner]) for rid, owner in routes]
        await h.barrier()
        route_payload = [route_info if h.rank == 0 else None]
        torch.distributed.broadcast_object_list(route_payload, src=0)
        route_info = route_payload[0]
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, (admissions, h.witnesses))
        all_admissions = [row for rank_data in gathered for row in rank_data[0]]
        all_witnesses = [row for rank_data in gathered for row in rank_data[1]]
        targets = [rid for rid, _ in route_info[1:]]
        warm_id, repeat_id, different_id = targets
        by_id = {row[0]: row for row in all_admissions}
        assert len(all_admissions) == 3 and set(by_id) == set(targets)
        assert all(
            by_id[rid][1] == owner
            and any(step[0] == owner and rid in step[1] for step in all_witnesses)
            for rid, owner in route_info[1:]
        )

        expected_calls = (1, 0 if policy == MediaCacheCoordinatorPolicy.AFFINITY else 1, 1)
        for request_id, count in zip(targets, expected_calls):
            assert by_id[request_id][3] == by_id[request_id][4] == count
            assert any(
                request_id in step[1] and step[2] > 0 and step[3] for step in all_witnesses
            ), f"Request {request_id} must consume its own projected image embeddings"

        assert by_id[warm_id][2] == by_id[repeat_id][2] != by_id[different_id][2]
        different_steps = [step for step in all_witnesses if different_id in step[1]]
        assert different_steps and all(step[4][different_id] == 0 for step in different_steps)
        h.assert_retired()
