# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sampling oracles separate deterministic forward results from shared RNG draws."""

import asyncio
import copy

import pytest
import torch

from tests.unit_tests.inference.coordinator_pairwise_utils import greedy_params, routed_model, until


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["torch", "flashinfer"])
@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "processed_logprobs"])
@pytest.mark.parametrize("filters", [(4, 0.0), (0, 0.7)], ids=["top-k", "top-p"])
async def test_routed_stochastic_first_step_distribution(
    monkeypatch, backend, logprobs_mode, filters
):
    """Same prompt/history gives the same distribution, not necessarily the same draw."""
    prompt = list(range(4, 20))
    params = greedy_params(
        num_tokens_to_generate=1,
        temperature=0.7,
        top_k=filters[0],
        top_p=filters[1],
        return_log_probs=True,
        skip_prompt_log_probs=True,
    )
    async with routed_model(
        monkeypatch, sampling_backend=backend, logprobs_mode=logprobs_mode
    ) as h:
        samples = []
        sampler = h.engine.controller._sampling
        sample = sampler.sample_kernel

        def observe(logits, n, context, **kwargs):
            indices = kwargs.get("gather_indices")
            rows = logits[:n] if indices is None else logits[indices[:n]]
            raw = rows.detach().clone()
            distribution = sampler.log_probs_kernel(raw, context).detach().clone()
            rng_before = h.engine.controller.sampling_rng.get_state()
            output = sample(logits, n, context, **kwargs)
            rng_after = h.engine.controller.sampling_rng.get_state()
            assert not torch.equal(rng_before, rng_after), "Stochastic sampling must consume RNG"
            ids = context.request_ids[
                context.paused_request_count : context.total_request_count
            ].tolist()
            samples.append((ids, raw.cpu(), distribution.cpu(), output[:n].clone().cpu()))
            return output

        monkeypatch.setattr(sampler, "sample_kernel", observe)
        await h.direct(prompt, params)
        assert samples and len(samples[0][0]) == 1
        _, direct_logits, direct_distribution, _ = samples[0]
        samples.clear()
        await h.start()
        await h.pause()
        pending = []
        if h.rank == 0:
            pending = [c.add_request(prompt, copy.deepcopy(params)) for c in h.clients]
            await until(lambda: len(h.service.coordinator.request_id_to_rank) == h.dp_size)
            assert h.service.coordinator._pending_counts.tolist() == [1] * h.dp_size
        await h.barrier()
        await h.unpause()
        if h.rank == 0:
            results = await asyncio.wait_for(asyncio.gather(*pending), timeout=60)
            assert all(len(r["generated_tokens"]) == 1 for r in results)
            assert all(len(r["generated_log_probs"]) == 1 for r in results)
        await h.barrier()
        assert len(samples) == 1, "One-token requests sample exactly one distribution"
        ids, logits, distribution, tokens = samples[0]
        assert len(ids) == 1 and ids[0] != 10001
        assert any(ids[0] in step["ids"] for step in h.witnesses)
        torch.testing.assert_close(logits, direct_logits, rtol=0, atol=0)
        torch.testing.assert_close(distribution, direct_distribution, rtol=0, atol=0)
        assert torch.isfinite(distribution[0, tokens[0]])
        assert 1 < torch.isfinite(distribution[0]).sum() < h.config.vocab_size
        # Equality of token values is neither required nor forbidden.
        assert tokens.numel() == 1
        h.assert_retired()


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("interval", [1, 3])
async def test_routed_stream_holdback_keeps_top_n_scores_aligned(monkeypatch, interval):
    """Withheld stop-prefix tokens must not leak their top-N scores in a partial."""
    prompt = list(range(4, 20))
    params = greedy_params(
        num_tokens_to_generate=12,
        return_log_probs=True,
        skip_prompt_log_probs=True,
        top_n_logprobs=3,
        streaming_interval=interval,
    )
    async with routed_model(monkeypatch) as h:
        direct = await h.direct(prompt, params)
        # Choose a genuine three-token stop pattern absent from the model's output.
        # Its two-token holdback still executes at every streaming boundary.
        generated = direct["generated_tokens"]
        stop = next(
            [token, token, token]
            for token in range(h.config.vocab_size)
            if not any(generated[i : i + 3] == [token] * 3 for i in range(len(generated) - 2))
        )
        params.stop_words = [" ".join(map(str, stop))]
        await h.start()
        if h.rank == 0:
            stream = h.clients[0].add_request_streaming(prompt, params)

            async def collect():
                return [item async for item in stream]

            replies = await asyncio.wait_for(collect(), timeout=60)
            final = replies[-1]["final"]
            assert final["generated_tokens"] == generated
            partials = [r["partial"] for r in replies if "partial" in r]
            assert partials, "The stop holdback must coexist with real incremental delivery"
            sent = 0
            for part in partials:
                n = len(part["new_tokens"])
                assert len(part["new_top_n_logprobs"]) == n
                assert len(part["new_log_probs"]) == n
                assert part["new_tokens"] == generated[sent : sent + n]
                assert (
                    part["new_top_n_logprobs"] == final["generated_top_n_logprobs"][sent : sent + n]
                )
                sent += n
            assert sent <= len(generated) - 2, "Two stop-prefix tokens remain withheld until final"
        await h.barrier()
        h.assert_retired()


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("echo,detokenize", [(False, False), (True, True), (True, False)])
async def test_routed_output_opt_ins(monkeypatch, echo, detokenize):
    """Wire prompt/text opt-ins survive routing while total length stays exact."""
    prompt = list(range(4, 20))
    params = greedy_params(
        num_tokens_to_generate=None,
        num_tokens_total=len(prompt) + 5,
        return_prompt_tokens=echo,
        detokenize_generations=detokenize,
    )
    async with routed_model(monkeypatch) as h:
        await h.start()
        if h.rank == 0:
            result = await asyncio.wait_for(h.clients[0].add_request(prompt, params), timeout=60)
            assert result["prompt_length"] == len(prompt)
            assert len(result["generated_tokens"]) == 5
            assert (result["prompt_tokens"] is not None) == echo
            if detokenize:
                assert result["generated_text"] == h.tokenizer.detokenize(
                    result["generated_tokens"]
                )
            else:
                assert result["generated_text"] is None
            assert any(0 in step["ids"] for step in h.witnesses)
        await h.barrier()
        h.assert_retired()
