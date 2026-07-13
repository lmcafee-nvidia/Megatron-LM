# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import Tensor


class Sampling(ABC):
    """Abstract base for inference sampling backends.

    Subclasses implement `sample_kernel` and `log_probs_kernel`.
    CUDA graphs are added via `CudaGraphManager`.
    """

    def __init__(self, sampled_tokens_buffer: Optional[Tensor] = None) -> None:
        """Initialize a sampling backend.

        Args:
            sampled_tokens_buffer: Stable destination used by ordinary dynamic sampling.
        """
        self._sampled_tokens_buffer = sampled_tokens_buffer

    @abstractmethod
    def sample_kernel(
        self,
        logits: Tensor,
        n: int,
        context,
        *,
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
        output: Optional[Tensor] = None,
        eager: bool = False,
        cache_key: Any = None,
    ) -> Tensor:
        """Sample `n` tokens from `logits` and return them.

        Args:
            logits: Logits tensor of shape `[>=n, vocab_size]`.
            n: Number of rows to sample.
            context: The active DynamicInferenceContext.
            gather_indices: If provided, only sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: Per-token request mapping; when set, sampling
                parameters are gathered per-token instead of per-request.
            output: Optional caller-owned destination tensor of shape `[n]`.
            eager, cache_key: Consumed by `CudaGraphManager` when it wraps this kernel.

        Returns:
            Sampled token ids of shape `[n]`.
        """
        ...

    def sample_kernel_into(
        self,
        logits: Tensor,
        n: int,
        context,
        *,
        gather_indices: Optional[Tensor] = None,
        eager: bool = False,
        cache_key: Any = None,
    ) -> tuple:
        """Sample ordinary dynamic requests into the stable controller buffer.

        Args:
            logits: Logits tensor of shape `[>=n, vocab_size]`.
            n: Number of rows to sample.
            context: The active DynamicInferenceContext.
            gather_indices: If provided, only sample from `logits[gather_indices[:n], :]`.
            eager: Whether to bypass a wrapped CUDA graph.
            cache_key: CUDA graph lookup key.
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key

        assert self._sampled_tokens_buffer is not None
        assert n <= self._sampled_tokens_buffer.numel()
        self.sample_kernel(
            logits,
            n,
            context,
            gather_indices=gather_indices,
            output=self._sampled_tokens_buffer[:n],
        )
        # CudaGraphManager expects an iterable output, even when no tensor is returned.
        return ()

    def sample_speculative(
        self,
        required_logits: Tensor,
        num_decode: int,
        num_prefill: int,
        num_speculative_tokens: int,
        context,
        *,
        gather_indices: Optional[Tensor] = None,
        eager: bool = False,
        cache_key: Any = None,
    ) -> Tensor:
        """Sample tokens for the speculative-verify path.

        Decode requests contribute `1 + num_speculative_tokens` rows; prefill requests contribute 1.
        Builds the per-token request mapping and dispatches to the return-valued `sample_kernel`.

        When `gather_indices` is supplied, the kernel selects via `logits[gather_indices[:n], :]`.
        When `gather_indices` is None, `required_logits` is expected to be already pre-gathered to
        the layout described above (e.g. when `materialize_only_last_token_logits=True` upstream).

        Args:
            required_logits: Logits containing base and speculative rows.
            num_decode: Number of decode requests.
            num_prefill: Number of prefill requests.
            num_speculative_tokens: Number of draft tokens per decode request.
            context: The active DynamicInferenceContext.
            gather_indices: Optional rows to gather from `required_logits`.
            eager: Whether to bypass a wrapped CUDA graph.
            cache_key: CUDA graph lookup key.

        Returns:
            Sampled token IDs for all required base and speculative rows.
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key

        n_spec = num_speculative_tokens
        num_decode_tokens = num_decode * (1 + n_spec)
        num_tokens = num_decode_tokens + num_prefill
        device = required_logits.device

        token_to_request_index = torch.cat(
            [
                torch.arange(num_decode, device=device).repeat_interleave(
                    1 + n_spec, output_size=num_decode_tokens
                ),
                torch.arange(num_decode, num_decode + num_prefill, device=device),
            ]
        )
        return self.sample_kernel(
            required_logits,
            num_tokens,
            context,
            gather_indices=gather_indices,
            token_to_request_index=token_to_request_index,
        )

    @abstractmethod
    def log_probs_kernel(
        self, logits: Tensor, context, *, token_to_request_index: Optional[Tensor] = None
    ) -> Tensor:
        """Per-row log-probs of the distribution this backend samples from.

        Args:
            logits: `[num_rows, vocab_size]` raw logits.
            context: The active DynamicInferenceContext.
            token_to_request_index: Optional per-row request mapping. When
                omitted, each logits row maps to the request at the same index.

        Returns:
            `[num_rows, vocab_size]` log-probs; filtered-out tokens are `-inf`.
        """
        ...
