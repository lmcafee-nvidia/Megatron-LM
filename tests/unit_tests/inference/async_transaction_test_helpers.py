# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch


@dataclass
class FakeCudaEvent:
    """Small event stand-in for async transaction unit tests."""

    name: str
    ledger: list[tuple[str, str]] = field(default_factory=list)
    recorded: bool = False
    waited: bool = False
    synchronized: bool = False

    def record(self, stream: object | None = None) -> None:
        self.recorded = True
        self.ledger.append((self.name, "record"))

    def wait(self, stream: object | None = None) -> None:
        self.waited = True
        self.ledger.append((self.name, "wait"))

    def synchronize(self) -> None:
        self.synchronized = True
        self.ledger.append((self.name, "synchronize"))


@dataclass
class FakeParticipant:
    """Participant probe that records lifecycle hook order."""

    name: str
    ledger: list[tuple[str, str]]
    prepared: int = 0
    accepted: int = 0
    rolled_back: int = 0
    retired: int = 0

    def prepare(self, plan: object) -> dict[str, str]:
        self.prepared += 1
        self.ledger.append((self.name, "prepare"))
        return {"name": self.name}

    def validate(self, plan: object, current_state: object) -> bool:
        self.ledger.append((self.name, "validate"))
        return True

    def commit(self, plan: object) -> None:
        self.accepted += 1
        self.ledger.append((self.name, "accept"))

    def rollback(self, plan: object) -> None:
        self.rolled_back += 1
        self.ledger.append((self.name, "rollback"))

    def retire(self, plan: object | None = None) -> None:
        self.retired += 1
        self.ledger.append((self.name, "retire"))

    def diagnostics(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "prepared": self.prepared,
            "accepted": self.accepted,
            "rolled_back": self.rolled_back,
            "retired": self.retired,
        }


@dataclass
class OverlapOrderProbe:
    """Records the relative ordering of launch and CPU-only work."""

    events: list[str] = field(default_factory=list)

    def hard_commit(self) -> None:
        self.events.append("hard_commit")

    def launch(self) -> None:
        self.events.append("launch")

    def post_launch_cpu(self) -> None:
        self.events.append("post_launch_cpu")

    def assert_launch_before_cpu(self) -> None:
        assert self.events.index("launch") < self.events.index("post_launch_cpu")


def fake_committed_decode_context(
    request_ids: list[int] | torch.Tensor,
    *,
    query_length: int = 1,
    kv_offset_start: int = 4,
    tokens_per_request: int = 1,
    padded_active_request_count: int | None = None,
    mamba_slots: list[int] | torch.Tensor | None = None,
    decode_only: bool = True,
    using_cuda_graph: bool = True,
) -> SimpleNamespace:
    """Build a committed decode-only context stub for async plan tests."""

    request_ids_tensor = torch.as_tensor(request_ids, dtype=torch.int32, device="cpu")
    active_request_count = int(request_ids_tensor.numel())
    if padded_active_request_count is None:
        padded_active_request_count = active_request_count
    token_count = active_request_count * tokens_per_request
    token_rows = torch.arange(active_request_count, dtype=torch.int32, device="cpu")
    token_to_request_idx = token_rows.repeat_interleave(tokens_per_request)
    token_offsets = torch.arange(tokens_per_request, dtype=torch.int64, device="cpu").repeat(
        active_request_count
    )
    kv_offsets = torch.arange(
        kv_offset_start,
        kv_offset_start + active_request_count,
        dtype=torch.int64,
        device="cpu",
    )
    token_to_pos_ids = kv_offsets.repeat_interleave(tokens_per_request) + token_offsets
    mamba_slots_tensor = (
        None
        if mamba_slots is None
        else torch.as_tensor(mamba_slots, dtype=torch.int32, device="cpu")
    )
    return SimpleNamespace(
        request_ids=request_ids_tensor,
        paused_request_count=0,
        total_request_count=active_request_count,
        active_token_count=token_count,
        padded_active_request_count=padded_active_request_count,
        request_query_lengths=torch.full(
            (active_request_count,), query_length, dtype=torch.int32, device="cpu"
        ),
        request_kv_length_offsets=kv_offsets.to(dtype=torch.int32),
        token_to_request_idx=token_to_request_idx,
        token_to_pos_ids=token_to_pos_ids,
        token_to_position_in_request=token_to_pos_ids.clone(),
        token_to_block_idx=torch.arange(token_count, dtype=torch.int32, device="cpu"),
        token_to_local_position_within_kv_block=token_to_pos_ids.to(dtype=torch.int32),
        mamba_slot_ids=mamba_slots_tensor,
        is_decode_only=lambda: decode_only,
        using_cuda_graph_this_step=lambda: using_cuda_graph,
    )


def assert_exact_plan_identity(left: Any, right: Any) -> None:
    """Assert the fields that define committed async launch identity."""

    assert torch.equal(left.request_ids.to(device="cpu"), right.request_ids.to(device="cpu"))
    assert torch.equal(left.row_order.to(device="cpu"), right.row_order.to(device="cpu"))
    assert left.active_decode_count == right.active_decode_count
    assert left.cuda_graph_shape == right.cuda_graph_shape
    assert left.decode_stride == right.decode_stride
    assert torch.equal(left.mamba_slot_ids.to(device="cpu"), right.mamba_slot_ids.to(device="cpu"))
    assert left.identity_fingerprint == right.identity_fingerprint
