# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.inference.async_txn import (
    AsyncTxnDiagnostics,
    AsyncTxnSkipReason,
    StepTxn,
    TxnRetireQueue,
    classify_decode_child_launch,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)


class FakeEvent:
    def __init__(self, done: bool = False):
        self.done = done

    def query(self) -> bool:
        return self.done


class FakeKVAllocator:
    def __init__(self):
        self.released = []

    def release_memory_blocks(self, block_ids: torch.Tensor) -> None:
        self.released.extend(int(block_id) for block_id in block_ids.tolist())


class FakeMambaMetadata:
    def __init__(self):
        self.request_to_mamba_state_idx = torch.tensor([5, 7, -1], dtype=torch.int32)
        self.request_to_mamba_state_bank = torch.tensor([0, 0, 0], dtype=torch.int32)
        self.freed_slot_ids = []
        self.free_slot_request_indices = []
        self.reset_called = False
        self.reset_varlen_called = False

    def free_slot_ids(self, slot_ids: torch.Tensor) -> None:
        self.freed_slot_ids.extend(int(slot_id) for slot_id in slot_ids.tolist() if slot_id != -1)

    def free_slots(self, request_indices: torch.Tensor) -> None:
        self.free_slot_request_indices.extend(int(idx) for idx in request_indices.tolist())
        self.free_slot_ids(self.request_to_mamba_state_idx[request_indices])
        self.request_to_mamba_state_idx[request_indices] = -1
        self.request_to_mamba_state_bank[request_indices] = 0

    def reset(self) -> None:
        self.reset_called = True
        self.request_to_mamba_state_idx.fill_(-1)
        self.request_to_mamba_state_bank.zero_()

    def reset_varlen_metadata(self) -> None:
        self.reset_varlen_called = True


class FakeLaunchContext:
    def __init__(self, *, paused_request_count=0):
        self.total_request_count = 2
        self.paused_request_count = paused_request_count
        self.is_hybrid_model = True
        self.chunked_prefill_request_id = -1
        self.num_speculative_tokens = 0
        self.block_size_tokens = 4
        self.request_last_kv_block_offset = torch.tensor([0, 0], dtype=torch.int32)
        self.request_ids = torch.tensor([101, 102], dtype=torch.int32)
        self.kv_block_allocator = SimpleNamespace(is_memory_available=lambda count: True)

    def is_decode_only(self) -> bool:
        return True

    def using_cuda_graph_this_step(self) -> bool:
        return False


def _make_release_context() -> DynamicInferenceContext:
    context = object.__new__(DynamicInferenceContext)
    context.is_hybrid_model = True
    context.mamba_metadata = FakeMambaMetadata()
    context.mamba_slot_allocator = None
    context.kv_block_allocator = FakeKVAllocator()
    context.async_txn_diagnostics = AsyncTxnDiagnostics(enabled=True)
    context.async_txn_retire_queue = TxnRetireQueue(context.async_txn_diagnostics)
    context.request_to_kv_block_ids = torch.tensor(
        [[10, -1], [20, -1], [-1, -1]], dtype=torch.int32
    )
    return context


def _make_controller(context):
    controller = object.__new__(TextGenerationController)
    controller.num_speculative_tokens = 0
    controller._enable_cuda_graph = False
    controller._get_stop_word_finished_ids_callback = None
    controller.model_config = SimpleNamespace(
        expert_model_parallel_size=1,
        num_moe_experts=None,
    )
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    return controller


def test_finished_hybrid_row_mamba_slot_is_not_reused_before_retire():
    context = _make_release_context()
    event = FakeEvent(done=False)

    DynamicInferenceContext.release_memory_blocks_from_request_indexes(
        context,
        torch.tensor([0], dtype=torch.int64),
        retire_event=event,
        defer_release=True,
    )

    assert context.kv_block_allocator.released == []
    assert context.mamba_metadata.freed_slot_ids == []
    assert context.mamba_metadata.request_to_mamba_state_idx.tolist() == [-1, 7, -1]

    event.done = True
    assert context.async_txn_retire_queue.drain_ready() == 2
    assert context.kv_block_allocator.released == [10]
    assert context.mamba_metadata.freed_slot_ids == [5]


def test_immediate_hybrid_release_uses_existing_logical_slot_free_path():
    context = _make_release_context()

    DynamicInferenceContext.release_memory_blocks_from_request_indexes(
        context, torch.tensor([1], dtype=torch.int64)
    )

    assert context.kv_block_allocator.released == [20]
    assert context.mamba_metadata.free_slot_request_indices == [1]
    assert context.mamba_metadata.freed_slot_ids == [7]


def test_all_finished_async_hybrid_commit_does_not_double_free_mamba_slot():
    context = _make_release_context()
    context.num_speculative_tokens = 0
    context.num_prefill_requests = 0
    context.paused_request_count = 0
    context.total_request_count = 1
    context.active_token_count = 1
    context.chunked_prefill_request_id = -1
    context.request_in_prefill_status_tensor = torch.zeros(3, dtype=torch.int32)
    context.request_to_kv_block_ids = torch.tensor(
        [[10, -1], [-1, -1], [-1, -1]], dtype=torch.int32
    )
    context.mamba_metadata.request_to_mamba_state_idx = torch.tensor(
        [5, -1, -1], dtype=torch.int32
    )
    context.mamba_metadata.request_to_mamba_state_bank = torch.zeros(3, dtype=torch.int32)
    context.reset_attention_state = lambda: None
    context.get_index_of_chunked_prefill_request = lambda safe=True: -1
    event = FakeEvent(done=False)
    txn = StepTxn(step_id=4, request_ids=[101], forward_done_event=event, launched=True)

    DynamicInferenceContext.update_requests(
        context, torch.tensor([0], dtype=torch.uint8), torch.tensor([7]), async_txn=txn
    )

    assert context.mamba_metadata.reset_called is False
    assert context.mamba_metadata.reset_varlen_called is True
    assert context.mamba_metadata.freed_slot_ids == []
    assert context.mamba_metadata.request_to_mamba_state_idx.tolist() == [-1, -1, -1]

    event.done = True
    assert context.async_txn_retire_queue.drain_ready() == 2
    assert context.kv_block_allocator.released == [10]
    assert context.mamba_metadata.freed_slot_ids == [5]


def test_hybrid_child_launch_is_not_rejected_when_no_mutation_gate_applies():
    context = FakeLaunchContext()
    controller = _make_controller(context)
    child_txn = StepTxn(step_id=3, request_ids=[101, 102], mamba_slot_ids=(5, 7))

    reason = controller._async_child_launch_skip_reason(
        child_txn,
        return_log_probs=False,
        return_top_n_logprobs=False,
        skip_bookkeeping=False,
    )

    assert reason is None


def test_hybrid_pause_pressure_still_forces_sync_before_child_launch():
    context = FakeLaunchContext(paused_request_count=1)

    eligibility = classify_decode_child_launch(context, async_enabled=True)

    assert not eligibility.eligible
    assert eligibility.reason == AsyncTxnSkipReason.PAUSED_REQUESTS


def test_step_txn_records_logical_mamba_slots_without_bank_internals():
    txn = StepTxn(step_id=3, request_ids=[101, 102], mamba_slot_ids=(5, 7))

    assert txn.mamba_slot_ids == (5, 7)
    assert not hasattr(txn, "candidate_mamba_slot_ids")


def test_async_mamba_indices_read_current_and_write_next_bank():
    context = object.__new__(DynamicInferenceContext)
    context.is_hybrid_model = True
    context.mamba_state_bank_count = 2
    context.paused_request_count = 1
    context.total_request_count = 4
    context.request_ids = torch.tensor([900, 101, 102, 103], dtype=torch.int32)
    context.mamba_metadata = SimpleNamespace(
        request_to_mamba_state_idx=torch.tensor([90, 5, 7, 9], dtype=torch.int32),
        request_to_mamba_state_bank=torch.tensor([0, 0, 1, 0], dtype=torch.int32),
    )

    assert DynamicInferenceContext._mamba_flat_indices(
        context, slice(context.paused_request_count, context.total_request_count)
    ).tolist() == [10, 15, 18]
    assert DynamicInferenceContext._mamba_flat_indices(
        context,
        slice(context.paused_request_count, context.total_request_count),
        use_next_bank=True,
    ).tolist() == [11, 14, 19]
    assert context.mamba_metadata.request_to_mamba_state_bank.tolist() == [0, 0, 1, 0]


def test_consumed_async_mamba_forward_advances_bank_without_accept_reject_contract():
    context = object.__new__(DynamicInferenceContext)
    context.is_hybrid_model = True
    context.mamba_state_bank_count = 2
    context.paused_request_count = 1
    context.total_request_count = 4
    context.request_ids = torch.tensor([900, 101, 102, 103], dtype=torch.int32)
    context.mamba_metadata = SimpleNamespace(
        request_to_mamba_state_idx=torch.tensor([90, 5, 7, 9], dtype=torch.int32),
        request_to_mamba_state_bank=torch.tensor([0, 0, 1, 0], dtype=torch.int32),
    )

    DynamicInferenceContext.advance_async_mamba_state(context, [101, 103])

    assert context.mamba_metadata.request_to_mamba_state_bank.tolist() == [0, 1, 1, 1]


def _install_minimal_child_bookkeeping(context):
    context.max_tokens = 4
    context.max_requests = 4
    context.max_kv_block_count = 2
    total_bytes = (
        context.max_tokens * 8 * 2
        + context.max_tokens * 4 * 4
        + context.max_requests * 4 * 7
        + context.max_requests * 4
        + (context.max_requests + 1) * 4
        + context.max_requests * 4
        + (context.max_requests + 1) * 4
        + context.max_requests * context.max_kv_block_count * 4
        + context.max_requests * 4 * 2
    )
    context._cpu_bookkeeping_buf = torch.zeros(total_bytes, dtype=torch.uint8)
    offset = 0

    def take(numel, dtype, shape=None):
        nonlocal offset
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
        view = context._cpu_bookkeeping_buf[offset : offset + numel * bytes_per_elem].view(dtype)
        offset += numel * bytes_per_elem
        if shape is not None:
            view = view.view(shape)
        return view

    context.token_to_input_ids = take(context.max_tokens, torch.int64)
    context.token_to_pos_ids = take(context.max_tokens, torch.int64)
    context.token_to_block_idx = take(context.max_tokens, torch.int32)
    context.token_to_local_position_within_kv_block = take(context.max_tokens, torch.int32)
    context.token_to_request_idx = take(context.max_tokens, torch.int32)
    context.token_to_position_in_request = take(context.max_tokens, torch.int32)
    context._staging_request_in_prefill_status = take(context.max_requests, torch.int32)
    context._staging_request_query_lengths = take(context.max_requests, torch.int32)
    context._staging_request_kv_length_offsets = take(context.max_requests, torch.int32)
    context._staging_temperature = take(context.max_requests, torch.float32)
    context._staging_top_k = take(context.max_requests, torch.int32)
    context._staging_top_p = take(context.max_requests, torch.float32)
    context.active_request_last_token_idxs = take(context.max_requests, torch.int32)
    context._cpu_mha_query_lengths = take(context.max_requests, torch.int32)
    context._cpu_mha_cu_query_seq_lengths = take(context.max_requests + 1, torch.int32)
    context._cpu_mha_kv_seq_lengths = take(context.max_requests, torch.int32)
    context._cpu_mha_cu_kv_seq_lengths = take(context.max_requests + 1, torch.int32)
    context._cpu_mha_block_table = take(
        context.max_requests * context.max_kv_block_count,
        torch.int32,
        (context.max_requests, context.max_kv_block_count),
    )
    context._cpu_mamba_batch_indices_decode = take(context.max_requests, torch.int32)
    context._cpu_mamba_batch_indices_decode_write = take(context.max_requests, torch.int32)


def test_source_child_mamba_banks_follow_parent_transaction_snapshot():
    context = object.__new__(DynamicInferenceContext)
    _install_minimal_child_bookkeeping(context)
    context.is_hybrid_model = True
    context.mamba_state_bank_count = 2
    context.block_size_tokens = 16
    context.paused_request_count = 0
    context.total_request_count = 2
    context.padded_active_token_count = 2
    context.padded_active_request_count = 2
    context.batch_dimensions = SimpleNamespace(req_count=2)
    context.padded_batch_dimensions = SimpleNamespace(req_count=2)
    context.kv_block_allocator = SimpleNamespace(dummy_block_idx=-1)
    context.request_ids = torch.tensor([101, 102], dtype=torch.int32)
    context.mamba_metadata = SimpleNamespace(
        request_to_mamba_state_idx=torch.tensor([5, 7], dtype=torch.int32),
        request_to_mamba_state_bank=torch.tensor([0, 1], dtype=torch.int32),
    )

    source_buf = torch.zeros_like(context._cpu_bookkeeping_buf)
    source_pos = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context.token_to_pos_ids, source_buf
    )
    source_block = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context.token_to_block_idx, source_buf
    )
    source_req = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context.token_to_request_idx, source_buf
    )
    source_pos_in_req = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context.token_to_position_in_request, source_buf
    )
    source_local = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context.token_to_local_position_within_kv_block, source_buf
    )
    source_query = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context._staging_request_query_lengths, source_buf
    )
    source_kv = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context._staging_request_kv_length_offsets, source_buf
    )
    source_temp = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context._staging_temperature, source_buf
    )
    source_top_k = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context._staging_top_k, source_buf
    )
    source_top_p = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context._staging_top_p, source_buf
    )
    source_last = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context.active_request_last_token_idxs, source_buf
    )
    source_mha_query = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context._cpu_mha_query_lengths, source_buf
    )
    source_mha_cu_query = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context._cpu_mha_cu_query_seq_lengths, source_buf
    )
    source_mha_kv = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context._cpu_mha_kv_seq_lengths, source_buf
    )
    source_mha_block = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context._cpu_mha_block_table, source_buf
    )
    source_mamba_read = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context._cpu_mamba_batch_indices_decode, source_buf
    )
    source_mamba_write = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context._cpu_mamba_batch_indices_decode_write, source_buf
    )

    source_pos[:2] = torch.tensor([4, 7], dtype=torch.int64)
    source_block[:2] = torch.tensor([1, 2], dtype=torch.int32)
    source_req[:2] = torch.tensor([0, 1], dtype=torch.int32)
    source_pos_in_req[:2] = torch.tensor([4, 7], dtype=torch.int32)
    source_local[:2] = torch.tensor([4, 7], dtype=torch.int32)
    source_query[:2] = 1
    source_kv[:2] = torch.tensor([4, 7], dtype=torch.int32)
    source_temp[:2] = 1.0
    source_top_k[:2] = 1
    source_top_p[:2] = 0.0
    source_last[:2] = 0
    source_mha_query[:2] = 1
    source_mha_cu_query[:3] = torch.tensor([0, 1, 2], dtype=torch.int32)
    source_mha_kv[:2] = torch.tensor([5, 8], dtype=torch.int32)
    source_mha_block[:2] = torch.tensor([[11, -1], [12, -1]], dtype=torch.int32)
    source_mamba_read[:2] = torch.tensor([10, 15], dtype=torch.int32)
    source_mamba_write[:2] = torch.tensor([11, 14], dtype=torch.int32)

    child_buf = torch.zeros_like(context._cpu_bookkeeping_buf)
    DynamicInferenceContext._build_plain_decode_child_bookkeeping(
        context,
        child_buf,
        2,
        source_cpu_buf=source_buf,
        source_request_ids=(101, 102),
    )

    child_mamba_read = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context._cpu_mamba_batch_indices_decode, child_buf
    )
    child_mamba_write = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context._cpu_mamba_batch_indices_decode_write, child_buf
    )
    assert child_mamba_read[:2].tolist() == [11, 14]
    assert child_mamba_write[:2].tolist() == [10, 15]
