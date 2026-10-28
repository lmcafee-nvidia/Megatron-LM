# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scratch probes comparing two MXFP8 scale-layout fixes. Not intended for landing.

Every probe uses only APIs that exist on main, so the same file runs unchanged
against main's production code and against either candidate fix.
"""

import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor, validate_mxfp8_tensor
from megatron.core.resharding.refit import (
    clear_all_caches,
    prepare_swap_model_weights,
    swap_model_weights,
)
from megatron.core.resharding.transforms import MXFP8ReshardTransform
from tests.unit_tests.test_utilities import Utils

_IS_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 10

try:
    from flashinfer import mxfp8_quantize as _fi_quantize

    _HAVE_FLASHINFER = True
except ImportError:
    _HAVE_FLASHINFER = False

pytestmark = pytest.mark.skipif(not _IS_BLACKWELL, reason="MXFP8 probes require Blackwell")

BACKENDS = ("triton", "flashinfer")
SHAPES = ((1, 32), (64, 128), (128, 128), (129, 160), (127, 96), (257, 352), (256, 4096))
WHOLE = (slice(None), slice(None))
NAME = "decoder.weight"


def _need(backend):
    if backend == "flashinfer" and not _HAVE_FLASHINFER:
        pytest.skip("FlashInfer unavailable")


def _scale_dtype(backend):
    return torch.uint8 if backend == "flashinfer" else torch.float8_e8m0fnu


def _decode(tensor):
    """Independently decode cuBLAS 128x4 swizzled scales to logical [M, K // 32] bytes."""
    rows, cols = tensor.data.shape
    scale_cols = cols // 32
    assert tensor.scale.ndim == 1
    raw = tensor.scale.reshape(-1).view(torch.uint8)
    padded_cols = (scale_cols + 3) // 4 * 4
    padded_rows = (rows + 127) // 128 * 128
    assert raw.numel() == padded_rows * padded_cols
    row = torch.arange(rows, device=raw.device)[:, None]
    col = torch.arange(scale_cols, device=raw.device)[None, :]
    offset = (
        ((row // 128) * (padded_cols // 4) + col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + col % 4
    )
    return raw[offset]


def _values(rows, cols, seed, spread=20):
    """BF16 values whose 32-element blocks span many E8M0 exponents."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn(rows, cols, device="cuda", generator=gen)
    exps = torch.randint(-spread, spread + 1, (rows, cols // 32, 1), device="cuda", generator=gen)
    values = (base.view(rows, cols // 32, 32) * torch.exp2(exps.float())).view(rows, cols)
    if rows * cols > 32:
        values[0, :32] = 0
    return values.to(torch.bfloat16)


def _reference(values, backend):
    """Default (swizzled) quantization of the full tensor, decoded to logical bytes."""
    ref = MXFP8Tensor.from_bf16(values.contiguous(), backend=backend)
    assert ref.scale.ndim == 1, "default from_bf16 must keep the swizzled layout"
    return ref.data.view(torch.uint8), _decode(ref)


def _logical(values, backend):
    """A logical 2D-scale MXFP8 tensor built from main's default quantizer."""
    ref = MXFP8Tensor.from_bf16(values.contiguous(), backend=backend)
    return MXFP8Tensor(
        data=ref.data.clone(),
        scale=_decode(ref).clone().view(_scale_dtype(backend)),
        backend=backend,
        dtype=values.dtype,
    )


def _poisoned_2d(rows, cols, backend):
    return MXFP8Tensor(
        data=torch.full((rows, cols), 0xA5, device="cuda", dtype=torch.uint8).view(
            torch.float8_e4m3fn
        ),
        scale=torch.full((rows, cols // 32), 0xA5, device="cuda", dtype=torch.uint8).view(
            _scale_dtype(backend)
        ),
        backend=backend,
        dtype=torch.bfloat16,
    )


def _pointers(tensor):
    return tensor.data.data_ptr(), tensor.scale.data_ptr()


def _assert_exact(buf, values, backend):
    data, scale = _reference(values, backend)
    assert torch.equal(buf.data.view(torch.uint8), data), "data bytes differ"
    assert torch.equal(buf.scale.view(torch.uint8), scale), "logical scale bytes differ"


def _receiver(buf, convert_on_send=False):
    return MXFP8ReshardTransform(
        convertible_params={NAME},
        persistent_buffers={"weight": buf},
        buffer_key_prefix="decoder.",
        convert_on_send=convert_on_send,
    )


def _receive_bf16(transform, values, region):
    received = transform.prepare_recv(NAME, region)
    received[0].copy_(values[region])
    transform.finalize_recv(NAME, region, received)


# ---------------------------------------------------------------------------
# FlashInfer's own layouts: is native linear output the decoded swizzle?
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_flashinfer_linear_layout_equals_decoded_swizzle(shape):
    _need("flashinfer")
    rows, cols = shape
    values = _values(rows, cols, 7)
    data_s, scale_s = _fi_quantize(values)
    data_l, scale_l = _fi_quantize(values, is_sf_swizzled_layout=False)
    swizzled = MXFP8Tensor(data=data_s, scale=scale_s.view(torch.uint8), backend="flashinfer")
    assert torch.equal(data_s.view(torch.uint8), data_l.view(torch.uint8))
    assert scale_l.numel() == rows * cols // 32
    assert torch.equal(scale_l.view(torch.uint8).view(rows, cols // 32), _decode(swizzled))


# ---------------------------------------------------------------------------
# Receiver-side conversion into logical 2D storage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_receiver_2d_row_slices(backend, shape):
    _need(backend)
    rows, cols = shape
    buf = _poisoned_2d(rows, cols, backend)
    pointers = _pointers(buf)
    transform = _receiver(buf)
    values = _values(rows, cols, 11)
    split = rows // 2 + 1 if rows > 1 else rows
    _receive_bf16(transform, values, (slice(0, split), slice(None)))
    if split < rows:
        assert torch.all(buf.data[split:].view(torch.uint8) == 0xA5)
        assert torch.all(buf.scale[split:].view(torch.uint8) == 0xA5)
        _receive_bf16(transform, values, (slice(split, rows), slice(None)))
    _assert_exact(buf, values, backend)
    assert _pointers(buf) == pointers


@pytest.mark.parametrize("backend", BACKENDS)
def test_receiver_2d_column_slices(backend):
    _need(backend)
    rows, cols = 129, 160
    buf = _poisoned_2d(rows, cols, backend)
    pointers = _pointers(buf)
    transform = _receiver(buf)
    values = _values(rows, cols, 12)
    _receive_bf16(transform, values, (slice(None), slice(0, 64)))
    assert torch.all(buf.scale[:, 2:].view(torch.uint8) == 0xA5)
    _receive_bf16(transform, values, (slice(None), slice(64, 160)))
    _assert_exact(buf, values, backend)
    assert _pointers(buf) == pointers


# ---------------------------------------------------------------------------
# In-place requantization (MXFP8Tensor.copy_ / quantize_)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_inplace_requantize_2d(backend, shape):
    _need(backend)
    rows, cols = shape
    buf = _poisoned_2d(rows, cols, backend)
    pointers = _pointers(buf)
    values = _values(rows, cols, 21)
    buf.copy_(values)
    _assert_exact(buf, values, backend)
    assert _pointers(buf) == pointers
    second = _values(rows, cols, 22)
    buf.quantize_(second.float())
    _assert_exact(buf, second, backend)
    assert _pointers(buf) == pointers


# ---------------------------------------------------------------------------
# Default swizzled callers must be unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_swizzled_1d_paths_unchanged(backend, shape):
    _need(backend)
    rows, cols = shape
    old, new = _values(rows, cols, 3), _values(rows, cols, 5)
    buf = MXFP8Tensor.from_bf16(old, backend=backend)
    pointers, shape_before = _pointers(buf), tuple(buf.scale.shape)
    buf.copy_(new)
    ref = MXFP8Tensor.from_bf16(new, backend=backend)
    assert tuple(buf.scale.shape) == shape_before == tuple(ref.scale.shape)
    assert torch.equal(buf.data.view(torch.uint8), ref.data.view(torch.uint8))
    assert torch.equal(_decode(buf), _decode(ref))
    if backend == "flashinfer":
        assert torch.equal(buf.scale.view(torch.uint8), ref.scale.view(torch.uint8))
    assert _pointers(buf) == pointers

    transform = _receiver(buf)
    split = rows // 2 + 1 if rows > 1 else rows
    regions = [(slice(0, split), slice(None))]
    if split < rows:
        regions.append((slice(split, rows), slice(None)))
    for region in regions:
        _receive_bf16(transform, old, region)
    ref = MXFP8Tensor.from_bf16(old, backend=backend)
    assert torch.equal(buf.data.view(torch.uint8), ref.data.view(torch.uint8))
    assert torch.equal(_decode(buf), _decode(ref))
    assert _pointers(buf) == pointers


# ---------------------------------------------------------------------------
# Sender-side conversion (convert_on_send=True) into a logical 2D receiver
# ---------------------------------------------------------------------------


def _send_and_receive(backend, values, region):
    rows, cols = values.shape
    sender = MXFP8ReshardTransform(
        convertible_params={NAME}, persistent_buffers={}, convert_on_send=True, backend=backend
    )
    dest = _poisoned_2d(rows, cols, backend)
    receiver = _receiver(dest, convert_on_send=True)
    source = torch.nn.Parameter(values, requires_grad=False)
    sent = sender.prepare_send(NAME, region, source)
    received = receiver.prepare_recv(NAME, region)
    assert [t.numel() for t in sent] == [t.numel() for t in received], (
        "wire element counts differ "
        f"({[tuple(t.shape) for t in sent]} vs {[tuple(t.shape) for t in received]}); "
        "a real point-to-point transfer cannot complete"
    )
    # Point-to-point transports move raw elements and ignore shape.
    for src, dst in zip(sent, received):
        dst.reshape(-1).view(torch.uint8).copy_(src.reshape(-1).view(torch.uint8))
    receiver.finalize_recv(NAME, region, received)
    return dest


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_sender_conversion_whole_into_2d_receiver(backend, shape):
    _need(backend)
    rows, cols = shape
    values = _values(rows, cols, 31)
    dest = _send_and_receive(backend, values, WHOLE)
    _assert_exact(dest, values, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sender_conversion_slices_into_2d_receiver(backend):
    _need(backend)
    values = _values(256, 4096, 32)
    for region in ((slice(0, 128), slice(None)), (slice(None), slice(1024, 2048))):
        dest = _send_and_receive(backend, values, region)
        data, scale = _reference(values, backend)
        scale_region = (region[0], slice((region[1].start or 0) // 32, (region[1].stop or 4096) // 32))
        assert torch.equal(dest.data[region].view(torch.uint8), data[region])
        assert torch.equal(dest.scale[scale_region].view(torch.uint8), scale[scale_region])


@pytest.mark.parametrize("shape", ((128, 128), (256, 4096)))
def test_sender_conversion_real_nccl_pairs(shape, request):
    """Equal wire element counts let NCCL complete, so the receiver sees the wire bytes."""
    _need("flashinfer")
    Utils.initialize_model_parallel(1, 1)
    request.addfinalizer(Utils.destroy_model_parallel)
    world, rank = dist.get_world_size(), dist.get_rank()
    if world < 2:
        pytest.skip("needs two ranks")
    rows, cols = shape
    values = _values(rows, cols, 41)
    if rank >= world // 2 * 2:
        return
    if rank % 2 == 0:
        sender = MXFP8ReshardTransform(
            convertible_params={NAME},
            persistent_buffers={},
            convert_on_send=True,
            backend="flashinfer",
        )
        for tensor in sender.prepare_send(NAME, WHOLE, torch.nn.Parameter(values, False)):
            dist.send(tensor.contiguous().view(torch.uint8), rank + 1)
        torch.cuda.synchronize()
        return
    dest = _poisoned_2d(rows, cols, "flashinfer")
    pointers = _pointers(dest)
    receiver = _receiver(dest, convert_on_send=True)
    received = receiver.prepare_recv(NAME, WHOLE)
    for tensor in received:
        dist.recv(tensor.view(torch.uint8), rank - 1)
    receiver.finalize_recv(NAME, WHOLE, received)
    torch.cuda.synchronize()
    _assert_exact(dest, values, "flashinfer")
    assert _pointers(dest) == pointers


# ---------------------------------------------------------------------------
# GEMM consumers must not silently read logical bytes as swizzled storage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_gemm_never_silently_consumes_logical_scales(backend):
    _need(backend)
    from megatron.core.inference.quantization.utils import mm_mxfp8

    rows, cols = 128, 128
    weight_values = _values(rows, cols, 51, spread=4)
    weight = MXFP8Tensor.from_bf16(weight_values, backend=backend)
    logical = _logical(weight_values, backend)
    assert not torch.equal(_decode(weight).reshape(-1), weight.scale.view(torch.uint8))
    x = torch.randn(8, 1, cols, device="cuda", dtype=torch.bfloat16)
    expected = mm_mxfp8(x, weight)
    try:
        actual = mm_mxfp8(x, logical)
        torch.cuda.synchronize()
    except (ValueError, RuntimeError, AssertionError) as exc:
        print(f"PROBE gemm-logical {backend}: rejected with {type(exc).__name__}: {exc}")
        return
    assert torch.equal(actual, expected), (
        "GEMM silently consumed logical 2D scales as swizzled storage: "
        f"max |diff| {(actual.float() - expected.float()).abs().max().item()}"
    )


def test_validation_cost_per_call():
    """Report the host cost of validate_mxfp8_tensor, the check A adds to every mm_mxfp8."""
    weight = MXFP8Tensor.from_bf16(
        torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16), backend="triton"
    )
    for _ in range(100):
        validate_mxfp8_tensor(weight, tensor_name="w")
    count = 20000
    start = time.perf_counter()
    for _ in range(count):
        validate_mxfp8_tensor(weight, tensor_name="w")
    elapsed = (time.perf_counter() - start) / count
    print(f"PROBE validate_mxfp8_tensor host cost: {elapsed * 1e6:.2f} us/call")


# ---------------------------------------------------------------------------
# API-neutral port of B's public NCCL refit + CUDA-graph regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_public_refit_logical_scales_preserve_graph_storage(backend, request):
    _need(backend)
    Utils.initialize_model_parallel(1, 1)
    request.addfinalizer(Utils.destroy_model_parallel)

    class TinyModel(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.decoder = torch.nn.Linear(
                160, 129, bias=False, device="cuda", dtype=torch.bfloat16
            )
            group = dist.group.WORLD
            self.pg_collection = SimpleNamespace(
                tp=group,
                pp=group,
                ep=group,
                dp=group,
                expt_tp=group,
                gtp_remat=None,
                expt_gtp_remat=None,
            )
            self.config = SimpleNamespace(num_moe_experts=None)

        def forward(self, inputs):
            weight = self.decoder.weight
            scale_bits = weight.scale.view(torch.uint8).to(torch.int32) << 23
            scales = scale_bits.view(torch.float32).repeat_interleave(32, dim=1)
            dequantized = weight.data.float() * scales
            return torch.matmul(inputs.float(), dequantized.t()).to(torch.bfloat16)

    source = TinyModel(111)
    target = TinyModel(222)
    target_buffer = _logical(target.decoder.weight.detach(), backend)
    clear_all_caches()
    prepare_swap_model_weights([source], [target])
    del target.decoder._parameters["weight"]
    target.decoder.weight = target_buffer
    transform = _receiver(target_buffer)
    source_weight = source.decoder.weight.detach().clone()
    reference = _logical(source_weight, backend)
    reference_model = TinyModel(333)
    del reference_model.decoder._parameters["weight"]
    reference_model.decoder.weight = reference
    inputs = torch.linspace(-1, 1, 320, device="cuda", dtype=torch.bfloat16).reshape(2, 1, 160)
    expected = reference_model(inputs)
    pointers = _pointers(target_buffer)

    target(inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = target(inputs)
    swap_model_weights([source], [target], refit_method="nccl", transform=transform)
    torch.cuda.synchronize()

    assert _pointers(target_buffer) == pointers
    assert torch.equal(target_buffer.data.view(torch.uint8), reference.data.view(torch.uint8))
    assert torch.equal(target_buffer.scale.view(torch.uint8), reference.scale.view(torch.uint8))
    graph.replay()
    torch.testing.assert_close(graph_output.clone(), expected, atol=0, rtol=0)
    clear_all_caches()
