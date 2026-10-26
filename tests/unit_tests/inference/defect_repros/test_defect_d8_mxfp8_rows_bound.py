# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for defect d8: ``mxfp8_quantize_rows`` does not bound the device-side row count.

``_mxfp8_quant_swizzle_rows_kernel`` (``megatron/core/inference/quantization/mxfp8_quantize.py:229``)
loops ``for row in tl.range(pid, n_rows, NUM_PROGRAMS)`` (``:247-249``) with no bound against the
buffer height ``M``. The wrapper ``mxfp8_quantize_rows`` (``:263-298``) checks only
``n_rows.numel() == 1``. With ``n_rows > M`` the kernel reads rows ``[M, n_rows)`` past ``x``,
writes FP8 data rows past ``out_data`` (``torch.empty(M, K)``) and writes scale bytes for those
rows: inside ``out_scale``'s padded last tile when ``M % 128 != 0``, and past ``out_scale`` once
``n_rows > ceil(M / 128) * 128``.

Intended behavior: ``n_rows`` is clamped to ``M`` (an explicit error is not possible without a
host sync, since ``n_rows`` is a device scalar captured in CUDA graphs). The clamped result is
exactly ``mxfp8_quantize(x)``.

Tests:
* CONTROL ``n_rows <= M``: rows ``[0, n_rows)`` bitwise equal ``mxfp8_quantize``; every scale byte
  of a row ``>= n_rows`` is zero (scale bytes mapped to rows with the cuBLAS 32x4x4 blocked layout
  written out in plain torch below).
* DEFECT ``M % 128 != 0`` and ``n_rows = ceil(M / 128) * 128``: ``x`` is the first ``M`` rows of a
  larger buffer whose tail rows hold a sentinel, so the out-of-bounds reads stay inside that
  buffer and are deterministic. The overrun scale bytes land in ``out_scale``'s own padding, which
  makes the effect observable. The overrun data rows (at most 256 B here) land in the 512-byte
  rounding slack the caching allocator gives ``out_data``, so the test corrupts nothing.
* SANITIZER ``n_rows`` far beyond ``M``: real out-of-bounds reads and writes, no assertion. Runs
  only under ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` (or ``MCORE_REPRO_SANITIZER=1``), because the
  caching allocator hides the overrun from memcheck and an unguarded run would corrupt memory:
  ``PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck python -m pytest <file>
  -k sanitizer``.

Requirements: 1 GPU with FP8 (e4m3) Triton support (SM89+; H100 is fine). No other dependency.
"""

import os

import pytest
import torch

from megatron.core.inference.quantization.mxfp8_quantize import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_COL_BLOCK,
    MXFP8_SCALE_ROW_BLOCK,
    mxfp8_quantize,
    mxfp8_quantize_rows,
)

# Max GPUs any test here needs.
REQUIRED_GPUS = 1

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mxfp8_quantize_rows asserts CUDA inputs"
)

SENTINEL = 1.0  # rows past x: amax 1.0 -> e8m0 scale byte 127 + ceil(log2(1/448)) = 119


def _ceil_div(a, b):
    return (a + b - 1) // b


def _scale_byte_rows(M, K):
    """Logical row of every byte of the swizzled scale buffer (-1 for padding columns).

    cuBLAS blocked (SWIZZLE_32_4_4) layout, from its definition: the [rows, K/32] scale matrix is
    tiled 128 rows x 4 columns, each tile is 512 contiguous bytes, and inside a tile the byte for
    (r, c) sits at (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4).
    """
    n_row_blocks = _ceil_div(M, MXFP8_SCALE_ROW_BLOCK)
    n_col_blocks = _ceil_div(K // MXFP8_BLOCK_SIZE, MXFP8_SCALE_COL_BLOCK)
    r = torch.arange(n_row_blocks * MXFP8_SCALE_ROW_BLOCK)[:, None]
    c = torch.arange(n_col_blocks * MXFP8_SCALE_COL_BLOCK)[None, :]
    tile = (r // 128) * n_col_blocks + c // 4
    off = tile * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    rows = torch.full((n_row_blocks * n_col_blocks * 512,), -1, dtype=torch.int64)
    rows[off.reshape(-1)] = r.expand_as(off).reshape(-1)
    real_col = (c < K // MXFP8_BLOCK_SIZE).expand_as(off).reshape(-1)
    rows[off.reshape(-1)[~real_col]] = -1
    return rows


def _x_in_sentinel_buffer(M, K, seed):
    g = torch.Generator().manual_seed(seed)
    rows = _ceil_div(M, MXFP8_SCALE_ROW_BLOCK) * MXFP8_SCALE_ROW_BLOCK + 128
    base = torch.full((rows, K), SENTINEL, dtype=torch.bfloat16)
    base[:M] = (torch.randn(M, K, generator=g) * 4).to(torch.bfloat16)
    base = base.cuda()
    x = base[:M]
    assert x.is_contiguous() and x.data_ptr() == base.data_ptr()
    return base, x


@pytest.mark.parametrize("M,K", [(120, 32), (124, 64), (256, 128)])
@pytest.mark.parametrize("n_frac", ["zero", "half", "all"])
def test_control_rows_within_height_match_mxfp8_quantize(M, K, n_frac):
    n = {"zero": 0, "half": M // 2, "all": M}[n_frac]
    _, x = _x_in_sentinel_buffer(M, K, seed=M + K)
    data, scale = mxfp8_quantize_rows(x, torch.tensor([n], dtype=torch.int32, device="cuda"))
    ref_data, ref_scale = mxfp8_quantize(x)
    assert data.shape == ref_data.shape and scale.shape == ref_scale.shape
    assert torch.equal(data[:n].view(torch.uint8), ref_data[:n].view(torch.uint8))
    byte_row = _scale_byte_rows(M, K).cuda()
    live = (byte_row >= 0) & (byte_row < n)
    expected = ref_scale.view(torch.uint8) * live.to(torch.uint8)
    assert torch.equal(scale.view(torch.uint8), expected)
    if n == M:
        assert torch.equal(scale.view(torch.uint8), ref_scale.view(torch.uint8))


@pytest.mark.parametrize("M,K", [(120, 32), (124, 64)])
def test_defect_rows_past_height_are_clamped(M, K):
    """n_rows = 128 > M: the result must be mxfp8_quantize(x), i.e. n_rows clamped to M.

    Shapes keep every out-of-bounds access contained: reads hit the sentinel rows of x's backing
    buffer, scale writes hit out_scale's padding rows, and the (128 - M) * K <= 256 data bytes
    written past out_data (M*K = 3840 / 7936 B) fall in its 512-B allocator rounding.
    """
    n_rows = _ceil_div(M, MXFP8_SCALE_ROW_BLOCK) * MXFP8_SCALE_ROW_BLOCK
    assert n_rows > M and (n_rows - M) * K <= (-M * K) % 512  # overrun fits the rounding slack
    _, x = _x_in_sentinel_buffer(M, K, seed=M + K)
    data, scale = mxfp8_quantize_rows(x, torch.tensor([n_rows], dtype=torch.int32, device="cuda"))
    ref_data, ref_scale = mxfp8_quantize(x)

    got = scale.view(torch.uint8).cpu()
    ref = ref_scale.view(torch.uint8).cpu()
    byte_row = _scale_byte_rows(M, K)
    bad = (got != ref).nonzero().flatten()
    assert bad.numel() == 0, (
        f"DEFECT d8: mxfp8_quantize_rows(x[{M}, {K}], n_rows={n_rows}) quantized rows past the "
        f"buffer height: {bad.numel()} scale bytes differ from mxfp8_quantize(x), all in rows "
        f"{sorted(set(byte_row[bad].tolist()))} (>= M={M}), holding {sorted(set(got[bad].tolist()))} "
        f"(the scale of the sentinel rows read past x; reference {sorted(set(ref[bad].tolist()))}). "
        f"The kernel loop is bounded only by n_rows; it must clamp to M."
    )
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8)), (
        "DEFECT d8: data rows [0, M) differ from mxfp8_quantize(x) when n_rows > M."
    )


_SANITIZER_RUN = (
    os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1"
    or os.environ.get("MCORE_REPRO_SANITIZER") == "1"
)


@pytest.mark.skipif(
    not _SANITIZER_RUN,
    reason="writes out of bounds for real; run under compute-sanitizer --tool memcheck with "
    "PYTORCH_NO_CUDA_MEMORY_CACHING=1 (or set MCORE_REPRO_SANITIZER=1)",
)
def test_sanitizer_rows_far_past_height():
    """n_rows = 8 * M: expect memcheck 'Invalid __global__ read' (x) and 'Invalid __global__ write'
    (out_data, out_scale) in _mxfp8_quant_swizzle_rows_kernel. No assertion: on a fixed kernel
    memcheck reports nothing."""
    M, K = 128, 64
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    n_rows = torch.tensor([8 * M], dtype=torch.int32, device="cuda")
    mxfp8_quantize_rows(x, n_rows)
    torch.cuda.synchronize()
