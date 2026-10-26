# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Repro for defect d7: torch fallback of ``apply_rotary_pos_emb_with_cos_sin``.

``apply_rotary_pos_emb_with_cos_sin(t, cos, sin, rotary_interleaved)``
(``megatron/core/models/common/embeddings/rope_utils.py:430``) receives the *precomputed*
``cos(theta)`` / ``sin(theta)`` tables of shape ``(seq, d_rot / 2)`` (from
``RotaryEmbedding.get_cos_sin``). When flash-attn's rotary kernel is not importable
(``apply_rotary_emb_flash is None``) it stacks ``[cos, sin]`` into ``freqs`` and hands them to
``_apply_rotary_pos_emb_bshd``, which treats ``freqs`` as *angles* and evaluates
``torch.cos(freqs)`` / ``torch.sin(freqs)``, i.e. ``cos(cos theta)``.

The only production caller is static flash-decode (``attention.py:688-699``), which passes
``query`` / ``key`` laid out ``[seq, batch, heads, head_dim]``.

Reference: plain torch, written from the RoPE definition (each pair ``(x_a, x_b)`` is rotated by
the angle ``theta``), for both the GPT-NeoX (half split) and GPT-J (interleaved) pairings.

Runs on CPU or one GPU. The flash-attn control needs CUDA and ``flash_attn.layers.rotary``.
"""

import pytest
import torch

import megatron.core.models.common.embeddings.rope_utils as rope_utils
from megatron.core.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
    apply_rotary_pos_emb_with_cos_sin,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

# Max GPUs any test here needs: the flash-attn control needs one; the reference control and the
# defect test also run on CPU.
REQUIRED_GPUS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ, BATCH, HEADS = 5, 3, 2
ATOL = 1e-5
RTOL = 1e-5

# (head_dim, kv_channels used for the rotary table). rotary dim = kv_channels.
# First case: full rotary. Second: partial rotary (head_dim > d_rot), as in
# tests/unit_tests/inference/test_flash_decode.py::TestRotaryEmbeddingWithPrecomputedCosSin.
SHAPES = [(16, 16), (16, 8)]


def _angles(d_rot: int, seq: int, offset: int = 0) -> torch.Tensor:
    """theta[pos, i] = (pos + offset) * inv_freq[i], shape (seq, d_rot/2), from RotaryEmbedding."""
    rope = RotaryEmbedding(kv_channels=d_rot, rotary_percent=1.0, use_cpu_initialization=True)
    return rope.get_freqs_non_repeated(seq, offset).to(DEVICE)


def _production_cos_sin(d_rot: int, seq: int, offset: int = 0):
    """cos/sin tables exactly as the static flash-decode caller obtains them."""
    rope = RotaryEmbedding(kv_channels=d_rot, rotary_percent=1.0, use_cpu_initialization=True)
    if torch.cuda.is_available():
        # Production path: get_cos_sin moves inv_freq to the current CUDA device.
        cos, sin = rope.get_cos_sin(seq, offset)
    else:
        # get_cos_sin hard-codes a CUDA move; its body is cos/sin of get_freqs_non_repeated.
        freqs = rope.get_freqs_non_repeated(seq, offset)
        cos, sin = torch.cos(freqs), torch.sin(freqs)
    return cos.to(DEVICE), sin.to(DEVICE)


def _reference_rope_cos_sin(
    t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool
) -> torch.Tensor:
    """RoPE from its definition, given cos/sin values.

    t: [seq, batch, heads, head_dim]; cos, sin: [seq, d_rot/2]. Only the first d_rot channels
    are rotated; the rest pass through. Pair i is (x[i], x[i + d_rot/2]) for the half-split
    (GPT-NeoX) layout and (x[2i], x[2i+1]) for the interleaved (GPT-J) layout; each pair is
    rotated by the angle whose cosine/sine are cos[:, i] / sin[:, i].
    """
    half = cos.shape[-1]
    d_rot = 2 * half
    rot, passthrough = t[..., :d_rot], t[..., d_rot:]
    c = cos[:, None, None, :].to(t.dtype)
    s = sin[:, None, None, :].to(t.dtype)
    if interleaved:
        a, b = rot[..., 0::2], rot[..., 1::2]
    else:
        a, b = rot[..., :half], rot[..., half:]
    ra = a * c - b * s
    rb = b * c + a * s
    if interleaved:
        out = torch.stack((ra, rb), dim=-1).flatten(start_dim=-2)
    else:
        out = torch.cat((ra, rb), dim=-1)
    return torch.cat((out, passthrough), dim=-1)


def _reference_rope_complex(t: torch.Tensor, theta: torch.Tensor, interleaved: bool):
    """Second, independent formulation: multiply each pair, viewed as a complex number, by
    exp(i * theta)."""
    half = theta.shape[-1]
    d_rot = 2 * half
    rot, passthrough = t[..., :d_rot].double(), t[..., d_rot:]
    if interleaved:
        z = torch.complex(rot[..., 0::2], rot[..., 1::2])
    else:
        z = torch.complex(rot[..., :half], rot[..., half:])
    z = z * torch.polar(torch.ones_like(theta.double()), theta.double())[:, None, None, :]
    if interleaved:
        out = torch.stack((z.real, z.imag), dim=-1).flatten(start_dim=-2)
    else:
        out = torch.cat((z.real, z.imag), dim=-1)
    return torch.cat((out.to(t.dtype), passthrough), dim=-1)


def _training_emb(theta: torch.Tensor, interleaved: bool) -> torch.Tensor:
    """The angle table the training path feeds _apply_rotary_pos_emb_bshd.

    Same construction as RotaryEmbedding.get_emb (which hard-codes a CUDA move, so it is
    rebuilt here): [seq, 1, 1, d_rot].
    """
    if not interleaved:
        emb = torch.cat((theta, theta), dim=-1)
    else:
        emb = torch.stack((theta.view(-1, 1), theta.view(-1, 1)), dim=-1).view(theta.shape[0], -1)
    return emb[:, None, None, :]


def _inputs(head_dim: int, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(SEQ, BATCH, HEADS, head_dim, generator=g, dtype=torch.float32).to(DEVICE)


@pytest.mark.parametrize("interleaved", [False, True], ids=["half_split", "interleaved"])
@pytest.mark.parametrize("head_dim,d_rot", SHAPES, ids=["full_rot", "partial_rot"])
def test_control_reference_is_self_consistent(interleaved, head_dim, d_rot):
    """CONTROL: the cos/sin reference agrees with the complex-exponential formulation, with
    Megatron's own angle-based (training) RoPE, preserves pair norms, and is the identity at
    position 0."""
    t = _inputs(head_dim)
    theta = _angles(d_rot, SEQ, offset=3)
    cos, sin = _production_cos_sin(d_rot, SEQ, offset=3)
    torch.testing.assert_close(cos, torch.cos(theta), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(sin, torch.sin(theta), atol=1e-6, rtol=1e-6)

    ref = _reference_rope_cos_sin(t, cos, sin, interleaved)
    torch.testing.assert_close(
        ref, _reference_rope_complex(t, theta, interleaved), atol=ATOL, rtol=RTOL
    )

    # Megatron's training RoPE takes angles; on angles it must agree with the reference.
    training = _apply_rotary_pos_emb_bshd(
        t, _training_emb(theta, interleaved), rotary_interleaved=interleaved
    )
    torch.testing.assert_close(ref, training, atol=ATOL, rtol=RTOL)

    # Rotation preserves the norm of the rotated slice and leaves the tail untouched.
    torch.testing.assert_close(
        ref[..., :d_rot].norm(dim=-1), t[..., :d_rot].norm(dim=-1), atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(ref[..., d_rot:], t[..., d_rot:], atol=0, rtol=0)

    # Position 0 (offset 0): theta = 0, so RoPE is the identity.
    cos0, sin0 = _production_cos_sin(d_rot, 1, offset=0)
    torch.testing.assert_close(
        _reference_rope_cos_sin(t[:1], cos0, sin0, interleaved), t[:1], atol=0, rtol=0
    )


@pytest.mark.parametrize("interleaved", [False, True], ids=["half_split", "interleaved"])
@pytest.mark.parametrize("head_dim,d_rot", SHAPES, ids=["full_rot", "partial_rot"])
def test_control_flash_attn_rotary_matches_reference(interleaved, head_dim, d_rot):
    """CONTROL: with flash-attn's rotary kernel installed, the production path matches the
    reference."""
    if rope_utils.apply_rotary_emb_flash is None:
        pytest.skip("flash_attn.layers.rotary.apply_rotary_emb is not importable")
    if not torch.cuda.is_available():
        pytest.skip("flash-attn's rotary kernel needs CUDA")
    t = _inputs(head_dim)
    cos, sin = _production_cos_sin(d_rot, SEQ, offset=3)
    out = apply_rotary_pos_emb_with_cos_sin(t, cos, sin, rotary_interleaved=interleaved)
    ref = _reference_rope_cos_sin(t, cos, sin, interleaved)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("interleaved", [False, True], ids=["half_split", "interleaved"])
@pytest.mark.parametrize("head_dim,d_rot", SHAPES, ids=["full_rot", "partial_rot"])
def test_defect_torch_fallback_matches_reference(monkeypatch, interleaved, head_dim, d_rot):
    """DEFECT: the torch fallback (flash-attn rotary unavailable) must rotate by theta given
    cos(theta)/sin(theta)."""
    monkeypatch.setattr(rope_utils, "apply_rotary_emb_flash", None)
    t = _inputs(head_dim)
    cos, sin = _production_cos_sin(d_rot, SEQ, offset=3)
    out = apply_rotary_pos_emb_with_cos_sin(t, cos, sin, rotary_interleaved=interleaved)
    ref = _reference_rope_cos_sin(t, cos, sin, interleaved)
    assert out.shape == ref.shape, f"shape {tuple(out.shape)} != {tuple(ref.shape)}"
    max_err = (out - ref).abs().max().item()
    matches = bool(torch.allclose(out, ref, atol=ATOL, rtol=RTOL))
    assert matches, (
        f"DEFECT d7: torch fallback of apply_rotary_pos_emb_with_cos_sin "
        f"(interleaved={interleaved}, head_dim={head_dim}, d_rot={d_rot}) does not apply RoPE: "
        f"it feeds the precomputed cos/sin values to _apply_rotary_pos_emb_bshd, which takes "
        f"cos()/sin() of them again (cos(cos theta)); max |out - ref| = {max_err:.4g}"
    )
