# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validation-only probe: what optimized inference experts do with FP8/FP4 parameter storage.

Each case builds one inference-optimized MoE layer, compares its inference output with a
BF16 layer holding the same (dequantized) weights, then rewrites the expert weights the
way refit does (``quantize_`` or ``copy_``), refreshes caches and compares again.
Results are printed as PROBE lines; the probe asserts nothing about the outcome.
"""

import traceback

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.utils import InferenceMode
from megatron.core.resharding import refresh_module_caches
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

BASE = dict(
    num_layers=1,
    hidden_size=128,
    ffn_hidden_size=256,
    num_attention_heads=4,
    num_moe_experts=4,
    moe_ffn_hidden_size=256,
    moe_router_topk=2,
    moe_router_dtype="fp32",
    moe_grouped_gemm=True,
    moe_token_dispatcher_type="alltoall",
    inference_moe_token_dispatcher_type="nccl",
    activation_func=squared_relu,
    normalization="RMSNorm",
    num_query_groups=2,
    add_bias_linear=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    transformer_impl="inference_optimized",
    expert_tensor_parallel_size=1,
    use_cpu_initialization=False,
)

CASES = {
    "bf16-params": {},
    "fp8-tensorwise-compute": dict(fp8="hybrid", fp8_recipe="tensorwise"),
    "fp8-tensorwise-param": dict(fp8="hybrid", fp8_recipe="tensorwise", fp8_param=True),
    "fp8-delayed-param": dict(fp8="hybrid", fp8_recipe="delayed", fp8_param=True),
    "fp4-compute": dict(fp4="e2m1", fp4_recipe="nvfp4"),
    "fp4-param": dict(fp4="e2m1", fp4_recipe="nvfp4", fp4_param=True),
}


def _layer(**overrides):
    from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec

    config = TransformerConfig(**{**BASE, **overrides})
    with get_fp8_context(config, 0, is_init=True):
        layer = get_inference_optimized_moe_spec()(config=config).cuda().eval()
    return config, layer


def _dense(tensor):
    return tensor.dequantize() if hasattr(tensor, "dequantize") else tensor


def _run(config, layer, hidden):
    with torch.no_grad(), get_fp8_context(config, 0), InferenceMode.active():
        output, _ = layer(hidden.clone())
    return output.float()


def _training_run(config, layer, hidden):
    with torch.no_grad(), get_fp8_context(config, 0):
        output, _ = layer(hidden.clone())
    return output.float()


def _expert_params(layer):
    return [p for n, p in layer.experts.named_parameters() if ".weight" in n]


@pytest.mark.parametrize("case", list(CASES))
def test_probe_quantized_expert_storage(case):
    from megatron.core.transformer.moe.token_dispatcher_inference import (
        InferenceAllGatherDispatcherBase,
    )

    Utils.initialize_model_parallel(1, 1)
    InferenceAllGatherDispatcherBase.allocate_valid_tokens_tensor()
    try:
        if case.startswith("fp4") and torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("TE NVFP4 needs Blackwell")
        torch.manual_seed(1234)
        report = {"case": case}
        try:
            config, layer = _layer(**CASES[case])
        except Exception:  # noqa: BLE001
            print(f"PROBE {report} build-error\n{traceback.format_exc()}")
            return
        ref_config, ref = _layer()
        params = dict(layer.named_parameters())
        report["param_type"] = type(layer.experts.linear_fc1.weight0).__name__
        with torch.no_grad():
            for name, ref_param in ref.named_parameters():
                ref_param.copy_(_dense(params[name]))
        hidden = torch.randn(64, 1, BASE["hidden_size"], device="cuda", dtype=torch.bfloat16)
        try:
            out = _run(config, layer, hidden)
        except Exception:  # noqa: BLE001
            print(f"PROBE {report} first-forward-error\n{traceback.format_exc()}")
            return
        out_ref = _run(ref_config, ref, hidden)
        report["first_vs_bf16_ref"] = (out - out_ref).abs().max().item()
        report["ref_scale"] = out_ref.abs().max().item()
        experts = layer.experts
        report["param_type_after_build"] = type(experts.linear_fc1.weight0).__name__
        fc1 = getattr(experts, "_fc1_weight", None)
        report["concat_dtype"] = None if fc1 is None else str(fc1.dtype)
        report["param_shares_concat"] = fc1 is not None and (
            experts.linear_fc1.weight0.data_ptr() == fc1[0].data_ptr()
        )
        try:
            report["training_vs_inference"] = (
                (_training_run(config, layer, hidden) - out).abs().max().item()
            )
        except Exception as error:  # noqa: BLE001
            report["training_vs_inference"] = f"error {error!r}"

        # Rewrite expert weights the way refit does, then refresh caches.
        with torch.no_grad():
            for param in _expert_params(layer):
                new = 0.05 * torch.randn(param.shape, device="cuda", dtype=torch.bfloat16)
                if hasattr(param, "quantize_"):
                    param.quantize_(new)
                else:
                    param.copy_(new)
            refresh_module_caches(layer)
            params = dict(layer.named_parameters())
            for name, ref_param in ref.named_parameters():
                ref_param.copy_(_dense(params[name]))
        try:
            out_after = _run(config, layer, hidden)
        except Exception:  # noqa: BLE001
            print(f"PROBE {report} post-update-forward-error\n{traceback.format_exc()}")
            return
        out_ref_after = _run(ref_config, ref, hidden)
        report["after_vs_new_ref"] = (out_after - out_ref_after).abs().max().item()
        report["after_vs_before"] = (out_after - out).abs().max().item()
        report["new_ref_scale"] = out_ref_after.abs().max().item()
        print(f"PROBE {report}")
    finally:
        Utils.destroy_model_parallel()
