# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validation-only probe: an NVLS inference MoE layer at expert parallelism 3 on main.

Prints PROBE lines with the inference output compared to the same layer's training
(all-to-all) forward, or the error raised.
"""

import traceback

import pytest
import torch

from megatron.core.inference.utils import InferenceMode
from tests.unit_tests.inference.test_moe_dispatching_and_routing import _make_base_config
from tests.unit_tests.test_utilities import Utils


def test_probe_nvls_ep3():
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
    from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec
    from megatron.core.parallel_state import get_expert_model_parallel_group
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    if Utils.world_size != 3:
        pytest.skip("probe runs on exactly 3 GPUs")
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=3)
    try:
        try:
            config = _make_base_config(
                expert_model_parallel_size=3,
                num_moe_experts=12,
                inference_moe_token_dispatcher_type="nvls",
                moe_shared_expert_intermediate_size=None,
            )
            NVLSAllGatherVDispatcher.allocate_buffers(
                per_rank_worst_case_token_count=8,
                topk=config.moe_router_topk,
                hidden_size=config.hidden_size,
                ep_group=get_expert_model_parallel_group(),
            )
            layer = get_inference_optimized_moe_spec()(config=config).cuda().eval()
            hidden = torch.randn(3, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda")
            with torch.no_grad():
                training_output, _ = layer(hidden.clone())
                with InferenceMode.active():
                    output, _ = layer(hidden.clone())
            torch.cuda.synchronize()
            print(
                f"PROBE rank={Utils.rank} ok finite={bool(torch.isfinite(output).all())} "
                f"max_diff_vs_training={(output.float() - training_output.float()).abs().max().item()} "
                f"training_scale={training_output.float().abs().max().item()}"
            )
        except Exception:  # noqa: BLE001
            print(f"PROBE rank={Utils.rank} error\n{traceback.format_exc()}")
    finally:
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()
