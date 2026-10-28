# Disposable probe: NIXL metadata exchange after registering memory, as NixlCopyService does.
import pytest
import torch
import torch.distributed as dist

from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("memory", ["none", "cpu", "cuda"])
def test_nixl_exchange_after_registration(memory):
    pytest.importorskip("nixl._api")
    from nixl._api import nixl_agent, nixl_agent_config

    Utils.initialize_model_parallel(1, 1)
    rank, world = dist.get_rank(), dist.get_world_size()
    name = f"probe2-{memory}-{rank}"
    agent = nixl_agent(name, nixl_agent_config(backends=["UCX"]))
    tensors = []
    if memory != "none":
        device = "cpu" if memory == "cpu" else f"cuda:{torch.cuda.current_device()}"
        tensors = [torch.zeros(1024, device=device), torch.zeros(2048, device=device)]
        agent.register_memory(tensors)
    gathered = [None] * world
    dist.all_gather_object(gathered, (name, agent.get_agent_metadata()))
    failures = []
    for peer, (peer_name, metadata) in enumerate(gathered):
        if peer == rank:
            continue
        try:
            agent.add_remote_agent(metadata)
        except Exception as exc:
            failures.append((peer, type(exc).__name__))
    print(f"PROBE2 rank={rank} world={world} memory={memory} failures={failures}", flush=True)
    dist.barrier()
    Utils.destroy_model_parallel()
    assert not failures
