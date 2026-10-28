# Disposable probe: which registrations make a peer's NIXL metadata unloadable.
import pytest
import torch
import torch.distributed as dist

from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("layout", ["distinct", "disjoint-views", "duplicate", "overlap"])
def test_nixl_exchange_registration_layout(layout):
    pytest.importorskip("nixl._api")
    from nixl._api import nixl_agent, nixl_agent_config

    Utils.initialize_model_parallel(1, 1)
    rank, world = dist.get_rank(), dist.get_world_size()
    name = f"probe3-{layout}-{rank}"
    agent = nixl_agent(name, nixl_agent_config(backends=["UCX"]))
    base = torch.zeros(4096, device=f"cuda:{torch.cuda.current_device()}")
    tensors = {
        "distinct": [base, torch.zeros(4096, device=base.device)],
        "disjoint-views": [base[:2048], base[2048:]],
        "duplicate": [base, base],
        "overlap": [base, base[:2048]],
    }[layout]
    agent.register_memory(tensors)
    gathered = [None] * world
    dist.all_gather_object(gathered, (name, agent.get_agent_metadata()))
    failures = []
    for peer, (peer_name, metadata) in enumerate(gathered):
        if peer != rank:
            try:
                agent.add_remote_agent(metadata)
            except Exception as exc:
                failures.append((peer, type(exc).__name__))
    print(f"PROBE3 rank={rank} world={world} layout={layout} failures={failures}", flush=True)
    dist.barrier()
    Utils.destroy_model_parallel()
    assert not failures


def test_nixl_copy_service_sends_one_tensor_to_two_peers():
    """Megatron's service, with one source tensor sent to two peers, as GTP does."""
    pytest.importorskip("nixl._api")
    from megatron.core.resharding.copy_services.nixl_copy_service import NixlCopyService

    Utils.initialize_model_parallel(1, 1)
    rank, world = dist.get_rank(), dist.get_world_size()
    device = f"cuda:{torch.cuda.current_device()}"
    service = NixlCopyService()
    source = torch.full((64,), float(rank), device=device)
    peers = [(rank + 1) % world, (rank + 2) % world]
    received = {}
    for offset, peer in enumerate(peers):
        service.submit_send(source, peer, task_id=rank * world + peer)
        sender = (rank - offset - 1) % world
        received[sender] = torch.zeros(64, device=device)
        service.submit_recv(received[sender], sender, task_id=sender * world + rank)
    error = None
    try:
        service.run()
        torch.cuda.synchronize()
    except Exception as exc:
        error = type(exc).__name__
    print(f"PROBE3 rank={rank} world={world} service-two-peers error={error}", flush=True)
    dist.barrier()
    Utils.destroy_model_parallel()
    assert error is None
    assert all(torch.all(t == float(src)) for src, t in received.items())
