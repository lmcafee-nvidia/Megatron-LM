# Disposable probe: a pure-NIXL metadata exchange, with no Megatron copy service.
import os

import pytest
import torch
import torch.distributed as dist

from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("ucx_tls", ["inherited", "unset"])
def test_pure_nixl_metadata_exchange(ucx_tls):
    pytest.importorskip("nixl._api")
    from nixl._api import nixl_agent, nixl_agent_config

    Utils.initialize_model_parallel(1, 1)
    rank, world = dist.get_rank(), dist.get_world_size()
    print(f"PROBE rank={rank} world={world} UCX_TLS={os.environ.get('UCX_TLS')!r}", flush=True)
    if ucx_tls == "unset":
        os.environ.pop("UCX_TLS", None)
    name = f"probe-{ucx_tls}-{rank}"
    agent = nixl_agent(name, nixl_agent_config(backends=["UCX"]))
    gathered = [None] * world
    dist.all_gather_object(gathered, (name, agent.get_agent_metadata()))
    failures = []
    for peer, (peer_name, metadata) in enumerate(gathered):
        if peer == rank:
            continue
        try:
            agent.add_remote_agent(metadata)
        except Exception as exc:  # report every peer, not only the first
            failures.append((peer, peer_name, type(exc).__name__))
    print(f"PROBE rank={rank} ucx_tls={ucx_tls} failures={failures}", flush=True)
    dist.barrier()
    Utils.destroy_model_parallel()
    assert not failures
