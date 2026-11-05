"""--seed makes the sim's episode sequence reproducible.

Without it, every eval run drew different cube positions / home poses, so two
checkpoints (or two IK versions) were never compared on the same episodes.
"""
from pathlib import Path

import numpy as np

from openarm_gripette_simu.server import SimulationServer

SCENE = Path(__file__).resolve().parents[1] / "scenes" / "table_grasp.xml"


def _episodes(seed, n=3):
    """Cube placement + arm home joints of n consecutive random resets."""
    srv = SimulationServer(scene_xml=SCENE, seed=seed)
    out = []
    for _ in range(n):
        cube = srv.reset_episode_random()
        assert cube is not None
        out.append(np.concatenate([cube, srv._sim.get_arm_positions()]))
    return np.array(out)


def test_same_seed_same_episodes():
    np.testing.assert_allclose(_episodes(0), _episodes(0))


def test_different_seed_different_episodes():
    assert not np.allclose(_episodes(0), _episodes(1))


def test_servicer_rng_is_seeded_too():
    """run() builds the ArmServicer with seed=_seed_servicer (legacy cube
    randomization draws from it); it must be reproducible as well."""
    import threading

    from openarm_gripette_simu.arm_servicer import ArmServicer

    def draws(seed):
        srv = SimulationServer(scene_xml=SCENE, seed=seed)
        svc = ArmServicer(srv._sim, srv._kin, threading.Lock(), 0.0, server=srv,
                          seed=srv._seed_servicer)
        return svc._rng.uniform(size=4)

    np.testing.assert_array_equal(draws(0), draws(0))
