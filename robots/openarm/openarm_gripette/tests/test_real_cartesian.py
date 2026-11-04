"""SendCartesianDelta guards on the real-arm servicer (fake arm, no hardware)."""

import numpy as np
from openarm_gripette_simu.proto import arm_pb2

IDENTITY_R6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


class JumpKin:
    """Real FK, but IK always returns a fixed (branch-flipped) solution."""

    def __init__(self, kin, solution):
        self._kin = kin
        self.solution = np.asarray(solution, dtype=float)

    def forward(self, q):
        return self._kin.forward(q)

    def inverse(self, target_tf, current_joint_positions=None):
        return self.solution.copy()


def _flip(q):
    """`q` with a 60 deg wrist_pitch jump: a typical singular branch flip."""
    out = np.array(q, dtype=float)
    out[6] += np.deg2rad(60.0)
    return out


# RA-4: the IK-jump watchdog compared against _latest_target_joints, which is
# None until the first ACCEPTED command, so the very first command after
# startup could flip a branch unchecked (success=True, 60 deg on the wrist).
def test_first_command_is_jump_guarded(make_servicer, fake_arm, kin):
    s = make_servicer(fake_arm)
    assert s._latest_target_joints is None  # fresh server, no command yet
    s._kin = JumpKin(kin, _flip(fake_arm.q))
    resp = s.SendCartesianDelta(arm_pb2.CartesianDelta(dx=0.0, dy=0.0, dz=0.0, dr6d=IDENTITY_R6D), None)
    assert not resp.success
    assert s._latest_target_joints is None  # nothing handed to the interpolator


def test_first_small_command_still_accepted(make_servicer, fake_arm):
    """The RA-4 guard must not trip on a normal first step (real Placo IK)."""
    s = make_servicer(fake_arm)
    resp = s.SendCartesianDelta(arm_pb2.CartesianDelta(dx=0.005, dy=0.0, dz=0.0, dr6d=IDENTITY_R6D), None)
    assert resp.success, resp.error
    assert s._latest_target_joints is not None
