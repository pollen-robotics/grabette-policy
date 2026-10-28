"""The MuJoCo side of `gripette_core.arm_service.ArmInterface`.

Three methods over a simulation instead of over CAN. That is the entire
difference between the simulated reBot and the real one as far as the gRPC
service is concerned: `grpc_server_real.py` hands the servicer a CAN-backed
interface, this hands it a MuJoCo-backed one, and the servicer - the Cartesian
integrator, the setpoint interpolator, the IK-jump watchdog, the reset ramp -
is the same object in both.

That identity is the point, not a convenience. A policy that passes in sim and
fails on hardware should fail for reasons about the ROBOT, not because the two
ran different control code.

RADIANS throughout, in ArmSpec order, as the protocol requires. The simulation
is also in radians, so nothing is converted here - unlike the real interface,
which crosses into the follower's motor degrees.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SimArmInterface:
    """`ArmInterface` backed by a `Simulation`.

    `set_torque(False)` is NOT a no-op. On the real arm it drops the arm; the
    honest simulation of that is to stop driving the actuators and let MuJoCo's
    gravity do what it does, which is what `_torque` gates below. A sim that
    quietly kept holding position would be the one place a policy could learn
    that releasing torque is harmless.
    """

    def __init__(self, sim, joint_names):
        self._sim = sim
        self._joint_names = list(joint_names)
        self._torque = True

    def get_positions(self) -> np.ndarray:
        """Current joint positions (rad), in ArmSpec order."""
        return self._sim.get_joint_positions(self._joint_names)

    def send_command_rad(self, joint_angles_rad: np.ndarray) -> None:
        """Command joint positions (rad), in ArmSpec order.

        Dropped while torque is off, so a command cannot silently re-engage an
        arm the caller believes is limp.
        """
        if not self._torque:
            return
        q = np.asarray(joint_angles_rad, dtype=float)
        if q.shape != (len(self._joint_names),):
            raise ValueError(
                f"expected {len(self._joint_names)} joint angles in ArmSpec "
                f"order {self._joint_names}, got shape {q.shape}")
        self._sim.set_joint_commands(q, self._joint_names)

    def set_torque(self, enable: bool) -> None:
        """Enable or disable the arm actuators.

        Disabling zeroes the position actuators' gains via the model, so the
        arm falls under gravity exactly as the real one does. Re-enabling
        restores them and holds the pose the arm is in at that moment, so a
        hand-placed arm is not snapped back to a stale target.
        """
        if enable == self._torque:
            return
        self._torque = bool(enable)
        if enable:
            self._sim.set_joint_commands(self.get_positions(), self._joint_names)
        self._sim.set_arm_actuators_enabled(self._torque, self._joint_names)
        logger.info("arm torque %s", "ENABLED" if enable else "DISABLED")
