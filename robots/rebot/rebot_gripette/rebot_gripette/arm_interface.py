"""The reBot's implementation of `gripette_core.arm_service.ArmInterface`.

Three methods over the RobStride CAN bus, in RADIANS and in ArmSpec order.
The shared `ArmServicer` holds all the control logic; everything here is unit
conversion, joint-name mapping, caching and bus hygiene.

The reBot is a 6-joint arm as far as this interface is concerned. Motor 7 is
the stock rack gripper, which the Gripette replaces; the Gripette's Feetech
servos are not on the CAN bus and serve their own GripperService. So motor 7
is read and clamped like the rest (it is still on the bus) but never commanded
from here.
"""

import logging
import threading
import time

import numpy as np

from rebot_gripette.limits import apply_limits
from rebot_gripette.spec import ARM_JOINT_NAMES  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi



class ReBotArmInterface:
    """ArmInterface over a connected `ReBotFollower`.

    Caches the most recent joint read with a short TTL so GetArmState and
    SendCartesianDelta inside one control cycle share a single CAN refresh.
    Without it every inference step triggers two full refreshes, doubling bus
    load and packet drops — the same problem the OpenArm driver solved this
    way, and worse here because the RobStride gs_usb queue is only 10 frames
    deep (an over-fast loop produced ENOBUFS during bring-up).
    """

    def __init__(self, robot, state_cache_ttl_s: float = 0.02):
        self._robot = robot
        self._lock = threading.Lock()
        self._state_cache_ttl = state_cache_ttl_s
        self._cached_positions_rad: np.ndarray | None = None
        self._cached_positions_ts: float = 0.0

    # -- reads ------------------------------------------------------------
    def get_positions(self) -> np.ndarray:
        """Arm joint positions in radians, in ARM_JOINT_NAMES order.

        There is exactly one joint space here: motor degrees, which are the
        URDF's angles, which are what the follower reports and accepts. The
        export was validated against raw mechPos (`22_mirror.py`) and confirmed
        on hardware with `examples/mirror_arm.py`, and `ReBotFollowerConfig`
        pins joint_directions to the identity so nothing transforms on the way
        through. These radians therefore go straight into Kinematics as URDF
        angles with no conversion - see follower.py for why the vendor's
        leader-arm mapping is not in play.

        Raises RuntimeError if any arm joint produced no feedback frame. The
        plugin substitutes 0.0 for a silent motor rather than reporting the
        gap, and 0 rad is the lower hard stop for shoulder_lift and
        elbow_flex - so a single dropped frame would otherwise tell the
        servicer the arm is folded onto its stops and let it interpolate from
        there. Staleness is NOT covered: motorbridge keeps the last state, so
        this catches "never arrived", not "arrived a while ago".
        """
        now = time.monotonic()
        if (self._cached_positions_rad is not None
                and (now - self._cached_positions_ts) < self._state_cache_ttl):
            return self._cached_positions_rad.copy()
        with self._lock:
            # Double-checked: another thread may have refreshed while we waited.
            now = time.monotonic()
            if (self._cached_positions_rad is not None
                    and (now - self._cached_positions_ts) < self._state_cache_ttl):
                return self._cached_positions_rad.copy()
            obs = self._robot.get_observation()

            # get_observation() has just polled, so a None state here is
            # exactly the condition under which it substituted 0.0.
            silent = [n for n in ARM_JOINT_NAMES
                      if self._robot.motors[n].get_state() is None]
            if silent:
                raise RuntimeError(
                    f"No feedback from {silent}. The plugin reports 0.0 deg "
                    f"for a silent motor, and zero is the lower hard stop for "
                    f"shoulder_lift and elbow_flex - refusing to report a pose "
                    f"that would read as the arm folded onto its stops.")

            deg = np.array([obs[f"{n}.pos"] for n in ARM_JOINT_NAMES],
                           dtype=np.float64)
            self._cached_positions_rad = deg * DEG_TO_RAD
            self._cached_positions_ts = time.monotonic()
            return self._cached_positions_rad.copy()

    # -- writes -----------------------------------------------------------
    def send_command_rad(self, joint_angles_rad: np.ndarray) -> None:
        """Command the six arm joints. The gripper is untouched.

        Same space as `get_positions` - motor degrees, which are the URDF's
        angles - so a pose read back and sent straight out is a no-op, and a
        target computed by the IK from the model is commanded as the model
        meant it. send_action multiplies by joint_directions on the way to the
        wire, and `ReBotFollowerConfig` pins those to 1.0, so nothing is
        transformed in either direction.
        """
        deg = np.asarray(joint_angles_rad, dtype=np.float64) * RAD_TO_DEG
        action = {f"{n}.pos": float(deg[i])
                  for i, n in enumerate(ARM_JOINT_NAMES)}
        with self._lock:
            self._robot.send_action(action)

    def set_torque(self, enable: bool) -> None:
        """Enable or disable torque on the arm motors.

        Disable = the motors freewheel and the arm FALLS. On this arm that is
        partly cushioned by hard mechanical stops it rests on, but do not rely
        on that: the elbow and shoulder carry real mass.

        Re-applying the limits on ENABLE is not belt-and-braces, it is
        required. The RobStride 0x7000 register range is runtime-only and does
        not survive a disable/enable cycle or a power cycle;
        `store_parameters()` covers 0x2000 only. An arm enabled without this
        runs at the firmware default torque ceiling.
        """
        with self._lock:
            if enable:
                self._robot.bus.enable_torque(num_retry=2)
                self._reapply_limits()
            else:
                self._robot.bus.disable_torque(num_retry=2)

    def _reapply_limits(self) -> None:
        handles = getattr(self._robot.bus, "motors", None)
        if not handles:
            logger.warning(
                "Could not reach the motor handles to re-apply torque limits; "
                "the arm may be running at the firmware default ceiling.")
            return
        try:
            apply_limits(handles, strict=False)
        except Exception as e:
            # Non-fatal by design: refusing to enable because a readback
            # mismatched would leave the arm limp, which is not obviously
            # safer. Loud instead.
            logger.error(f"Torque limits NOT fully applied: {e}")
