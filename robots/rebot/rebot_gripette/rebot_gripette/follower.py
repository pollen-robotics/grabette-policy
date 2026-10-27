"""reBot B601-RS follower, pinned to a single joint space.

THE VENDOR MAPPING, AND WHY IT IS GONE
    The upstream config ships
        joint_directions = {elbow_flex: -1.0, wrist_flex: -1.0,
                            wrist_yaw: -1.0, gripper: 6.0, ...}
    which is not geometry - `gripper: 6.0` is a gear ratio, not a sign. The
    dict maps a LEADER arm's joint convention and scale onto this follower's
    motors, for Seeed's own leader/follower teleop pairing.

    The plugin then applies it on one side only:
        send_action      position * joint_directions.get(name, 0.0)   (:655)
        get_observation  math.degrees(state.pos)  - raw motor          (:618)
    so observation and action speak different spaces. Feeding an observation
    straight back as an action - the core LeRobot pattern behind teleop
    recording, dataset replay and policy rollout - therefore commands the
    MIRROR of the current pose on the three joints with direction -1.0, and
    scales the gripper by 6.

    This stack has no leader arm: a policy drives the arm over gRPC. It also
    does not use the reBot's rack gripper, which the Gripette replaces on its
    own Feetech bus behind its own service. So the mapping buys nothing here
    and costs an entire second joint space to keep straight. Every entry is
    pinned to 1.0 below.

ONE SPACE
    motor degrees == the URDF's angles == what send_action takes == what
    get_observation returns. The export was validated against raw mechPos
    (rebot-lerobot/22_mirror.py) and confirmed again on hardware with
    examples/mirror_arm.py, so motor space is the model's space and there is
    nothing left to convert.

    That is the point of pinning rather than converting: a transform whose only
    job is to undo a vendor inconsistency is a transform someone eventually
    applies twice, or not at all. The cost of getting it wrong is a pose
    mirrored on three joints - elbow_flex +60 deg arriving as -60, which is
    past its lower stop at zero.

WHY THE DICT IS PINNED AND NOT EMPTIED
    send_action looks up `joint_directions.get(motor_name, 0.0)`. A joint
    MISSING from the dict is multiplied by zero - that is, commanded to its
    zero position - and on this arm zero is the lower hard stop for
    shoulder_lift and elbow_flex. An empty dict would be the most dangerous
    possible value, so every joint stays present with an explicit 1.0.

    connect() refuses to open the bus if anything has put a non-identity value
    back, so a plugin upgrade restoring the vendor default fails loudly instead
    of silently mirroring three joints.

NOT CHANGED
    send_action, and the max_relative_target clamp inside it: with identity
    directions the clamp compares a motor-space goal against a motor-space
    present position, which is the same space.

    Torque stays raw motor torque.
"""
import logging
from dataclasses import dataclass, field

from lerobot.robots.robot import RobotConfig
from lerobot_robot_seeed_b601 import SeeedB601RSFollower, SeeedB601RSFollowerConfig

logger = logging.getLogger(__name__)


def _identity_directions() -> dict[str, float]:
    """Every joint the upstream config knows about, mapped to 1.0.

    Derived from the base class's own default rather than a hardcoded list, so
    a joint added upstream arrives with 1.0 instead of being absent - and
    absent means send_action multiplies it by 0.0 and commands it to its stop.
    """
    base = SeeedB601RSFollowerConfig.__dataclass_fields__["joint_directions"]
    return {name: 1.0 for name in base.default_factory()}


@RobotConfig.register_subclass("seeed_b601_rs_follower_fixed")
@dataclass
class ReBotFollowerConfig(SeeedB601RSFollowerConfig):
    """Registers a --robot.type that resolves to the FIXED follower.

    Subclasses the upstream SeeedB601RSFollowerConfig. Every numeric default
    (joint_limits, mit_kp/kd, temperature thresholds) is inherited unchanged;
    joint_directions is the one exception and is pinned to the identity - see
    the module docstring. Without this registration, CLI entry points
    (lerobot-record, lerobot-replay, ...) invoked with
    --robot.type=seeed_b601_rs_follower construct the unfixed base class, and
    both the mirrored mapping and the observation/action mismatch are back.

    The registered name is deliberately still "seeed_b601_rs_follower_fixed",
    the string this config was validated under: renaming it silently breaks
    every existing command line and recorded dataset that names it.

    Resolution works by naming convention: make_device_from_device_class strips
    the trailing "Config" from this class name and looks for ReBotFollower in
    the PARENT module - i.e. in rebot_gripette/__init__.py, which is why that
    file re-exports it. Renaming either half, or dropping the re-export, breaks
    CLI instantiation.

    For CLI use this module must be imported before draccus parses the
    argument, since registration happens at import time. LeRobot auto-imports
    only distributions whose metadata name starts with "lerobot_robot_" (see
    lerobot.utils.import_utils.register_third_party_plugins); this one is
    called "rebot-gripette", so it is NOT auto-discovered - a script that wants
    --robot.type=seeed_b601_rs_follower_fixed must `import rebot_gripette`
    itself.
    """

    joint_directions: dict[str, float] = field(
        default_factory=_identity_directions)


class ReBotFollower(SeeedB601RSFollower):
    """Follower that speaks motor space, and only motor space.

    Subclasses the upstream SeeedB601RSFollower (lerobot_robot_seeed_b601).
    connect() is overridden to refuse a non-identity joint_directions before
    the bus is opened; the rest are new helpers. calibrate, send_action,
    get_observation, safe_zero and the temperature protection are inherited
    as-is - with identity directions the inherited ones are already correct.

    DO NOT override `name`. Robot.calibration_fpath is derived from it, so a
    different name points at a directory with no calibration file, is_calibrated
    then returns False, and connect() walks straight into calibrate() - which
    calls set_zero_position() on all seven motors and destroys the existing
    zero. Inheriting the base name keeps this class pointed at the same
    calibration as the robot it is fixing.
    """

    def check_joint_directions(self) -> None:
        """Refuse anything but the identity. See the module docstring.

        Two distinct failures, both of which move the arm rather than report
        anything: a direction that is not 1.0 mirrors or rescales every command
        on that joint, and a joint missing from the dict is multiplied by the
        base class's 0.0 default and commanded to its zero - a hard stop on
        shoulder_lift and elbow_flex.
        """
        directions = self.config.joint_directions
        missing = [n for n in self.config.motor_can_ids if n not in directions]
        if missing:
            raise ValueError(
                f"joint_directions has no entry for {missing}. send_action "
                "reads it with .get(name, 0.0), so those joints would be "
                "multiplied by zero - i.e. commanded to their zero position, "
                "which is a hard stop for shoulder_lift and elbow_flex.")

        wrong = {n: d for n, d in directions.items() if d != 1.0}
        if wrong:
            raise ValueError(
                f"joint_directions must be the identity here, got {wrong}. "
                "This package drives the arm in motor space, which is the "
                "URDF's space; a non-unit direction mirrors or rescales every "
                "command on that joint. If an upgraded plugin restored the "
                "vendor leader-arm mapping, re-read follower.py before running "
                "anything.")

    def connect(self, calibrate: bool = True) -> None:
        """Check the mapping, then connect as usual.

        Checked here because connect() is the one gate every entry point passes
        through - the servers, the examples and the LeRobot CLI alike - and
        because it runs before the bus is opened, so a bad config costs nothing.
        """
        self.check_joint_directions()
        super().connect(calibrate)

    def hold_action(self, observation=None):
        """Snapshot the current pose as an action that commands staying there.

        CALL ONCE AND REUSE THE RESULT. Calling this every control cycle is not
        a hold - it is a free-running integrator. The setpoint would be re-read
        from the measured position each time, so any drift from feedforward
        error moves the joint, the target follows it, and nothing ever opposes
        it. Measured at ~1.5 deg/s of unopposed drift on this arm.

        Correct:
            target = robot.hold_action()        # once
            while ...: robot.send_action(target)

        Wrong:
            while ...: robot.send_action(robot.hold_action())
        """
        obs = observation if observation is not None else self.get_observation()
        return {k: v for k, v in obs.items() if k.endswith(".pos")}

    def skip_safe_zero_on_disconnect(self) -> None:
        """Stop `disconnect()` driving the arm to zero on the way out.

        The base class's disconnect() calls safe_zero(), which interpolates all
        six arm joints to MOTOR ZERO in two stages - stage 2 being CAN ids 2
        and 3, i.e. shoulder_lift and elbow_flex. On this arm zero is the LOWER
        HARD STOP for exactly those two joints, so a plain disconnect lowers
        the arm onto its stops.

        That is a defensible park when a human asked for one. It is not
        defensible on SIGINT: the one thing an interrupt must never do is start
        a motion. Callers shutting down in response to a signal call this
        first, so the arm holds position and then goes limp.

        This was ALREADY the effective behaviour, but only by accident: torque
        was disabled first, disabled RobStride motors emit no feedback frames,
        safe_zero's position read therefore raised, and disconnect() swallowed
        it with logger.exception. Relying on that meant a plugin update which
        merely fixed the exception path would silently restore the motion.

        Reaches into a private attribute of the upstream plugin because there
        is no public knob - `disable_torque_on_disconnect` does not gate
        safe_zero. The guard below turns a rename or removal into a loud
        failure here, rather than an unannounced move of the arm.
        """
        if not hasattr(self, "_skip_safe_zero_on_disconnect"):
            raise AttributeError(
                "The upstream follower no longer has "
                "_skip_safe_zero_on_disconnect, so disconnect() may once again "
                "drive the arm to zero - which on this arm is two hard stops. "
                "Re-read seeed_b601_follower.disconnect() before running.")
        self._skip_safe_zero_on_disconnect = True


if __name__ == "__main__":
    # Offline check, no hardware needed: what the vendor mapping would do to a
    # pose, and that this package's config does nothing to it at all.
    vendor = SeeedB601RSFollowerConfig(id="check", port="can0").joint_directions
    ours = _identity_directions()

    pose = {"shoulder_pan": 10.0, "shoulder_lift": 20.0, "elbow_flex": 30.0,
            "wrist_flex": -15.0, "wrist_yaw": 5.0, "wrist_roll": 40.0,
            "gripper": 60.0}

    print(f"{'joint':<15}{'vendor':>8}{'ours':>6}{'motor':>10}"
          f"{'vendor cmd':>12}{'our cmd':>10}")
    worst = 0.0
    for name, motor_deg in pose.items():
        vendor_cmd = motor_deg * vendor[name]
        our_cmd = motor_deg * ours[name]
        worst = max(worst, abs(vendor_cmd - motor_deg))
        flag = "" if abs(vendor_cmd - motor_deg) < 1e-9 else "  <-- mirrored"
        print(f"{name:<15}{vendor[name]:>8.1f}{ours[name]:>6.1f}"
              f"{motor_deg:>10.2f}{vendor_cmd:>12.2f}{our_cmd:>10.2f}{flag}")
        assert abs(our_cmd - motor_deg) < 1e-9, f"not the identity for {name}"

    print(f"\nours: exact for all {len(pose)} joints")
    print(f"vendor mapping: worst error {worst:.2f} deg")
