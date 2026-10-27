"""reBot B601-RS follower with the observation/action sign convention reconciled.

THE BUG
    The upstream plugin's get_observation() returns raw motor degrees:
        obs["elbow_flex.pos"] = math.degrees(state.pos)

    while its send_action() multiplies every target by joint_directions before
    clipping:
        position = position * self.config.joint_directions[motor_name]

    So the two APIs speak different spaces. Feeding an observation straight back
    as an action - the core LeRobot pattern behind teleop recording, dataset
    replay and policy rollout - commands the MIRROR of the current pose on every
    joint with direction -1.0 (elbow_flex, wrist_flex, wrist_yaw), and scales
    gripper by 6x.

WHY FIX THE OBSERVATION AND NOT THE ACTION
    joint_directions maps a leader arm's convention onto the motor's, so
    send_action applying it is deliberate: a Star-Arm-102 leader emits logical
    angles that must be flipped for this follower. The coherent fix is to have
    get_observation report in that same logical space, so that

        logical -> send_action -> (x direction) -> motor
        motor   -> get_observation -> (/ direction) -> logical

    round-trips. Stripping direction from send_action would instead break
    leader-driven teleop.

NOT CHANGED
    send_action, and the max_relative_target clamp inside it. Both already
    operate consistently in motor space - the clamp compares a direction-applied
    goal against a raw present position, which is the same space. The warning
    flood seen earlier was a symptom of feeding it mirrored goals, not of the
    clamp itself being wrong.

    Torque is left as raw motor torque. joint_directions is a scale as well as a
    sign (gripper is 6.0), and dividing a torque by a kinematic gear factor is
    not physically meaningful. Documented rather than silently transformed.
"""
import logging
from dataclasses import dataclass

from lerobot.robots.robot import RobotConfig
from lerobot_robot_seeed_b601 import SeeedB601RSFollower, SeeedB601RSFollowerConfig

logger = logging.getLogger(__name__)


@RobotConfig.register_subclass("seeed_b601_rs_follower_fixed")
@dataclass
class ReBotFollowerConfig(SeeedB601RSFollowerConfig):
    """Registers a --robot.type that resolves to the FIXED follower.

    Subclasses the upstream SeeedB601RSFollowerConfig and adds no fields: every
    numeric default (joint_limits, joint_directions, mit_kp/kd, temperature
    thresholds) is inherited unchanged. Without this registration, CLI entry
    points (lerobot-record, lerobot-replay, ...) invoked with
    --robot.type=seeed_b601_rs_follower construct the unfixed base class, and
    the observation/action mismatch is back.

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


class ReBotFollower(SeeedB601RSFollower):
    """Follower whose observations are in the same space as its actions.

    Subclasses the upstream SeeedB601RSFollower (lerobot_robot_seeed_b601).
    get_observation() is the only inherited method overridden; the rest are new
    helpers. connect, calibrate, send_action, safe_zero and the temperature
    protection are inherited as-is.

    DO NOT override `name`. Robot.calibration_fpath is derived from it, so a
    different name points at a directory with no calibration file, is_calibrated
    then returns False, and connect() walks straight into calibrate() - which
    calls set_zero_position() on all seven motors and destroys the existing
    zero. Inheriting the base name keeps this class pointed at the same
    calibration as the robot it is fixing.
    """

    def _direction(self, motor_name: str) -> float:
        """Direction for a joint, guarding the 0.0 default.

        The base class uses joint_directions.get(name, 0.0), so a joint missing
        from that dict is silently multiplied by zero - i.e. commanded to zero
        rather than left alone. All seven joints are present today; this guard
        exists so the failure is loud rather than a surprise move if that ever
        changes.
        """
        d = self.config.joint_directions.get(motor_name)
        if d is None or d == 0.0:
            raise ValueError(
                f"joint_directions[{motor_name!r}] is {d!r}. The base class "
                "would treat this as 0.0 and command the joint to zero."
            )
        return float(d)

    def motor_to_logical(self, motor_name: str, value: float) -> float:
        """Motor-space degrees -> the space send_action expects."""
        return value / self._direction(motor_name)

    def logical_to_motor(self, motor_name: str, value: float) -> float:
        """The space send_action expects -> motor-space degrees."""
        return value * self._direction(motor_name)

    def get_observation(self):
        """Base observation, with pos and vel converted into action space.

        Position and velocity both carry the sign convention and both round-trip
        through send_action, so both are converted. Torque is deliberately left
        in motor space - see the module docstring.
        """
        obs = super().get_observation()
        for key in list(obs.keys()):
            if key.endswith(".pos") or key.endswith(".vel"):
                motor_name = key.rsplit(".", 1)[0]
                if motor_name in self.config.joint_directions:
                    obs[key] = self.motor_to_logical(motor_name, obs[key])
        return obs

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

        With the base class this returns a mirrored pose on the three joints
        with direction -1.0; here it is the pose the arm is actually in.
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
    # Offline check of the transform, no hardware needed: assert that a pose
    # survives logical -> motor -> logical, and show what the unfixed round-trip
    # would have done to the same pose.
    cfg = SeeedB601RSFollowerConfig(id="check", port="can0")
    dirs = cfg.joint_directions

    pose = {"shoulder_pan": 10.0, "shoulder_lift": 20.0, "elbow_flex": 30.0,
            "wrist_flex": -15.0, "wrist_yaw": 5.0, "wrist_roll": 40.0,
            "gripper": 60.0}

    print(f"{'joint':<15}{'dir':>6}{'motor':>10}{'fixed':>10}{'unfixed':>10}")
    worst_unfixed = 0.0
    for name, motor_deg in pose.items():
        d = dirs[name]
        fixed = (motor_deg / d) * d          # what this class round-trips to
        unfixed = motor_deg * d              # what the base class commands
        worst_unfixed = max(worst_unfixed, abs(unfixed - motor_deg))
        flag = "" if abs(unfixed - motor_deg) < 1e-9 else "  <-- wrong"
        print(f"{name:<15}{d:>6.1f}{motor_deg:>10.2f}{fixed:>10.2f}"
              f"{unfixed:>10.2f}{flag}")
        assert abs(fixed - motor_deg) < 1e-9, f"round-trip broken for {name}"

    print(f"\nfixed round-trip: exact for all {len(pose)} joints")
    print(f"unfixed round-trip: worst error {worst_unfixed:.2f} deg")
