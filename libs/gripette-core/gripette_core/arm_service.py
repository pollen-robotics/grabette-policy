"""The arm-side gRPC service, shared by every Gripette-equipped arm.

This is the behaviour a policy actually talks to: the Cartesian integrator,
the joint-space setpoint interpolator, the IK-jump watchdog, the target-lead
(contact) guard and the reset ramp. It was written and tuned against the
OpenArm on real hardware; none of it is OpenArm-specific, and all of it is
exactly what a second arm must reproduce bit-for-bit if one trained policy is
to drive either. Hence one copy, here, rather than one per robot.

An arm supplies two things:

  * an `ArmInterface` — three methods over whatever bus it has, and
  * a `Kinematics` bound to its `ArmSpec` (which carries the joint names and
    therefore the joint COUNT: 7 on the OpenArm, 6 on the reBot).

plus its own home pose, and optionally a `joint_gate` - a callable that can
veto an IK solution before it is commanded, which is how an arm declares the
obstacles it must not drive into. Everything else is common.

Deliberately free of placo and mujoco at import time: `Kinematics` is only
referenced as a type annotation, so a client can import the servicer without
the solver stack present.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Protocol

import numpy as np

from gripette_core.proto import arm_pb2, arm_pb2_grpc
from gripette_core.rotation import rotation_6d_to_matrix, rotation_matrix_to_6d

if TYPE_CHECKING:
    from gripette_core.kinematics import Kinematics

logger = logging.getLogger(__name__)

# Reset interpolation: move from current to home over this many seconds.
RESET_DURATION_S = 3.0
RESET_HZ = 50


class ArmInterface(Protocol):
    """What the servicer needs from a physical (or simulated) arm.

    Three methods, all in RADIANS and in the order given by the arm's
    `ArmSpec.joint_names`. Unit conversion, joint-name mapping, caching and
    bus retries are the implementation's business, not the servicer's.
    """

    def get_positions(self) -> np.ndarray:
        """Current joint positions (rad), in ArmSpec order."""
        ...

    def send_command_rad(self, joint_angles_rad: np.ndarray) -> None:
        """Command joint positions (rad), in ArmSpec order."""
        ...

    def set_torque(self, enable: bool) -> None:
        """Enable or disable torque on every arm motor.

        Disabling means the motors freewheel and the arm FALLS under gravity.
        Implementations are expected to make this robust against a single lost
        frame — a motor that silently keeps torque after a shutdown is a real
        hazard, observed on the OpenArm.
        """
        ...


class ArmServicer(arm_pb2_grpc.ArmServiceServicer):
    """Implements the same delta-target accumulation pattern as the simulator.

    SendCartesianDelta maintains an internal (_target_pos, _target_r6d). Deltas
    accumulate on that target (not on the current FK pose), avoiding drift from
    mechanical tracking errors. After IK, the target is re-synced to the FK of
    the commanded joints so it stays within a reachable neighborhood.

    Joint-space setpoint interpolation: SendCartesianDelta only computes IK and
    writes the target into a slot. A background thread at interp_hz drives the
    motors using an exponential approach toward that slot, filling in the time
    between sparse policy commands with a smooth joint trajectory. This is the
    standard fix for "smooth Cartesian in -> jerky motors out" with stiff MIT
    gains.
    """

    def __init__(
        self,
        arm: ArmInterface,
        kin: "Kinematics",
        home_joints: np.ndarray,
        start_time: float,
        interp_hz: float = 50.0,
        interp_alpha: float = 0.3,
        max_ik_jump_deg: float = 15.0,
        max_ik_jump_violations: int = 2,
        max_target_lead_mm: float = 50.0,
        joint_gate=None,
    ):
        self._arm = arm
        self._kin = kin
        # Optional veto on an IK solution: f(joint_angles_rad) -> reason | None.
        # A reason string rejects the command; None accepts it. See the call
        # site in SendCartesianDelta for why it is a callable and not geometry.
        self._joint_gate = joint_gate
        self._home_joints = np.asarray(home_joints, dtype=np.float64)
        self._start_time = start_time
        if len(self._home_joints) != len(kin.joint_names):
            raise ValueError(
                f"home_joints has {len(self._home_joints)} entries but the arm "
                f"has {len(kin.joint_names)} joints")
        self._cmd_lock = threading.Lock()

        # Setpoint interpolator state.
        # `_latest_target_joints` is an atomic slot (reference assignment is
        # atomic under the GIL); `_current_cmd_joints` is only touched by the
        # interp thread, so it needs no lock.
        self._interp_hz = interp_hz
        self._interp_alpha = interp_alpha
        self._latest_target_joints: np.ndarray | None = None
        self._current_cmd_joints: np.ndarray | None = None
        self._interp_enabled = True
        self._interp_running = True

        # IK-jump watchdog state. The threshold is per-joint per-step. A
        # singularity-driven branch flip on a 50 Hz policy loop usually shows
        # up as 30°+ on a single joint in one step, so 15° is a comfortable
        # margin between "normal motion" and "abrupt flip" at typical speeds.
        # Set <= 0 to disable.
        self._max_ik_jump_rad = float(np.deg2rad(max_ik_jump_deg))
        self._max_ik_jump_violations = int(max_ik_jump_violations)
        self._ik_jump_violations = 0

        # Target-runaway ("rubber band") guard. The integrator accumulates
        # deltas OPEN-LOOP on the target — right for tracking accuracy, but
        # under CONTACT (arm pressing the table/object, motors stalled by the
        # per-write safety clamp) the target keeps marching while the arm
        # doesn't, so (a) press torque grows unbounded and (b) a later retreat
        # command must first unwind the accumulated excess before the arm
        # physically moves — corrections look ignored. Cap how far the target
        # position may LEAD the measured FK pose. <= 0 disables.
        self._max_target_lead_m = float(max_target_lead_mm) / 1000.0
        self._lead_clamps = 0
        # Windowed trip history (1=trip, 0=clean) — a dithering policy
        # interleaves lead-increasing and lead-reducing commands, so a
        # CONSECUTIVE counter never fires while the arm stays pressed at cap
        # depth. Relax triggers on >=3 trips within the last 8 commands.
        self._lead_trip_window = []

        self._sync_target_from_robot()

        self._interp_thread = threading.Thread(target=self._interp_loop, name="ArmInterpLoop", daemon=True)
        self._interp_thread.start()
        logger.info(
            f"Joint interpolator ON: {self._interp_hz:.0f} Hz, alpha={self._interp_alpha:.2f} "
            f"(e-folding time ~{1000.0 / (self._interp_alpha * self._interp_hz):.0f} ms)"
        )

    def stop(self):
        self._interp_running = False
        if self._interp_thread.is_alive():
            self._interp_thread.join(timeout=2.0)

    def _sync_target_from_robot(self):
        """Reset internal target from the current robot FK pose."""
        arm_joints = self._arm.get_positions()
        tf = self._kin.forward(arm_joints)
        self._target_pos = tf[:3, 3].copy()
        self._target_r6d = rotation_matrix_to_6d(tf[:3, :3]).copy()

    def _interp_loop(self):
        """Drive motors at interp_hz with exponential approach toward latest target."""
        period = 1.0 / self._interp_hz
        while self._interp_running:
            tick = time.monotonic()
            if self._interp_enabled:
                target = self._latest_target_joints  # atomic read (ref assignment)
                if target is not None:
                    try:
                        if self._current_cmd_joints is None:
                            # Seeding from the measured pose can RAISE: a driver
                            # may refuse to report a pose when a motor sent no
                            # feedback, rather than substituting 0.0 (which on
                            # the reBot is a hard stop for two joints). Inside
                            # the try because nothing joins this thread — an
                            # escaping exception would stop the arm being driven
                            # with no error surfacing anywhere.
                            self._current_cmd_joints = self._arm.get_positions()
                        # next = cur + alpha * (target - cur)
                        self._current_cmd_joints = self._current_cmd_joints + self._interp_alpha * (
                            target - self._current_cmd_joints
                        )
                        self._arm.send_command_rad(self._current_cmd_joints)
                    except Exception as e:
                        logger.warning(f"Interp tick skipped: {e}")
            elapsed = time.monotonic() - tick
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def SendCartesianDelta(self, request, context):
        try:
            delta_pos = np.array([request.dx, request.dy, request.dz])
            delta_r6d = np.array(request.dr6d)

            if len(delta_r6d) != 6:
                return arm_pb2.ArmCommandResponse(
                    success=False, error=f"dr6d must have 6 values, got {len(delta_r6d)}"
                )

            with self._cmd_lock:
                # Once the IK-jump watchdog latched (or torque is off), the
                # interpolator is disabled and motion is frozen until Reset.
                # Reject outright instead of running IK: solving against the
                # stale pre-latch joint solution just re-trips the watchdog
                # with confusing repeated "would change by N°" errors while
                # the arm isn't moving at all.
                if not self._interp_enabled:
                    return arm_pb2.ArmCommandResponse(
                        success=False,
                        error="motion frozen (IK-jump watchdog latch or torque off) — call Reset to re-home and re-arm",
                    )

                # Camera-LOCAL frame deltas (Stage-6 convention) applied to
                # the INTEGRATOR target. See sim arm_servicer.SendCartesianDelta
                # for the full math explanation. Crucial: deltas are applied
                # to (_target_pos, _target_r6d), not to the FK-read pose, so
                # the commanded trajectory exactly reproduces the dataset
                # trajectory regardless of arm tracking error.
                R_target = rotation_6d_to_matrix(self._target_r6d)

                prev_target_pos = self._target_pos.copy()
                delta_pos_world = R_target @ delta_pos
                self._target_pos = self._target_pos + delta_pos_world

                R_delta = rotation_6d_to_matrix(delta_r6d)
                R_target_new = R_target @ R_delta
                self._target_r6d = rotation_matrix_to_6d(R_target_new).copy()

                # Target-runaway guard, TRIP-ONLY: if this delta would put the
                # target more than the cap ahead of the MEASURED pose (arm in
                # contact / stalled while the open-loop integrator marches on),
                # REJECT the command and roll it back — exactly like the
                # IK-jump watchdog. Never silently modify the target: pulling
                # it toward the measured pose creates a delayed-feedback loop
                # on the reference that corrupts the whole trajectory
                # (measured to drag the arm downward with gravity sag).
                # Deltas that REDUCE the lead (retreats) always pass, so
                # contact stays bounded and corrections act immediately.
                if self._max_target_lead_m > 0:
                    fk_meas = self._kin.forward(self._arm.get_positions())
                    lead_prev = float(np.linalg.norm(prev_target_pos - fk_meas[:3, 3]))
                    lead_new = float(np.linalg.norm(self._target_pos - fk_meas[:3, 3]))
                    if lead_new > self._max_target_lead_m and lead_new > lead_prev:
                        self._lead_clamps += 1
                        self._lead_trip_window = (self._lead_trip_window + [1])[-8:]
                        if self._lead_clamps <= 3 or self._lead_clamps % 50 == 0:
                            logger.warning(
                                f"Target-lead trip #{self._lead_clamps}: rejecting delta — "
                                f"target would be {lead_new * 1000:.0f} mm ahead of the "
                                f"measured pose (cap {self._max_target_lead_m * 1000:.0f} mm). "
                                f"Arm in CONTACT or not tracking; only lead-reducing "
                                f"commands accepted until it catches up."
                            )
                        # RELAX-ON-CONTACT: N consecutive trips = confirmed
                        # contact (not a transient). Pull the pending target
                        # back to the measured pose so the interpolator stops
                        # PRESSING at cap depth — force drops to ~zero, motion
                        # resumes as soon as commands pull away. Event-
                        # triggered only (never fires in free motion), unlike
                        # the removed continuous rubber-band.
                        if sum(self._lead_trip_window) >= 3:
                            self._sync_target_from_robot()
                            self._latest_target_joints = self._arm.get_positions().copy()
                            self._current_cmd_joints = self._latest_target_joints.copy()
                            logger.warning(
                                "CONTACT confirmed (>=3 lead trips in last 8 commands) — "
                                "target relaxed to the measured pose; press force released."
                            )
                            self._lead_trip_window = []
                            return arm_pb2.ArmCommandResponse(
                                success=False,
                                error="contact: target relaxed to measured pose",
                            )
                        self._target_pos = prev_target_pos
                        self._target_r6d = rotation_matrix_to_6d(R_target).copy()
                        return arm_pb2.ArmCommandResponse(
                            success=False,
                            error=f"target lead {lead_new * 1000:.0f}mm > "
                                  f"{self._max_target_lead_m * 1000:.0f}mm cap (contact?)",
                        )
                    self._lead_trip_window = (self._lead_trip_window + [0])[-8:]

                target_tf = np.eye(4)
                target_tf[:3, :3] = R_target_new
                target_tf[:3, 3] = self._target_pos

                # IK seed: prefer the LAST COMMANDED joint config (continuity with
                # the previous IK solution) over the MEASURED joint config (which
                # lags behind by the interpolator e-folding time, ~67 ms at the
                # 50 Hz / alpha=0.3 default). Seeding from measured joints on real
                # makes Placo's frame-task (position weight 100x orientation) flip
                # the wrist toward whichever yaw value matches the lagged seed,
                # which compounds into visible yaw drift edge-by-edge in
                # cartesian_square. Sim doesn't hit this because MuJoCo position
                # controllers track commanded joints tightly, so measured ≈
                # commanded and the seed is already smooth.
                if self._latest_target_joints is not None:
                    ik_seed = self._latest_target_joints
                else:
                    ik_seed = self._arm.get_positions()
                try:
                    target_joints = self._kin.inverse(target_tf, current_joint_positions=ik_seed)
                except Exception as e:
                    # IK blew up (e.g. QP NaN near a singularity). Roll the
                    # delta back — without this the failed delta stays in the
                    # integrator and the target drifts away from every
                    # subsequent solvable command.
                    self._target_pos = prev_target_pos
                    self._target_r6d = rotation_matrix_to_6d(R_target).copy()
                    logger.error(f"IK solve failed ({e}); delta rolled back.")
                    return arm_pb2.ArmCommandResponse(success=False, error=f"IK failed: {e}")

                # IK-jump watchdog: refuse the update if any joint would change
                # by more than `_max_ik_jump_rad` in this single step. Singular
                # configurations (typical wrist alignment) can make Placo flip
                # to a different IK branch within a single inference step, and
                # the per-joint motor clamp + interpolator only soften the speed
                # of that flip, they don't prevent it. Comparing the new
                # solution to the last accepted one catches the branch flip
                # directly and trips the integrator before it propagates.
                if self._latest_target_joints is not None and self._max_ik_jump_rad > 0:
                    delta_joints = target_joints - self._latest_target_joints
                    max_jump = float(np.max(np.abs(delta_joints)))
                    if max_jump > self._max_ik_jump_rad:
                        bad_idx = int(np.argmax(np.abs(delta_joints)))
                        bad_name = self._kin.joint_names[bad_idx]
                        logger.error(
                            f"IK-jump watchdog tripped: joint '{bad_name}' "
                            f"(idx {bad_idx}) would change by "
                            f"{np.rad2deg(delta_joints[bad_idx]):+.1f}° in one step "
                            f"(limit {np.rad2deg(self._max_ik_jump_rad):.1f}°). "
                            f"Rejecting command; integrator NOT updated. "
                            f"Likely singularity branch flip — re-home or "
                            f"raise --max_ik_jump_deg if intentional."
                        )
                        self._ik_jump_violations += 1
                        if self._ik_jump_violations >= self._max_ik_jump_violations:
                            logger.error(
                                f"{self._ik_jump_violations} consecutive "
                                f"IK-jump violations — disabling interpolator "
                                f"and reverting integrator target to last "
                                f"FK pose for safety."
                            )
                            self._interp_enabled = False
                            self._sync_target_from_robot()
                        # Undo this Cartesian delta on the integrator so we don't
                        # leak it into the next step's cumulative target.
                        self._target_pos -= delta_pos_world
                        self._target_r6d = rotation_matrix_to_6d(R_target).copy()
                        return arm_pb2.ArmCommandResponse(
                            success=False,
                            error=f"IK jump on '{bad_name}': "
                                  f"{np.rad2deg(delta_joints[bad_idx]):+.1f}°",
                        )
                    # Healthy step — reset the consecutive-violation counter.
                    self._ik_jump_violations = 0

                # Joint gate: an arm-supplied veto on the solution itself,
                # rejected with the same rollback as a jump. The IK-jump
                # watchdog asks "did the solver flip branch?"; this asks "is
                # the place it wants to go allowed?" — which is where a
                # clearance check against a table, a pedestal or a fixture
                # belongs. Kept as a plain callable so the obstacle geometry
                # (MuJoCo) stays out of this package, which the real-hardware
                # drivers import; see gripette_sim.clearance for the
                # implementation that supplies one.
                if self._joint_gate is not None:
                    reason = self._joint_gate(target_joints)
                    if reason:
                        logger.error(
                            f"joint gate rejected the command: {reason}. "
                            f"Integrator NOT updated.")
                        self._target_pos -= delta_pos_world
                        self._target_r6d = rotation_matrix_to_6d(R_target).copy()
                        return arm_pb2.ArmCommandResponse(
                            success=False, error=f"blocked: {reason}")

                # Re-sync the integrator to what the accepted solution ACTUALLY
                # achieves (the class docstring always promised this; it was
                # never implemented). Exact IK -> FK(IK(target)) == target and
                # this is a no-op, so faithful replay is unchanged. Compromised
                # IK (joint limits, singular region, solver tolerance) -> the
                # target stays on the reachable manifold instead of marching
                # open-loop into unreachable space — measured as 70-120 mm of
                # cmd-vs-meas divergence pinned at the trip-guard cap. Uses
                # COMMANDED joints (deterministic), never measured ones — no
                # feedback of tracking error/gravity sag into the reference.
                fk_cmd = self._kin.forward(target_joints)
                self._target_pos = fk_cmd[:3, 3].copy()
                self._target_r6d = rotation_matrix_to_6d(fk_cmd[:3, :3]).copy()

                # Hand off to the interpolator — no direct motor write.
                self._latest_target_joints = target_joints.copy()

            return arm_pb2.ArmCommandResponse(success=True)

        except Exception as e:
            logger.exception("SendCartesianDelta failed")
            return arm_pb2.ArmCommandResponse(success=False, error=str(e))

    def GetArmState(self, request, context):
        """Return the current arm state (FK of measured joints)."""
        arm_joints = self._arm.get_positions()
        tf = self._kin.forward(arm_joints)
        pos = tf[:3, 3]
        r6d = rotation_matrix_to_6d(tf[:3, :3])
        return arm_pb2.ArmState(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            r6d=r6d.tolist(),
            joint_positions=arm_joints.tolist(),
        )

    def Reset(self, request, context):
        """Move the arm smoothly to the home (or specified) joint configuration.

        Pauses the setpoint interpolator for the duration of the linear ramp,
        then resyncs its state to the final pose so it resumes cleanly.

        Cube randomization is a no-op (no physical cube); dummy cube coords are
        returned for API compatibility with the simulator.
        """
        try:
            # Joint count follows the arm (7 on OpenArm, 6 on reBot).
            if len(request.joint_positions) == len(self._kin.joint_names):
                target_joints = np.array(request.joint_positions, dtype=np.float64)
            else:
                target_joints = self._home_joints.copy()

            self._interp_enabled = False
            try:
                with self._cmd_lock:
                    start_joints = self._arm.get_positions()

                    num_steps = int(RESET_DURATION_S * RESET_HZ)
                    dt = 1.0 / RESET_HZ

                    logger.info(
                        f"Reset: interpolating over {RESET_DURATION_S}s "
                        f"from {start_joints.round(3).tolist()} to {target_joints.round(3).tolist()}"
                    )

                    for i in range(1, num_steps + 1):
                        alpha = i / num_steps
                        interp = start_joints * (1 - alpha) + target_joints * alpha
                        self._arm.send_command_rad(interp)
                        time.sleep(dt)

                    # Resync interp state so it picks up from here without a jump.
                    self._current_cmd_joints = target_joints.copy()
                    self._latest_target_joints = target_joints.copy()
                    self._sync_target_from_robot()
                    # Re-homing un-latches the IK-jump watchdog: fresh count.
                    self._ik_jump_violations = 0
            finally:
                self._interp_enabled = True

            return arm_pb2.ResetResponse(
                success=True,
                cube_x=0.0,
                cube_y=0.0,
                cube_z=0.0,
                error="cube randomization skipped (real robot)",
            )

        except Exception as e:
            logger.exception("Reset failed")
            return arm_pb2.ResetResponse(success=False, error=str(e))

    def GetSuccessStatus(self, request, context):
        """No cube tracking on a real robot — always returns goal_reached=False."""
        return arm_pb2.SuccessStatusResponse(goal_reached=False, cube_displacement=0.0)

    def SetTorque(self, request, context):
        """Enable/disable motor torque on the whole arm.

        Disable: pauses the setpoint interpolator FIRST (so it stops streaming
        MIT commands), clears its targets, then disables torque on the bus — the
        motors freewheel and the arm FALLS under gravity.

        Enable: re-enables torque and HOLDS the current measured pose (the
        interpolator target is set to the measured joints), so a hand-placed
        arm stays exactly where you put it. The Cartesian integrator is
        resynced to the same pose, so the next delta command continues from
        there with no jump. (Previous behavior — enabled with NO target —
        meant MIT control at zero torque: the arm felt *freer* than when
        disabled, and FELL when released. Counterintuitive and unsafe.)
        """
        try:
            with self._cmd_lock:
                if request.enable:
                    self._arm.set_torque(True)
                    meas = self._arm.get_positions()
                    self._current_cmd_joints = meas.copy()
                    self._latest_target_joints = meas.copy()
                    self._sync_target_from_robot()
                    self._ik_jump_violations = 0
                    self._interp_enabled = True
                    logger.info("Torque ENABLED — holding the current pose.")
                else:
                    self._interp_enabled = False
                    self._latest_target_joints = None
                    self._current_cmd_joints = None
                    self._arm.set_torque(False)
                    logger.warning("Torque DISABLED — motors freewheeling, arm falls under gravity.")
            return arm_pb2.ArmCommandResponse(success=True)
        except Exception as e:
            logger.exception("SetTorque failed")
            return arm_pb2.ArmCommandResponse(success=False, error=str(e))

    def Ping(self, request, context):
        uptime = time.monotonic() - self._start_time
        return arm_pb2.ArmPingResponse(status="ok", uptime_seconds=uptime)
