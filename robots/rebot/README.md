# reBot-Gripette integration

The **reBot-Gripette** is a [reBot B601-RS](https://www.seeedstudio.com/) 6-DoF
desktop arm carrying the Grabette team's
[Gripette](https://github.com/pollen-robotics/grabette/tree/develop/packages/gripette)
gripper — the same end-effector the [OpenArm](../openarm/) runs. It is the
second arm in this repo, and it exists to answer whether a policy trained on
Grabette data generalises across arms, or only works on the one it was
recorded with.

Three sibling packages, mirroring the OpenArm's layout:

| Package | Purpose |
|---------|---------|
| [`rebot_gripette_model/`](rebot_gripette_model/) | URDF + MuJoCo XML + meshes for the B601-RS + Gripette assembly |
| [`rebot_gripette_simu/`](rebot_gripette_simu/) | MuJoCo simulator — physics, camera, grasp scenes, the two gRPC services |
| [`rebot_gripette/`](rebot_gripette/) | **Real-hardware** driver and `ArmService` gRPC server (CAN bus, via upstream LeRobot's Seeed B601-RS plugin) |

## The design claim: sim and real run the same service

Not "equivalent" implementations — **the same objects**. Both servers build
`gripette_core.arm_service.ArmServicer`; one over a CAN-backed `ArmInterface`,
the other over a MuJoCo-backed one. The Cartesian integrator, the setpoint
interpolator, the IK-jump watchdog and the reset ramp are therefore shared, so
a policy that works in sim and fails on the arm fails for reasons about the
robot.

```
Eval / teleop client ─── ArmService (gRPC) ──►  SIM:  rebot_gripette_simu.server
                                                REAL: rebot_gripette.grpc_server_real
                                                   …both: gripette_core.arm_service

Eval / teleop client ─── GripperService ─────►  SIM:  gripette_sim.gripper_service
                                                REAL: gripette's own service (grabette repo)
```

That sharing is why there is almost no reBot-specific simulation code: an
`ArmInterface` over MuJoCo, two class attributes binding the joint names and
default scene, and the scenes themselves.

## Dependency graph

```
rebot_gripette_model   (URDF + meshes, no code deps)
     ▲            ▲
     │            │
rebot_gripette    rebot_gripette_simu
  (CAN driver)      (MuJoCo)
     │                │
     └── gripette-core (proto, kinematics, rotation, ArmService — no mujoco)
                      └── gripette-sim (simulation, camera, gripper service,
                                        clearance, lift tracker — mujoco)
```

`gripette-core` refuses mujoco and opencv on purpose: the real-hardware
drivers import it. Anything needing geometry is injected — the bench guard as
a `joint_gate` callable, the success measurement as an `object_tracker`.

## Quick test: drive a policy at the simulated arm

Two terminals. This is the loop the project is built around, and the numbers
below are measured, not aspirational.

**1. The simulator** (drop `--headless` for the MuJoCo viewer):

```bash
uv run python -m rebot_gripette_simu \
  --home_joints -25.2 127.0 65.7 27.5 -61.8 -30.5 \
  --scene robots/rebot/rebot_gripette_simu/scenes/table_grasp.xml \
  --no_clearance_gate
```

**2. The policy** (`evaluate.py` is arm-agnostic and lives with the OpenArm):

```bash
uv run python robots/openarm/openarm_gripette_simu/examples/evaluate.py \
  --checkpoint SteveNguyen/test_pick_can_200_smooth_trim-best \
  --home_joints -0.4397 2.2162 1.1469 0.4798 -1.0790 -0.5327 \
  --num_episodes 20 --max_steps 250
```

Expect roughly:

```
Success rate:     11/20 (55.0%)
Avg displacement: 38.8 mm
```

Success is the can rising ≥ 50 mm above where `Reset` placed it
(`LIFT_SUCCESS_THRESHOLD`, the figure the dataset was collected under and the
OpenArm scores against).

Three things that will cost you an afternoon if you miss them:

* **The two `--home_joints` are the same pose in different units.** The server
  takes DEGREES, `evaluate.py` takes RADIANS. Crossing them gives a confusing
  result rather than an error.
* **`--no_clearance_gate` is required for a PICK**, and the reason is in
  [`rebot_gripette_simu/README.md`](rebot_gripette_simu/README.md): the gate
  holds every guarded body to one margin, while a grasp needs the fingertips
  below the bench top and the arm links above it. With the gate on, ~236 of
  250 commands are refused and the arm barely moves.
* **The start pose is not interchangeable.** It anchors the whole trajectory
  for a camera-local-delta policy — see below.

## The start pose is the experiment

`--home_joints` is REQUIRED on both servers, with no default, because "home"
is the startup position for an EXPERIMENT rather than a property of the robot,
and `Reset` drives the arm to it.

For these policies the start pose must match the training **range**, not only
the training bearing. The policy commands camera-local deltas, so it executes
an approach of a learned LENGTH rather than servoing to the object: start it
too close and it runs out of approach and closes on nothing.

| start | can at | success |
|---|---|---|
| `-15.1 124.9 88.3 -16.7 -32.8 -36.1` | 172 mm (−1.16σ) | **0/20** |
| `-25.2 127.0 65.7 27.5 -61.8 -30.5` | 232 mm | **11/20** |

Both have near-perfect bearing; the first was chosen by minimising a combined
z-score over bearing AND range, which bought bearing by spending 63 mm of
range. Training started at 235 ± 54.5 mm and a correct grasp puts the can at
104 mm, so the learned approach is ~131 mm — run that from 172 mm and you stop
~60 mm short. Require the range, then optimise bearing, slack and clearance
within it.

## Getting started

- **Run a trained policy in simulation** — [`rebot_gripette_simu/README.md`](rebot_gripette_simu/README.md)
- **Bring up / run the real arm** — [`rebot_gripette/README.md`](rebot_gripette/README.md)
- **Re-export or re-patch the model** — [`rebot_gripette_model/README.md`](rebot_gripette_model/README.md)

The meshes are in Git LFS (121 STLs, 67 MB): install
[git-lfs](https://git-lfs.com) before cloning.

## Known gaps

* **55% is a floor, not a result.** The remaining failures are uncharacterised,
  and the one visual difference still outstanding is the bench: training used a
  300 × 300 mm table floating in a void, this scene a 850 × 360 mm bench on a
  checkerboard floor.
* **No hardware run of a policy yet.** The driver and server are tested; an
  end-to-end `evaluate.py` against the real arm has not been done.
* **Centre of mass is not corrected** — it sits at the motor rather than the
  link centroid, so gravity torque MAGNITUDE is roughly right and its ANGLE
  dependence is not. Do not use this model for gravity compensation.
* **The bench guard cannot express a pick** (see above). Per-body margins are
  the fix, in `obstacles.py` and `clearance.py`.
