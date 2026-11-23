# gripette-sim

The MuJoCo half of the Gripette stack, shared by every arm that wears the
gripper.

`gripette-core` holds what a *client* and a *real robot* need: the gRPC
contract, the kinematics, the arm service. Its pyproject refuses mujoco and
opencv on purpose — it is imported by the hardware drivers, which run on
machines with no simulator. This package is the other half, and the split is
why both can exist.

| module | what it is | arm-specific? |
|---|---|---|
| `camera.py` | the Gripette camera: KannalaBrandt8 fisheye, real calibration | no — a property of the gripper |
| `gripper_service.py` | `GripperService` over a simulated gripper, same API as the real one | no |
| `simulation.py` | MuJoCo wrapper: stepping, joint control, readback, camera render | only via its scene and joint names, both injected |

An arm supplies three things and gets the rest: a scene XML, its arm joint
names, and an `ArmInterface` for `gripette_core.arm_service.ArmServicer`.

## Why `simulation.py` is here rather than copied per arm

It was OpenArm's, and all that was OpenArm about it was the default scene path
and the joint-name list. The XML surgery in `_load_model` — inlining the robot
include so MuJoCo resolves `meshdir`, injecting a `gripette_cam` at the
calibrated `camera` site — is the same work for any export from the same
toolchain, and getting it subtly different per arm would mean the two robots
render through different cameras while appearing to share one calibration.
