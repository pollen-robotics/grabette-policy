"""Shared core for Gripette-equipped arms: proto, kinematics, rotation.

No MuJoCo, no OpenCV, and placo only via the [kinematics] extra. Both the
simulation packages and the real-hardware drivers depend on this, and the
drivers run on robots where the simulation stack is neither wanted nor
installable.

Arm-specific packages supply an `ArmSpec` (URDF, joint names, masked DOFs) and
get the same FK/IK and the same gRPC contract, which is what lets one trained
policy drive either arm.

Re-exports are lazy (PEP 562) so that importing `proto` or `rotation` does not
pull in placo.
"""

import importlib

# Public name -> submodule that defines it.
_EXPORTS = {
    "ArmSpec": ".kinematics",
    "Kinematics": ".kinematics",
    "CAMERA_FRAME": ".kinematics",
    "GRIPPER_FRAME": ".kinematics",
    "OAKL_FRAME": ".kinematics",
    "CONTROL_FRAME": ".kinematics",
    "IKFeasibilityChecker": ".ik_feasibility",
    "PoseFeasibility": ".ik_feasibility",
    "TrajectoryFeasibility": ".ik_feasibility",
    "RejectionSamplingStats": ".ik_feasibility",
    "rotation_matrix_to_6d": ".rotation",
    "rotation_6d_to_matrix": ".rotation",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
