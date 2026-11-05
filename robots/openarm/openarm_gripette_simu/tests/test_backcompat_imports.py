"""The shared core must not have moved anything out from under OpenArm.

proto / kinematics / rotation / ik_feasibility were lifted into
`libs/gripette-core`, leaving re-export shims in this package. Those shims are
the whole reason the move was safe, and nothing else tests them — so this does.

Lives in the SIM package, not gripette_core: these are tests OF the shims, and
the shims are here. Putting them in gripette_core would mean making the sim
package a test dependency of the very package that exists to avoid depending
on it.
"""

import numpy as np
import pytest


def test_kinematics_import_paths_agree():
    from gripette_core.kinematics import Kinematics as Core
    from openarm_gripette_simu import Kinematics as ViaPackage
    from openarm_gripette_simu.kinematics import Kinematics as ViaModule

    assert ViaPackage is ViaModule
    assert issubclass(ViaPackage, Core)


def test_openarm_kinematics_still_takes_no_arguments():
    """Almost every call site is a bare `Kinematics()`; keep it that way."""
    from openarm_gripette_simu import Kinematics

    assert len(Kinematics().joint_names) == 7
    # The eval server's stricter weighting must stay constructible too.
    assert Kinematics(orientation_weight=10.0) is not None
    # And the positional form, which grpc_server_real could start using.
    assert Kinematics(None, 50.0, 5.0) is not None


def test_frame_constants_are_shared_not_copied():
    from gripette_core import kinematics as core
    from openarm_gripette_simu import kinematics as simu

    for name in ("CAMERA_FRAME", "GRIPPER_FRAME", "OAKL_FRAME", "CONTROL_FRAME"):
        assert getattr(simu, name) == getattr(core, name), name
    # oak_l is THE control frame; a silent change breaks every recorded
    # dataset, so pin the value and not merely the agreement.
    assert core.CONTROL_FRAME == "oak_l"


def test_arm_joint_names_stay_a_seven_element_list():
    from openarm_gripette_simu.kinematics import ARM_JOINT_NAMES

    assert isinstance(ARM_JOINT_NAMES, list)
    assert ARM_JOINT_NAMES == [
        "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_elbow",
        "r_wrist_yaw", "r_wrist_roll", "r_wrist_pitch",
    ]


def test_masked_dofs_are_declared():
    """Without masking, IK can 'reach' a target by closing the fingers."""
    from openarm_gripette_simu.kinematics import OPENARM_SPEC

    assert set(OPENARM_SPEC.masked_dofs) == {
        "proximal", "distal", "r_wrist_roll_mimic",
    }


@pytest.mark.parametrize("stub", ["arm_pb2", "arm_pb2_grpc",
                                  "gripper_pb2", "gripper_pb2_grpc"])
def test_proto_stubs_are_one_module_not_two(stub):
    import importlib

    core = importlib.import_module(f"gripette_core.proto.{stub}")
    simu = importlib.import_module(f"openarm_gripette_simu.proto.{stub}")
    assert core is simu, (
        f"{stub} resolved to two module objects; a duplicated generated copy "
        "registers twice in protobuf's descriptor pool"
    )


def test_proto_from_import_form_also_works():
    """`from ...proto import arm_pb2` is how every caller actually spells it."""
    from gripette_core.proto import arm_pb2 as core
    from openarm_gripette_simu.proto import arm_pb2 as simu

    assert core is simu
    assert isinstance(simu.ArmState(), core.ArmState)


def test_rotation_and_ik_feasibility_shims_resolve():
    from gripette_core.ik_feasibility import IKFeasibilityChecker as CoreIKF
    from gripette_core.rotation import rotation_matrix_to_6d as core_r
    from openarm_gripette_simu.ik_feasibility import IKFeasibilityChecker as SimuIKF
    from openarm_gripette_simu.rotation import rotation_matrix_to_6d as simu_r

    assert core_r is simu_r
    assert CoreIKF is SimuIKF


def test_fk_still_lands_where_it_used_to():
    """A spot value, so a silently different URDF or joint order is caught."""
    from openarm_gripette_simu import Kinematics

    kin = Kinematics()
    T = kin.forward(np.zeros(7))
    assert T.shape == (4, 4)
    # Round-trip through IK must return to the same pose.
    q = kin.inverse(T, current_joint_positions=np.zeros(7), n_iter=200)
    reached = kin.forward(q)
    assert np.linalg.norm(reached[:3, 3] - T[:3, 3]) < 1e-4
