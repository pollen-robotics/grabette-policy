"""gripette_core must stay installable and importable without the sim stack.

This is the property the whole package exists for: the real-hardware drivers
import it on robots where mujoco/opencv are not installed, and clients like
examples/teleop_bridge.py install it without placo (the `[kinematics]` extra).

Checked in a SUBPROCESS on purpose. In-process assertions would be worthless:
by the time this runs, pytest has usually imported the sim package already, so
`'mujoco' in sys.modules` would be true for reasons unrelated to gripette_core.
"""

import subprocess
import sys

import pytest

# placo is the interesting one: it is a real dependency of [kinematics], needs
# system urdfdom, and a bare `import gripette_core` must NOT pull it in. That
# laziness is what lets CI's import smoke test include this package, and what
# lets a proto-only consumer skip placo entirely.
_HEAVY = ("mujoco", "cv2", "placo")


def _imports_without(target: str) -> list[str]:
    code = (
        "import sys, importlib;"
        f"importlib.import_module({target!r});"
        f"print(','.join(m for m in {_HEAVY!r} if m in sys.modules))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return [m for m in r.stdout.strip().split(",") if m]


@pytest.mark.parametrize("target", [
    "gripette_core",
    "gripette_core.proto",
    "gripette_core.proto.arm_pb2",
    "gripette_core.proto.gripper_pb2",
    "gripette_core.rotation",
    "gripette_core.ik_feasibility",
])
def test_module_does_not_pull_in_the_heavy_stack(target):
    leaked = _imports_without(target)
    assert not leaked, f"importing {target} pulled in {leaked}"


def test_kinematics_is_the_one_module_allowed_to_need_placo():
    """Stated explicitly so the asymmetry above reads as intent, not oversight.

    Skipped without placo, i.e. in a base+test install — which is the
    environment CI runs this directory in, and the environment the tests above
    exist to protect.
    """
    pytest.importorskip("placo", reason="[kinematics] extra not installed")
    leaked = _imports_without("gripette_core.kinematics")
    assert "placo" in leaked
    assert "mujoco" not in leaked and "cv2" not in leaked


def test_rotation_roundtrip():
    np = pytest.importorskip("numpy")
    from gripette_core.rotation import rotation_6d_to_matrix, rotation_matrix_to_6d

    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(
        rotation_6d_to_matrix(rotation_matrix_to_6d(R)), R, atol=1e-12)


def test_the_proto_source_and_its_stubs_agree_on_the_services():
    """Cheap guard that the committed stubs were not generated from something else.

    Divergence against the GRIPPER's copy of gripper.proto lives in the
    `proto-sync` CI job instead - it has to reach the grabette repo at the
    pinned commit, which a unit test should not do.
    """
    from pathlib import Path

    from gripette_core.proto import arm_pb2, gripper_pb2

    assert "ArmService" in arm_pb2.DESCRIPTOR.services_by_name
    assert "GripperService" in gripper_pb2.DESCRIPTOR.services_by_name

    src = (Path(__file__).resolve().parents[1] / "proto" / "gripper.proto").read_text()
    for field in ("motor1", "motor2"):
        assert field in src, f"{field} missing from gripper.proto source"
