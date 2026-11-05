"""grpc_server_real.main() startup ordering (all hardware + gRPC mocked).

No CAN, no sockets, no signal handlers: the follower, arm interface,
kinematics, clearance gate and grpc.server are MagicMocks.
"""
import sys
from unittest.mock import MagicMock

import pytest

# The server module imports the follower, which needs the upstream plugin.
pytest.importorskip("lerobot_robot_seeed_b601")

from rebot_gripette import grpc_server_real as srv  # noqa: E402


@pytest.fixture
def mocked_main(monkeypatch):
    """Patch every hardware / network touchpoint of main(); return the mocks."""
    follower = MagicMock(name="follower")
    arm_iface = MagicMock(name="arm_iface")
    server = MagicMock(name="grpc_server")
    server.add_insecure_port.return_value = 1  # bind OK by default
    monkeypatch.setattr(srv, "ReBotFollowerConfig", MagicMock())
    monkeypatch.setattr(srv, "ReBotFollower", MagicMock(return_value=follower))
    monkeypatch.setattr(srv, "ReBotArmInterface", MagicMock(return_value=arm_iface))
    monkeypatch.setattr(srv, "Kinematics", MagicMock())
    monkeypatch.setattr(srv, "_build_clearance_gate", MagicMock(return_value=None))
    monkeypatch.setattr(srv, "check_home_pose", lambda q: q)
    monkeypatch.setattr(srv, "ArmServicer", MagicMock())
    monkeypatch.setattr(srv.arm_pb2_grpc, "add_ArmServiceServicer_to_server", MagicMock())
    monkeypatch.setattr(srv.grpc, "server", MagicMock(return_value=server))
    # main() installs SIGINT/SIGTERM handlers and forces the root logger
    # config; neither may leak into the pytest process.
    monkeypatch.setattr(srv.signal, "signal", MagicMock())
    monkeypatch.setattr(srv.logging, "basicConfig", MagicMock())
    monkeypatch.setattr(sys, "argv", ["grpc_server_real", "--home_joints",
                                      "0", "60", "60", "0", "0", "0"])
    return follower, arm_iface, server


# RA-5: connect() used to run BEFORE the single-instance port guard, on a bus
# a first server may be actively driving with torque enabled.
def test_port_busy_does_not_touch_the_bus(mocked_main):
    follower, _, server = mocked_main
    server.add_insecure_port.return_value = 0  # port already bound
    with pytest.raises(SystemExit):
        srv.main()
    follower.connect.assert_not_called()


# RA-5: an exception after connect() (servicer init, server start) used to
# exit with the bus still connected and no explicit torque-off.
@pytest.mark.parametrize("where", ["servicer", "start"])
def test_init_failure_after_connect_disables_and_disconnects(mocked_main, monkeypatch, where):
    follower, arm_iface, server = mocked_main
    if where == "servicer":
        monkeypatch.setattr(srv, "ArmServicer", MagicMock(side_effect=RuntimeError("boom")))
    else:
        server.start.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        srv.main()
    follower.connect.assert_called_once()
    arm_iface.set_torque.assert_called_with(False)
    follower.disconnect.assert_called_once()
    # A failing init must never START a motion on the way out.
    follower.controlled_descent.assert_not_called()


def test_normal_startup_connects_after_bind_and_serves(mocked_main):
    follower, _, server = mocked_main
    srv.main()  # wait_for_termination is a mock: returns at once
    server.add_insecure_port.assert_called_once()
    follower.connect.assert_called_once()
    server.start.assert_called_once()
    follower.disconnect.assert_not_called()  # shutdown is the signal handler's job
