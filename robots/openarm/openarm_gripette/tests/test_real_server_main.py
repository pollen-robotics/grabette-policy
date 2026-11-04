"""grpc_server_real.main() startup ordering (all hardware + gRPC mocked).

No CAN, no sockets: the robot, kinematics and grpc.server are MagicMocks.
"""

import sys
from unittest.mock import MagicMock

import pytest

from openarm_gripette import grpc_server_real as srv


@pytest.fixture
def mocked_main(monkeypatch):
    """Patch every hardware / network touchpoint of main(); return the mocks."""
    follower = MagicMock(name="follower")
    server = MagicMock(name="grpc_server")
    server.add_insecure_port.return_value = 1  # bind OK by default
    server.wait_for_termination.side_effect = KeyboardInterrupt
    monkeypatch.setattr(srv, "OpenArm7FollowerConfig", MagicMock())
    monkeypatch.setattr(srv, "OpenArm7Follower", MagicMock(return_value=follower))
    monkeypatch.setattr(srv, "Kinematics", MagicMock())
    monkeypatch.setattr(srv, "ArmServicer", MagicMock())
    monkeypatch.setattr(srv.arm_pb2_grpc, "add_ArmServiceServicer_to_server", MagicMock())
    monkeypatch.setattr(srv.grpc, "server", MagicMock(return_value=server))
    monkeypatch.setattr(sys, "argv", ["grpc_server_real"])
    return follower, server


# RA-5: connect() -> configure() toggles torque on all 7 motors. It used to
# run BEFORE the single-instance port guard, so a second server instance
# briefly dropped torque on an arm the first instance was driving.
def test_port_busy_does_not_touch_motors(mocked_main):
    follower, server = mocked_main
    server.add_insecure_port.return_value = 0  # port already bound
    with pytest.raises(SystemExit):
        srv.main()
    follower.connect.assert_not_called()


# RA-5: an exception after connect() (servicer init, server start) used to
# exit with the motors still enabled and the bus never disconnected.
def test_init_failure_after_connect_disables_and_disconnects(mocked_main, monkeypatch):
    follower, _ = mocked_main
    monkeypatch.setattr(srv, "ArmServicer", MagicMock(side_effect=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        srv.main()
    follower.connect.assert_called_once()
    follower.bus.disable_torque.assert_called()
    follower.disconnect.assert_called_once()


def test_normal_shutdown_disables_and_disconnects(mocked_main):
    follower, server = mocked_main
    srv.main()  # wait_for_termination raises KeyboardInterrupt -> clean shutdown
    server.start.assert_called_once()
    follower.bus.disable_torque.assert_called()
    follower.disconnect.assert_called_once()
