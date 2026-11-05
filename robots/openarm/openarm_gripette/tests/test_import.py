"""Smoke test: the real-arm driver and gRPC server import without hardware.

Anchors this package's CI lane; the hardware-free regression tests for the
server (fake bus / fake arm) live alongside it.
"""


def test_real_server_imports():
    from openarm_gripette import grpc_server_real

    assert hasattr(grpc_server_real, "ArmServicer")
    assert hasattr(grpc_server_real, "ArmInterface")
