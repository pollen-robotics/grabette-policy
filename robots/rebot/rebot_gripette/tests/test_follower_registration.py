"""The CLI wiring: --robot.type=seeed_b601_rs_follower_fixed must still resolve.

LeRobot does not look up a class by the registered name alone. draccus builds
the *config* from the name, then make_device_from_device_class strips the
trailing "Config" from the config class name and searches the config module's
PARENT package for the result. Both halves of that convention are load-bearing
and neither is exercised by any other test.
"""
import pytest

pytest.importorskip("lerobot_robot_seeed_b601")

import rebot_gripette
from lerobot.robots.robot import RobotConfig

from rebot_gripette import ReBotFollower, ReBotFollowerConfig

REGISTERED_TYPE = "seeed_b601_rs_follower_fixed"


def test_config_type_is_registered():
    assert RobotConfig.get_choice_class(REGISTERED_TYPE) is ReBotFollowerConfig
    assert ReBotFollowerConfig(id="x", port="can0").type == REGISTERED_TYPE


def test_device_class_name_follows_the_config_name():
    assert ReBotFollowerConfig.__name__.endswith("Config")
    assert ReBotFollowerConfig.__name__[: -len("Config")] == ReBotFollower.__name__


def test_device_class_is_reachable_from_the_parent_package():
    """make_device_from_device_class imports `rebot_gripette` (the parent of
    `rebot_gripette.follower`) and getattr()s the class off it."""
    parent = ReBotFollowerConfig.__module__.rsplit(".", 1)[0]
    assert parent == rebot_gripette.__name__
    assert getattr(rebot_gripette, ReBotFollower.__name__) is ReBotFollower


def test_config_overrides_exactly_one_default():
    """joint_directions is pinned to the identity (see follower.py); every
    other numeric default is inherited from upstream unchanged.

    Pinned here as an exact set: a second silent override would be a second
    place the arm's behaviour diverges from the plugin's documentation.
    """
    from lerobot_robot_seeed_b601 import SeeedB601RSFollowerConfig

    assert issubclass(ReBotFollowerConfig, SeeedB601RSFollowerConfig)
    assert set(ReBotFollowerConfig.__dict__.get("__annotations__", {})) == {
        "joint_directions"}
    assert set(ReBotFollowerConfig(id="_", port="can0").joint_directions.values()) == {1.0}
