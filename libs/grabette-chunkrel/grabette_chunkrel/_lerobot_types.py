"""Transition types across lerobot 0.6.0 (`lerobot.types`) and 0.6.1+ (`lerobot.lerobot_types`)."""

from __future__ import annotations

try:
    from lerobot.types import EnvTransition, TransitionKey
except ModuleNotFoundError:  # pragma: no cover - exercised on lerobot>=0.6.1
    from lerobot.lerobot_types import EnvTransition, TransitionKey

__all__ = ["EnvTransition", "TransitionKey"]
