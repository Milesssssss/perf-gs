# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Cooperative run boundaries shared by training and evaluation.

Work checks its caller's checkpoint between cameras, frames, and phases. Stop
unwinds optional work without treating the already-trained model as a failure.
"""
from __future__ import annotations

from typing import Callable

Checkpoint = Callable[[], None]


class RunStopped(Exception):
    """The user stopped this run; retain its model and completed artifacts."""


def uninterrupted() -> None:
    """Default checkpoint for evaluation without interactive controls."""