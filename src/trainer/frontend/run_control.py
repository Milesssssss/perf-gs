# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Cooperative run boundaries: work calls its checkpoint between cameras, frames and phases,
and Stop unwinds optional work without failing the already-trained model."""
from __future__ import annotations

from typing import Callable

Checkpoint = Callable[[], None]


class RunStopped(Exception):
    """The user stopped this run; retain its model and completed artifacts."""


def uninterrupted() -> None:
    """Default checkpoint for evaluation without interactive controls."""