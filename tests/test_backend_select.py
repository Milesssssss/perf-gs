# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Backend selection: Metal on Apple Silicon, Vulkan otherwise, explicit flags win."""

from __future__ import annotations

import platform
import sys

import pytest

from trainer.backend.compute.device import apple_silicon, default_backend
from trainer.frontend.cli import build_parser


def _host(monkeypatch: pytest.MonkeyPatch, system: str, machine: str) -> None:
    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(platform, "machine", lambda: machine)


def test_apple_silicon_defaults_to_metal(monkeypatch: pytest.MonkeyPatch) -> None:
    _host(monkeypatch, "darwin", "arm64")
    assert apple_silicon()
    assert default_backend() == "metal"
    args = build_parser().parse_args(["--preset", "quality+", "--iters", "0", "--no-viewer", "--save-ply"])
    assert args.backend == "metal"
    assert args.preset == "quality+"
    assert args.iters == 0
    assert args.no_viewer is True
    assert args.save_ply is True


def test_explicit_vulkan_wins_on_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    _host(monkeypatch, "darwin", "arm64")
    assert build_parser().parse_args(["--backend", "vulkan"]).backend == "vulkan"


def test_intel_mac_and_linux_stay_on_vulkan(monkeypatch: pytest.MonkeyPatch) -> None:
    _host(monkeypatch, "darwin", "x86_64")
    assert not apple_silicon()
    assert build_parser().parse_args([]).backend == "vulkan"
    _host(monkeypatch, "linux", "x86_64")
    assert build_parser().parse_args(["--preset", "balanced"]).backend == "vulkan"
    assert build_parser().parse_args(["--preset", "ultrafast"]).preset == "ultrafast"
