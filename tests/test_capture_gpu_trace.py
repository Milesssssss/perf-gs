# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""External Nsight capture command construction."""

from __future__ import annotations

import sys

import capture_gpu_trace


def _dry_run(monkeypatch, capsys, tmp_path, *extra: str) -> str:
    ngfx = tmp_path / "ngfx.exe"
    ngfx.touch()
    project = tmp_path / "capture.ngfx-proj"
    project.write_text("{}", encoding="ascii")

    monkeypatch.setattr(capture_gpu_trace, "find_nsight_tool", lambda *_: ngfx)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture_gpu_trace.py",
            "--project", str(project),
            "--output-dir", str(tmp_path / "out"),
            "--dry-run",
            *extra,
        ],
    )

    assert capture_gpu_trace.main() == 0
    return capsys.readouterr().out


def test_dry_run_enables_real_time_shader_profiler(
    monkeypatch, capsys, tmp_path,
) -> None:
    assert "--real-time-shader-profiler" in _dry_run(monkeypatch, capsys, tmp_path)


def test_frozen_frames_do_not_recount_the_presents_before_the_capture(
    monkeypatch, capsys, tmp_path,
) -> None:
    """--start-after-frames counts from process start, so a late --capture-iter must not
    inflate the frozen loop (it once presented capture_iter + 1515 frames)."""
    out = _dry_run(monkeypatch, capsys, tmp_path, "--capture-iter", "1340")
    assert "--capture-frames 1508" in out
    assert "--start-after-frames 1355" in out