# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""External Nsight capture command construction."""

from __future__ import annotations

import sys

import capture_gpu_trace


def test_dry_run_enables_real_time_shader_profiler(
    monkeypatch, capsys, tmp_path,
) -> None:
    ngfx = tmp_path / "ngfx.exe"
    ngfx.touch()
    project = tmp_path / "capture.ngfx-proj"
    project.write_text("{}", encoding="ascii")

    monkeypatch.setattr(capture_gpu_trace, "_find_ngfx_exe", lambda: ngfx)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture_gpu_trace.py",
            "--project", str(project),
            "--output-dir", str(tmp_path / "out"),
            "--dry-run",
        ],
    )

    assert capture_gpu_trace.main() == 0
    assert "--real-time-shader-profiler" in capsys.readouterr().out