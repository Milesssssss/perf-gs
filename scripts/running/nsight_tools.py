# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Nsight Graphics install lookup and capture opening, shared by the capture launchers."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_NVIDIA_ROOT = Path(r"C:\Program Files\NVIDIA Corporation")


def find_nsight_tool(exe_name: str, env_override: str) -> Path | None:
    """``exe_name`` from the newest Nsight Graphics install; the ``env_override`` variable wins."""
    override = os.environ.get(env_override)
    if override:
        path = Path(override)
        return path if path.is_file() else None
    if not _NVIDIA_ROOT.is_dir():
        return None

    def version(install: Path) -> list[int]:
        return [int(x) for x in re.findall(r"\d+", install.name)] or [-1]

    for install in sorted(_NVIDIA_ROOT.glob("Nsight Graphics *"), key=version, reverse=True):
        for exe in install.glob(f"host/*/{exe_name}"):
            if exe.is_file():
                return exe
    return None


def open_in_nsight(capture: Path, tool: Path) -> None:
    """Open a saved capture in the UI of the Nsight install that wrote it, outliving this script."""
    ui = tool.with_name("ngfx-ui.exe")
    if not ui.is_file():
        os.startfile(capture)
        return
    subprocess.Popen([str(ui), str(capture)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
