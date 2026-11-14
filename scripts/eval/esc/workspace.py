# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Ephemeral sandboxes: each run edits a fresh copy of the champion, never the real workspace.

Each copy gets a throwaway git repo so ``git diff`` yields an honest patch footprint.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from esc.config import EscConfig


@dataclass
class Sandbox:
    """A disposable working copy. Use as a context manager for auto-cleanup."""

    root: Path
    _tempdir: Path

    def cleanup(self) -> None:
        shutil.rmtree(self._tempdir, ignore_errors=True)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> None:
        self.cleanup()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True, check=False)


def _git_available() -> bool:
    return shutil.which("git") is not None


def git_baseline(root: Path) -> None:
    """Commit the current tree into a throwaway git repo so later diffs measure exactly what an agent changed."""
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "esc@harness.local")
    _git(root, "config", "user.name", "ESC Harness")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "esc baseline", "--no-verify")


def copy_codebase(source_root: Path, dest: Path, config: EscConfig,
                  init_git: bool = False) -> Path:
    """Copy ``source_root`` (minus excludes) to ``dest``; optionally git-baseline it."""
    shutil.copytree(
        source_root, dest,
        ignore=shutil.ignore_patterns(*config.scan_excludes),
        dirs_exist_ok=False,
    )
    if init_git and _git_available():
        git_baseline(dest)
    return dest


def create_sandbox(source_root: Path, config: EscConfig, init_git: bool = True) -> Sandbox:
    """Copy ``source_root`` (minus excludes) into a fresh temp dir."""
    tempdir = Path(tempfile.mkdtemp(prefix="esc_sandbox_"))
    dest = tempdir / "repo"
    copy_codebase(source_root, dest, config, init_git=init_git)
    return Sandbox(root=dest, _tempdir=tempdir)
