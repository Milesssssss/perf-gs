# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Get the two binaries reconstruction needs: ``colmap`` and (optionally) ``glomap``, downloaded
as prebuilt Windows zips; a CUDA build that cannot start here falls back to the no-CUDA zip."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

ProgressCallable = Callable[[str], None]

# .../src/trainer/frontend/reconstruction/tools.py -> repo root is parents[4]
_ROOT = Path(__file__).resolve().parents[4]
_REPO = {"colmap": "colmap/colmap", "glomap": "colmap/glomap"}
_WINDOWS = os.name == "nt"          # the only platform with prebuilt release zips
_MAX_DOWNLOAD = 768 * 1024 * 1024   # runaway guard; the colmap CUDA zip is ~360 MB
# STATUS_DLL_NOT_FOUND / STATUS_ENTRYPOINT_NOT_FOUND: a CUDA build missing its runtime DLLs.
_DLL_FAILURE = (-1073741515, -1073741511, 3221225781, 3221225785)


def _noop(_msg: str) -> None:
    pass


def _variant(cuda: bool) -> str:
    return "cuda" if cuda else "nocuda"


def _url(tool: str, cuda: bool) -> str:
    return (f"https://github.com/{_REPO[tool]}/releases/latest/download/{tool}-x64-windows-{_variant(cuda)}.zip")


def _find_exe(tool: str, root: Path) -> Path | None:
    """Find the exe anywhere under ``root``, leaving it beside its DLLs."""
    name = f"{tool}.exe" if _WINDOWS else tool
    return next((p for p in sorted(root.rglob(name)) if p.is_file()), None)


def _starts(exe: Path, progress: ProgressCallable) -> bool:
    """Whether the binary can actually run here (see ``_DLL_FAILURE``)."""
    try:
        rc = subprocess.run([str(exe), "help"], capture_output=True,
                            timeout=180, check=False).returncode
    except (OSError, subprocess.SubprocessError) as e:
        progress(f"[recon] {exe.name} did not launch: {e}")
        return False
    if rc in _DLL_FAILURE:
        progress(f"[recon] {exe.name} cannot load its DLLs (exit {rc}) — no CUDA runtime?")
        return False
    return True


def _download(url: str, dest: Path, progress: ProgressCallable) -> bool:
    if not url.startswith("https://github.com/"):   # HTTPS-only, one known host
        raise ValueError(f"refusing url: {url}")
    progress(f"[recon] downloading {url}")
    total = 0
    note = step = 32 * 1024 * 1024
    with urllib.request.urlopen(url, timeout=60) as r, dest.open("wb") as f:
        size = int(r.headers.get("Content-Length") or 0)
        while chunk := r.read(1 << 18):
            total += len(chunk)
            if total > _MAX_DOWNLOAD:
                progress("[recon] download exceeded the size cap; aborting")
                return False
            f.write(chunk)
            if total >= note:
                note += step
                progress(f"[recon] downloading [{total >> 20}/{size >> 20} MB]" if size
                         else f"[recon] downloaded {total >> 20} MB")
    return True


def _unzip(src: Path, dest: Path) -> None:
    with zipfile.ZipFile(src) as z:
        # zipfile already sanitizes member names; reject such an archive instead of rewriting.
        if any(Path(m).is_absolute() or ".." in Path(m).parts for m in z.namelist()):
            raise ValueError("zip member escapes its destination")
        z.extractall(dest)


class ReconTools:
    """Where colmap/glomap are, and how to go get them."""

    def __init__(self, colmap_bin: str | None = None, glomap_bin: str | None = None,
                 recon_cuda: bool = True, tools_dir: Path | str | None = None) -> None:
        self.dir = Path(tools_dir) if tools_dir else _ROOT / ".recon-tools"
        self.recon_cuda = bool(recon_cuda)
        self._override = {"colmap": colmap_bin, "glomap": glomap_bin}

    def resolve(self, tool: str) -> str | None:
        """An explicit path, then ``$COLMAP_PATH``/``$GLOMAP_PATH``, then what we
        installed (preferred variant first), then PATH."""
        for cand in (self._override[tool], os.environ.get(f"{tool.upper()}_PATH")):
            if cand and Path(cand).exists():
                return str(cand)
        for cuda in (self.recon_cuda, not self.recon_cuda):
            exe = _find_exe(tool, self.dir / f"{tool}-{_variant(cuda)}")
            if exe is not None:
                return str(exe)
        return shutil.which(tool)

    def is_ready(self) -> bool:
        """colmap is the whole front-end; glomap only replaces its mapper."""
        return self.resolve("colmap") is not None

    def setup(self, progress: ProgressCallable = _noop) -> bool:
        for tool in ("colmap", "glomap"):
            progress(f"[stage] {tool}")
            found = self.resolve(tool)
            if found:
                progress(f"[recon] {tool} already available: {found}")
            else:
                self._install(tool, self.recon_cuda, progress)
        progress("[stage] done")
        ready = self.is_ready()
        progress(f"[recon] tools READY — {self.summary()}" if ready else
                 "[recon] SETUP FAILED: no colmap. Install it and pass --colmap-bin.")
        return ready

    def summary(self) -> str:
        glomap = self.resolve("glomap")
        return (f"front-end: {self.resolve('colmap')}; "
                f"mapper: {glomap + ' (global SfM)' if glomap else 'colmap incremental'}")

    def _install(self, tool: str, cuda: bool, progress: ProgressCallable) -> str | None:
        if not _WINDOWS:
            progress(f"[recon] no prebuilt {tool} for this platform — install it (e.g. conda install -c conda-forge {tool}) or pass --{tool}-bin")
            return None
        dest = self.dir / f"{tool}-{_variant(cuda)}"
        dest.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                zpath = Path(tmp) / f"{tool}.zip"
                if not _download(_url(tool, cuda), zpath, progress):
                    return None
                _unzip(zpath, dest)
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            progress(f"[recon] {tool} install failed: {e}")
            shutil.rmtree(dest, ignore_errors=True)
            return None
        exe = _find_exe(tool, dest)
        if exe is not None and _starts(exe, progress):
            progress(f"[recon] {tool} READY ({_variant(cuda)}): {exe}")
            return str(exe)
        shutil.rmtree(dest, ignore_errors=True)
        if cuda:
            progress(f"[recon] retrying {tool} with the no-CUDA build")
            return self._install(tool, False, progress)
        return None


def gpu_flag(exe: str, verb: str) -> str | None:
    try:
        help_text = subprocess.run([exe, verb, "--help"], capture_output=True,
                                   text=True, timeout=180, errors="replace",
                                   check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"--(\w+)\.use_gpu", (help_text.stdout or "") + (help_text.stderr or ""))
    return match.group(0) if match else None
