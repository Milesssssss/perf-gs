# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Session provenance recorded once per suite run, plus the device-wide GPU memory poller.

The dirty flag is first-class: a dirty tree means the commit does not identify the code behind the numbers.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _run(cmd: list[str], timeout: float = 30.0) -> str | None:
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                              text=True, timeout=timeout)
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_state() -> dict:
    """Commit, branch, dirtiness, and which files are dirty (a dirty README is not a dirty rasterizer)."""
    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    dirty_files: list[str] = []
    if status:
        dirty_files = [line[3:].strip() for line in status.splitlines()
                       if line.strip()]
    # Changes to source/shaders are the ones that can move a result.
    relevant = [f for f in dirty_files
                if f.startswith(("src/", "shaders/", "scripts/"))
                and not f.endswith((".md", ".txt"))]
    return {
        "commit": commit,
        "commit_short": (commit[:12] if commit else None),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "describe": _run(["git", "describe", "--always", "--dirty"]),
        "dirty": bool(dirty_files),
        "dirty_file_count": len(dirty_files),
        "dirty_files": sorted(dirty_files)[:100],
        "dirty_source_files": sorted(relevant)[:100],
        "commit_identifies_code": not bool(relevant),
    }


def gpu_state() -> dict:
    """GPU name, memory and driver as the driver reports it; the run report separately records the bound adapter."""
    query = _run(["nvidia-smi",
                  "--query-gpu=name,memory.total,driver_version,pstate,"
                  "clocks.max.sm,clocks.max.memory,power.limit",
                  "--format=csv,noheader"])
    record: dict = {"nvidia_smi_available": query is not None}
    if query:
        rows = [r.strip() for r in query.splitlines() if r.strip()]
        record["device_count"] = len(rows)
        fields = [f.strip() for f in rows[0].split(",")]
        keys = ("name", "memory_total", "driver_version", "pstate",
                "clocks_max_sm", "clocks_max_memory", "power_limit")
        record.update(dict(zip(keys, fields)))
        if len(rows) > 1:
            # Multi-GPU: the run report's adapter name disambiguates which one the trainer picked.
            record["all_devices"] = rows
    record["cuda_version_smi"] = _run(
        ["nvidia-smi", "--query", "--display=COMPUTE"]) is not None
    return record


def stbn_asset_state() -> dict:
    """Content hash of the NVIDIA STBN volume the runs will read."""
    from trainer.backend.render import stbn_table

    path = stbn_table.nvidia_volume_path()
    record: dict = {"path": str(path), "name": path.name,
                    "license": "NVIDIA STBN, NON-COMMERCIAL (see .stbn/assets)"}
    try:
        data = path.read_bytes()
        record["sha256"] = hashlib.sha256(data).hexdigest()
        record["bytes"] = len(data)
        record["present"] = True
        record["shape"] = (f"{stbn_table.NVIDIA_SLICES}x"
                          f"{stbn_table.NVIDIA_HEIGHT}x"
                          f"{stbn_table.NVIDIA_WIDTH}")
    except OSError as error:
        record["present"] = False
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def package_state() -> dict:
    """Installed versions of the numeric/graphics stack, falling back to distribution metadata (e.g. ``lpips``)."""
    from importlib.metadata import PackageNotFoundError, version as dist_version

    # module name -> distribution name (they differ for PIL).
    modules = {"numpy": "numpy", "scipy": "scipy", "slangpy": "slangpy",
               "matplotlib": "matplotlib", "PIL": "pillow", "torch": "torch",
               "torchvision": "torchvision", "lpips": "lpips"}
    versions: dict[str, str] = {}
    for module, distribution in modules.items():
        try:
            imported = __import__(module)
        except Exception:                                  # noqa: BLE001
            versions[module] = "not installed"
            continue
        found = getattr(imported, "__version__", None)
        if found:
            versions[module] = str(found)
            continue
        try:
            versions[module] = str(dist_version(distribution))
        except PackageNotFoundError:
            versions[module] = "installed (version unknown)"
    return versions


def collect_environment() -> dict:
    """The full session provenance block, stamped onto every cell record."""
    return {
        "collected_at": utc_now(),
        "git": git_state(),
        "gpu": gpu_state(),
        "stbn_asset": stbn_asset_state(),
        "packages": package_state(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "cpu_count": __import__("os").cpu_count(),
    }


class GpuMemoryPoller:
    """Samples device-wide GPU memory (WDDM has no per-process attribution) in a background thread.

    The pre-launch baseline is reported so a non-idle GPU is visible; a missing nvidia-smi never fails a cell.
    """

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self._samples: list[float] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.baseline_mb: float | None = None
        self.available = False

    @staticmethod
    def _sample_mb() -> float | None:
        out = _run(["nvidia-smi", "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"], timeout=10.0)
        if not out:
            return None
        try:
            return float(out.splitlines()[0].strip())
        except (ValueError, IndexError):
            return None

    def start(self) -> None:
        self.baseline_mb = self._sample_mb()
        self.available = self.baseline_mb is not None
        if not self.available:
            return
        self._samples = []
        self._stop.clear()

        def loop() -> None:
            while not self._stop.wait(self.interval):
                value = self._sample_mb()
                if value is not None:
                    self._samples.append(value)

        self._thread = threading.Thread(target=loop, daemon=True,
                                        name="gpu-mem-poll")
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if not self.available:
            return {"available": False,
                    "reason": "nvidia-smi memory.used unavailable"}
        peak = max(self._samples) if self._samples else None
        return {
            "available": True,
            "scope": "device-wide (per-process attribution unavailable on WDDM)",
            "baseline_mb": self.baseline_mb,
            "peak_mb": peak,
            "peak_over_baseline_mb": (
                (peak - self.baseline_mb)
                if (peak is not None and self.baseline_mb is not None) else None),
            "samples": len(self._samples),
            "interval_s": self.interval,
            "note": ("interpretable only on an otherwise-idle GPU; a large "
                     "baseline_mb means something else held memory and the "
                     "delta understates this run's true peak"),
        }


def write_environment(path: Path, environment: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(environment, indent=2))
