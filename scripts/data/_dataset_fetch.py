# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Dependency-free download/extract machinery shared by the dataset fetchers.

Uses certifi's CA bundle when installed; TLS verification is never disabled.
"""
from __future__ import annotations

import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path

DOWNLOAD_CHUNK = 1 << 20  # 1 MiB


def default_datasets_dir() -> Path:
    """``<repo>/datasets`` resolved from this file's location (scripts/data/)."""
    return Path(__file__).resolve().parents[2] / "datasets"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def dirs_present(parent: Path, names: Iterable[str]) -> bool:
    """True when every named child folder exists and is non-empty."""
    return all((parent / n).is_dir() and any((parent / n).iterdir())
               for n in names)


def _print_progress(downloaded: int, total: int, *, final: bool = False) -> None:
    if total:
        pct = downloaded / total * 100.0
        msg = f"  {human_bytes(downloaded)} / {human_bytes(total)} ({pct:5.1f}%)"
    else:
        msg = f"  {human_bytes(downloaded)}"
    print(f"\r{msg}", end="\n" if final else "", flush=True)


def _ssl_context() -> ssl.SSLContext | None:
    """Verifying SSL context from certifi's CA bundle, or None (urllib's verifying default) without certifi.

    certifi completes chains the default store can't (e.g. repo-sam.inria.fr).
    """
    try:
        import certifi
    except ModuleNotFoundError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def stream_to_file(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` via a ``.part`` rename so an interrupted download never looks complete."""
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "slang-3dgs-trainer/dataset-fetch"})
    try:
        with urllib.request.urlopen(request, context=_ssl_context()) as response:
            # file:// responses report status None; only HTTP promises 200.
            status = getattr(response, "status", None)
            if status not in (None, 200):
                raise RuntimeError(f"HTTP {status} for {url}")
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            next_report = 0
            with part.open("wb") as out:
                while chunk := response.read(DOWNLOAD_CHUNK):
                    out.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        _print_progress(downloaded, total)
                        next_report = downloaded + DOWNLOAD_CHUNK * 8
            _print_progress(downloaded, total, final=True)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to download {url}: {exc.reason}") from exc
    part.replace(dest)


def _archive_root(staging: Path) -> Path:
    """The dir in ``staging`` holding the extracted folders (zip root or its single top-level folder)."""
    entries = [p for p in staging.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return staging


def extract_and_place(zip_path: Path, staging: Path, target: Path, *,
                      force: bool, only: Iterable[str] | None = None) -> list[str]:
    """Extract ``zip_path`` and move its top-level folders (restricted to ``only``) into ``target``.

    Existing folders are skipped unless ``force``; returns the names actually placed.
    """
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    print(f"  extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(staging)

    keep = set(only) if only is not None else None
    target.mkdir(parents=True, exist_ok=True)
    placed: list[str] = []
    for child in sorted(_archive_root(staging).iterdir()):
        if child.name == "__MACOSX" or (keep is not None and child.name not in keep):
            continue
        if move_into(child, target / child.name, force=force):
            placed.append(child.name)
    shutil.rmtree(staging, ignore_errors=True)
    return placed


def move_into(src: Path, dst: Path, *, force: bool) -> bool:
    """Move ``src`` onto ``dst``; an existing ``dst`` is replaced only with ``force`` (returns whether moved)."""
    if dst.exists():
        if not force:
            return False
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return True
