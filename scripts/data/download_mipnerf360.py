"""Download the Mip-NeRF 360 dataset into the repo's ``datasets/`` folder.

Source: https://jonbarron.info/mipnerf360/ (Barron et al., Mip-NeRF 360). The
scenes ship as two public Google Research archives:

  * ``360_v2.zip``           - the seven publicly redistributable scenes
                               (bicycle, bonsai, counter, garden, kitchen,
                               room, stump).
  * ``360_extra_scenes.zip`` - the two extra scenes (flowers, treehill), which
                               carry a redistribution caveat from the authors;
                               see the notice printed by ``--set extra``/``all``.

The archives are large (several GB). The download streams to disk with a
progress readout and uses only the Python standard library, so it adds no pip
dependencies.

Examples
--------
    # the seven public scenes -> datasets/360_v2/
    python scripts/data/download_mipnerf360.py

    # everything, including the extra scenes -> datasets/360_extra_scenes/
    python scripts/data/download_mipnerf360.py --set all

    # custom location, keep the downloaded .zip archives
    python scripts/data/download_mipnerf360.py --dest D:/data/mipnerf360 --keep-zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Public Google Research archives linked from https://jonbarron.info/mipnerf360/.
# Centralized so a URL change is a one-line fix.
DATASET_URLS: dict[str, str] = {
    "360_v2": "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
    "360_extra_scenes": "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip",
}

# Scene folders each archive is expected to yield, used for the idempotent
# "already downloaded" skip check.
SET_SCENES: dict[str, tuple[str, ...]] = {
    "360_v2": ("bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"),
    "360_extra_scenes": ("flowers", "treehill"),
}

# ``--set`` value -> the archive keys it pulls.
SET_ARCHIVES: dict[str, tuple[str, ...]] = {
    "v2": ("360_v2",),
    "extra": ("360_extra_scenes",),
    "all": ("360_v2", "360_extra_scenes"),
}

# Printed before fetching the extra scenes (mirrors datasets/360_v2/flowers.txt).
EXTRA_SCENES_NOTICE = (
    "NOTE: the extra scenes (flowers, treehill) carry a redistribution caveat "
    "from the Mip-NeRF 360 authors. Only download them if your use complies "
    "with the terms at https://jonbarron.info/mipnerf360/."
)

DOWNLOAD_CHUNK = 1 << 20  # 1 MiB


def default_datasets_dir() -> Path:
    """``<repo>/datasets`` resolved from this file's location."""
    return Path(__file__).resolve().parents[2] / "datasets"


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def _scenes_present(set_dir: Path, scenes: tuple[str, ...]) -> bool:
    """True when every expected scene folder exists and is non-empty."""
    return all((set_dir / s).is_dir() and any((set_dir / s).iterdir())
               for s in scenes)


def _print_progress(downloaded: int, total: int, *, final: bool = False) -> None:
    if total:
        pct = downloaded / total * 100.0
        msg = f"  {_human_bytes(downloaded)} / {_human_bytes(total)} ({pct:5.1f}%)"
    else:
        msg = f"  {_human_bytes(downloaded)}"
    print(f"\r{msg}", end="\n" if final else "", flush=True)


def _download(url: str, dest_zip: Path) -> None:
    """Stream ``url`` to ``dest_zip`` with a progress readout (stdlib only).

    Writes to a ``.part`` sibling first and renames on success so an
    interrupted download never masquerades as a complete archive.
    """
    part = dest_zip.with_name(dest_zip.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "slang-3dgs-trainer/dataset-fetch"})
    try:
        with urllib.request.urlopen(request) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} for {url}")
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
    part.replace(dest_zip)


def _archive_root(staging: Path) -> Path:
    """The dir inside ``staging`` that directly holds the scene folders.

    Handles both archive layouts: scene folders at the zip root, or nested
    under a single top-level folder (e.g. ``360_v2/``).
    """
    entries = [p for p in staging.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return staging


def _extract_and_place(zip_path: Path, set_name: str, dest: Path, *,
                       force: bool) -> None:
    """Extract ``zip_path`` and move its scene folders under ``dest/set_name``."""
    staging = dest / ".download_tmp" / f"{set_name}_extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print(f"  extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(staging)

    root = _archive_root(staging)
    target = dest / set_name
    target.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        placed = target / child.name
        if placed.exists():
            if not force:
                continue
            shutil.rmtree(placed) if placed.is_dir() else placed.unlink()
        shutil.move(str(child), str(placed))
    shutil.rmtree(staging)


def download_archive(archive_name: str, dest: Path, *, force: bool,
                     keep_zip: bool) -> None:
    """Fetch + extract one archive (skipping if its scenes already exist)."""
    scenes = SET_SCENES[archive_name]
    set_dir = dest / archive_name
    if not force and _scenes_present(set_dir, scenes):
        print(f"[{archive_name}] already present at {set_dir} - skipping "
              f"(use --force to re-download).")
        return

    url = DATASET_URLS[archive_name]
    zip_path = dest / ".download_tmp" / f"{archive_name}.zip"
    print(f"[{archive_name}] downloading {url}")
    _download(url, zip_path)
    _extract_and_place(zip_path, archive_name, dest, force=force)
    if not keep_zip:
        zip_path.unlink(missing_ok=True)
    print(f"[{archive_name}] done -> {set_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Mip-NeRF 360 dataset into datasets/.")
    parser.add_argument(
        "--set", choices=sorted(SET_ARCHIVES), default="v2",
        help="Scenes to fetch: v2 (7 public scenes, default), extra "
             "(flowers, treehill), or all.")
    parser.add_argument(
        "--dest", type=Path, default=default_datasets_dir(),
        help="Destination folder (default: the repo's datasets/).")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the scenes already exist.")
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="Keep the downloaded .zip archives after extraction.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dest: Path = args.dest
    archives = SET_ARCHIVES[args.set]

    if "360_extra_scenes" in archives:
        print(EXTRA_SCENES_NOTICE)

    dest.mkdir(parents=True, exist_ok=True)
    try:
        for archive_name in archives:
            download_archive(archive_name, dest, force=args.force,
                             keep_zip=args.keep_zip)
    except (RuntimeError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    tmp = dest / ".download_tmp"
    if tmp.is_dir() and not any(tmp.iterdir()):
        tmp.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
