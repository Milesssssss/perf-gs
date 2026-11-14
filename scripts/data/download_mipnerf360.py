# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Download the Mip-NeRF 360 dataset into the repo's ``datasets/`` folder (stdlib only).

    python scripts/data/download_mipnerf360.py --set all
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from _dataset_fetch import (
    default_datasets_dir,
    dirs_present,
    extract_and_place,
    stream_to_file,
)

# Public Google Research archives linked from https://jonbarron.info/mipnerf360/.
DATASET_URLS: dict[str, str] = {
    "360_v2": "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
    "360_extra_scenes": "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip",
}

# Scene folders each archive yields, used for the "already downloaded" skip check.
SET_SCENES: dict[str, tuple[str, ...]] = {
    "360_v2": ("bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"),
    "360_extra_scenes": ("flowers", "treehill"),
}

SET_ARCHIVES: dict[str, tuple[str, ...]] = {
    "v2": ("360_v2",),
    "extra": ("360_extra_scenes",),
    "all": ("360_v2", "360_extra_scenes"),
}

EXTRA_SCENES_NOTICE = (
    "NOTE: the extra scenes (flowers, treehill) carry a redistribution caveat "
    "from the Mip-NeRF 360 authors. Only download them if your use complies "
    "with the terms at https://jonbarron.info/mipnerf360/."
)

def download_archive(archive_name: str, dest: Path, *, force: bool,
                     keep_zip: bool) -> None:
    """Fetch + extract one archive (skipping if its scenes already exist)."""
    scenes = SET_SCENES[archive_name]
    set_dir = dest / archive_name
    if not force and dirs_present(set_dir, scenes):
        print(f"[{archive_name}] already present at {set_dir} - skipping "
              f"(use --force to re-download).")
        return

    url = DATASET_URLS[archive_name]
    tmp = dest / ".download_tmp"
    zip_path = tmp / f"{archive_name}.zip"
    print(f"[{archive_name}] downloading {url}")
    stream_to_file(url, zip_path)
    extract_and_place(zip_path, tmp / f"{archive_name}_extract", set_dir,
                      force=force)
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
