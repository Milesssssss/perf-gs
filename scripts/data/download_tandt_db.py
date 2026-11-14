# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Download the Tanks & Temples and Deep Blending COLMAP datasets into datasets/.

Research use only: both carry their originators' terms (the 3DGS authors redistribute them).
    python scripts/data/download_tandt_db.py --set tandt
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

# The combined COLMAP archive from the INRIA 3DGS project (holds tandt/ + db/).
TANDT_DB_URL = (
    "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/"
    "tandt_db.zip")

# Dataset folder -> its scene subfolders, used for the idempotent skip check.
DATASET_SCENES: dict[str, tuple[str, ...]] = {
    "tandt": ("train", "truck"),
    "db": ("drjohnson", "playroom"),
}

SET_DATASETS: dict[str, tuple[str, ...]] = {
    "tandt": ("tandt",),
    "db": ("db",),
    "all": ("tandt", "db"),
}

TERMS_NOTICE = (
    "NOTE: Tanks & Temples and Deep Blending are redistributed for RESEARCH by "
    "the 3D Gaussian Splatting authors and carry their originators' terms. Only "
    "download if your use complies — see https://www.tanksandtemples.org/ and "
    "http://visual.cs.ucl.ac.uk/pubs/deepblending/ ."
)


def download(datasets: tuple[str, ...], dest: Path, *, force: bool,
             keep_zip: bool) -> None:
    """Fetch ``tandt_db.zip`` once and keep the selected ``datasets`` (both live in one archive)."""
    if not force and all(dirs_present(dest / d, DATASET_SCENES[d])
                         for d in datasets):
        print(f"[tandt_db] {', '.join(datasets)} already present under {dest} - "
              f"skipping (use --force to re-download).")
        return

    tmp = dest / ".download_tmp"
    zip_path = tmp / "tandt_db.zip"
    print(f"[tandt_db] downloading {TANDT_DB_URL}")
    stream_to_file(TANDT_DB_URL, zip_path)
    extract_and_place(zip_path, tmp / "tandt_db_extract", dest,
                      force=force, only=datasets)
    if not keep_zip:
        zip_path.unlink(missing_ok=True)
    for name in datasets:
        print(f"[tandt_db] {name} -> {dest / name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Tanks & Temples + Deep Blending into datasets/.")
    parser.add_argument(
        "--set", choices=sorted(SET_DATASETS), default="all",
        help="Which datasets to keep: tandt (Tanks & Temples), db (Deep "
             "Blending), or all (default). Both come from the one archive.")
    parser.add_argument(
        "--dest", type=Path, default=default_datasets_dir(),
        help="Destination folder (default: the repo's datasets/).")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the datasets already exist.")
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="Keep the downloaded .zip archive after extraction.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dest: Path = args.dest
    datasets = SET_DATASETS[args.set]

    print(TERMS_NOTICE)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        download(datasets, dest, force=args.force, keep_zip=args.keep_zip)
    except (RuntimeError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    tmp = dest / ".download_tmp"
    if tmp.is_dir() and not any(tmp.iterdir()):
        tmp.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
