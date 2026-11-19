# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Fetch NVIDIA's scalar spatiotemporal blue noise into the gitignored ``.stbn/`` cache.

The upstream license is non-commercial research only, so the assets are never committed.
    python scripts/data/download_stbn.py
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from trainer.frontend.stbn_download import download


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download NVIDIA's scalar spatiotemporal blue noise "
                    "textures into the local (gitignored) STBN cache.")
    parser.add_argument("--dest", type=Path, default=None,
                        help="Cache root (default: the .stbn/ cache the trainer reads).")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the volume already exists.")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep the 30 MB STBN.zip after extraction.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        download(args.dest, force=args.force, keep_zip=args.keep_zip)
    except (RuntimeError, zipfile.BadZipFile, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
