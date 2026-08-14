"""Fetch NVIDIA's pregenerated spatiotemporal blue noise into the local cache.

Source: https://github.com/NVIDIA-RTX/STBN (the SpatiotemporalBlueNoiseSDK that
accompanies Wolfe et al., "Scalar Spatiotemporal Blue Noise Masks",
arXiv:2112.09629 / EGSR 2022). ``Assets/STBN.zip`` in that repo holds the
textures their ScalarApp (void-and-cluster) produced; this script pulls the
scalar set — ``stbn_scalar_2Dx1Dx1D_128x128x64x1_<t>.png``, 64 slices of 128x128
8-bit grayscale — and stores it as one ``(64, 128, 128)`` uint8 ``.npy`` for
:mod:`trainer.backend.render.stbn` to consume.

    python scripts/data/download_stbn.py

LICENSING — READ THIS
---------------------
NVIDIA ships that repo under the "NVIDIA Source Code License for Spatiotemporal
Blue Noise Texture Generation", which limits use to NON-COMMERCIAL research and
evaluation, and requires that any redistribution carry a complete copy of the
license plus the original attribution notices.

Consequences this script is built around:

  * The assets land in a GITIGNORED cache (``.stbn/assets/``) and are never
    committed. Vendoring them into a shared repo would be a redistribution, and
    whether that is acceptable is a licensing call for a human to make, not a
    default for a download script to quietly take.
  * ``License.txt`` and a ``NOTICE.md`` recording the upstream URL/commit are
    written next to the data, so the provenance travels with it.
  * Running this script is OPT-IN. STBN pixel sampling is off by default.

If the intended use of this trainer is commercial, do not use this source. The
trainer's ``--stbn-source file`` accepts any scalar STBN volume instead — e.g.
one you generate by building that SDK's ScalarApp yourself under whatever terms
apply to you — and ``--stbn-source white`` needs no asset at all (it is the
control, not an equivalent).
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = "NVIDIA-RTX/STBN"
BRANCH = "main"
ASSET_ZIP_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/Assets/STBN.zip"
LICENSE_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/License.txt"
REPO_URL = f"https://github.com/{REPO}"

# The scalar (1-channel) STBN set inside STBN.zip: 2D space x 1D time x 1D
# value, 128x128x64. The vector sets (vec2/vec3/unitvec*) are for sampling
# directions and are not what a pixel mask wants.
SLICE_TEMPLATE = "STBN/stbn_scalar_2Dx1Dx1D_128x128x64x1_{index}.png"
SLICES = 64
SIZE = 128

VOLUME_NAME = "stbn_scalar_2Dx1Dx1D_128x128x64.npy"
LICENSE_NAME = "License.txt"
NOTICE_NAME = "NOTICE.md"

DOWNLOAD_CHUNK = 1 << 20  # 1 MiB

LICENSE_WARNING = (
    "NOTICE: NVIDIA's spatiotemporal blue noise SDK is licensed for "
    "NON-COMMERCIAL research/evaluation use only, and redistribution must "
    "carry the license + attribution. The assets are cached OUTSIDE version "
    f"control; see {REPO_URL}/blob/{BRANCH}/License.txt for the full terms."
)


def default_cache_dir() -> Path:
    """``<repo>/.stbn`` — gitignored (the repo ignores dot-directories)."""
    return Path(__file__).resolve().parents[2] / ".stbn"


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def _fetch(url: str, dest: Path, *, progress: bool = True) -> None:
    """Stream ``url`` to ``dest`` via a ``.part`` sibling (stdlib only), so an
    interrupted transfer can never be mistaken for a complete file."""
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "slang-3dgs-trainer/stbn-fetch"})
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
                    if progress and downloaded >= next_report:
                        print(f"\r  {_human_bytes(downloaded)} / "
                              f"{_human_bytes(total) if total else '?'}",
                              end="", flush=True)
                        next_report = downloaded + DOWNLOAD_CHUNK * 4
            if progress:
                print(f"\r  {_human_bytes(downloaded)} / "
                      f"{_human_bytes(total) if total else '?'}", flush=True)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to download {url}: {exc.reason}") from exc
    part.replace(dest)


def extract_scalar_volume(zip_path: Path) -> np.ndarray:
    """Assemble the 64 scalar slices into one ``(slices, height, width)`` uint8
    volume, in slice order.

    Each slice is an 8-bit grayscale PNG whose values ARE the mask's scalar
    field (0..255). We keep the raw 8-bit values rather than normalizing: the
    consumer ranks them per tile, which only needs the order.
    """
    with zipfile.ZipFile(zip_path) as archive:
        available = set(archive.namelist())
        missing = [SLICE_TEMPLATE.format(index=t) for t in range(SLICES)
                   if SLICE_TEMPLATE.format(index=t) not in available]
        if missing:
            raise RuntimeError(
                f"{zip_path.name} is missing {len(missing)} scalar slice(s), "
                f"e.g. {missing[0]}. Upstream may have renamed the asset; check "
                f"{REPO_URL}/tree/{BRANCH}/Assets.")
        slices = []
        for index in range(SLICES):
            with Image.open(io.BytesIO(
                    archive.read(SLICE_TEMPLATE.format(index=index)))) as image:
                array = np.array(image.convert("L"))
            if array.shape != (SIZE, SIZE):
                raise RuntimeError(
                    f"slice {index} is {array.shape}, expected ({SIZE}, {SIZE})")
            slices.append(array)
    return np.ascontiguousarray(np.stack(slices).astype(np.uint8))


def _write_notice(assets: Path, volume: np.ndarray) -> None:
    (assets / NOTICE_NAME).write_text(
        f"""# Spatiotemporal blue noise assets (third party)

Downloaded by `scripts/data/download_stbn.py` from <{REPO_URL}> (branch
`{BRANCH}`), file `Assets/STBN.zip`, scalar set
`stbn_scalar_2Dx1Dx1D_128x128x64x1_*.png`.

Generated by NVIDIA's ScalarApp (void-and-cluster) and published with:

  Wolfe, Morrical, Akenine-Moller, Ramamoorthi,
  "Scalar Spatiotemporal Blue Noise Masks", arXiv:2112.09629 (EGSR 2022).

`{VOLUME_NAME}` is those 64 PNG slices stacked into one
`{tuple(volume.shape)}` uint8 array — a repackaging, not a modification.

## License

NVIDIA Source Code License for Spatiotemporal Blue Noise Texture Generation
(full text in `{LICENSE_NAME}`). **Non-commercial use only** — research or
evaluation purposes. Redistribution requires shipping the license and retaining
attribution notices.

This directory is deliberately OUTSIDE version control. Do not commit it, and
do not redistribute these files without checking the license terms first.
""",
        encoding="utf-8")


def download(cache_dir: Path, *, force: bool, keep_zip: bool) -> Path:
    """Fetch + convert the scalar STBN volume; returns its ``.npy`` path."""
    assets = cache_dir / "assets"
    volume_path = assets / VOLUME_NAME
    if volume_path.is_file() and not force:
        print(f"[stbn] already present at {volume_path} — skipping "
              f"(use --force to re-download).")
        return volume_path

    zip_path = cache_dir / "download" / "STBN.zip"
    if not zip_path.is_file() or force:
        print(f"[stbn] downloading {ASSET_ZIP_URL}")
        _fetch(ASSET_ZIP_URL, zip_path)
    else:
        print(f"[stbn] reusing {zip_path}")

    print("[stbn] extracting the scalar 128x128x64 set ...")
    volume = extract_scalar_volume(zip_path)
    assets.mkdir(parents=True, exist_ok=True)
    np.save(volume_path, volume)

    print(f"[stbn] fetching {LICENSE_URL}")
    _fetch(LICENSE_URL, assets / LICENSE_NAME, progress=False)
    _write_notice(assets, volume)

    if not keep_zip:
        shutil.rmtree(zip_path.parent, ignore_errors=True)
    print(f"[stbn] done -> {volume_path}  {volume.shape} {volume.dtype}")
    return volume_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download NVIDIA's scalar spatiotemporal blue noise "
                    "textures into the local (gitignored) STBN cache.")
    parser.add_argument("--dest", type=Path, default=default_cache_dir(),
                        help="Cache root (default: the repo's .stbn/).")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the volume already exists.")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep the 30 MB STBN.zip after extraction.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(LICENSE_WARNING)
    try:
        download(args.dest, force=args.force, keep_zip=args.keep_zip)
    except (RuntimeError, zipfile.BadZipFile, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
