"""Turn a folder of images into a COLMAP dataset ``load_colmap`` can open.

Four colmap commands, in the trainer's own process — the same sequence every 3DGS
preprocessing script runs:

  1. ``feature_extractor``   (GPU on a CUDA build, else the CPU)
  2. ``<matcher>_matcher``   (exhaustive | sequential | vocab_tree)
  3. mapping: ``glomap mapper`` (global SfM) if glomap resolved, else ``colmap mapper``
  4. ``image_undistorter``   (load_colmap only supports SIMPLE_PINHOLE / PINHOLE)

then the result is arranged as ``<out>/sparse/0`` + ``<out>/images``. Source images
are never modified; everything else lands under ``<out>``.

Every step prints a ``[stage] <name>`` line that the viewer turns into a progress
fraction (see :mod:`.progress`), and streams the tool's stdout to ``progress``.
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
from collections.abc import Callable
from pathlib import Path

from trainer.data.colmap import _find_sparse_dir
from trainer.frontend.reconstruction.tools import ReconTools, gpu_flag

ProgressCallable = Callable[[str], None]

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_MATCH_VERB = {"exhaustive": "exhaustive_matcher",
               "sequential": "sequential_matcher",
               "vocab_tree": "vocab_tree_matcher"}
# GLOMAP links its own older COLMAP, which refuses to open a database carrying
# these two tables as COLMAP 4.x writes them.
_GLOMAP_DROP = ("descriptors", "pose_priors")


def _has_images(d: Path) -> bool:
    return d.is_dir() and any(f.suffix.lower() in _IMAGE_EXTS for f in d.iterdir())


def classify_dataset(path: str | Path) -> str:
    """``"colmap"`` (a sparse model is there), ``"images"``, or ``"unknown"``.

    COLMAP detection reuses the loader's own probe, so ``"colmap"`` guarantees
    ``load_colmap`` can read it.
    """
    p = Path(path)
    if not p.is_dir():
        return "unknown"
    if _find_sparse_dir(p)[0] is not None:
        return "colmap"
    return "images" if _has_images(p) or _has_images(p / "images") else "unknown"


def _run(cmd: list[str], progress: ProgressCallable) -> int:
    """Run a tool, relaying its output line by line. Arg list, never a shell string."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        progress(line.rstrip())
    return proc.wait()


def _run_on_gpu_else_cpu(colmap: str, verb: str, args: list[str], cuda: bool,
                         progress: ProgressCallable) -> bool:
    """Feature extraction / matching are the only GPU-capable steps. Ask for the
    GPU, then retry on the CPU — a no-CUDA build cannot do SIFT on the GPU at all,
    and a CUDA one still can't if the driver is unhappy."""
    flag = gpu_flag(colmap, verb)
    for want in (["1", "0"] if cuda else ["0"]) if flag else [None]:
        extra = [flag, want] if want is not None else []
        progress(f"[recon] colmap {verb} (use_gpu={want or 'default'})")
        rc = _run([colmap, verb, *args, *extra], progress)
        if rc == 0:
            return True
        progress(f"[recon] colmap {verb} exited {rc}")
    return False


def _glomap_database(database: Path, workspace: Path, progress: ProgressCallable) -> Path:
    """A copy of ``database`` without ``_GLOMAP_DROP``, so glomap can open it.

    Replays the original's own CREATE statements against a fresh file (with the
    source ATTACHed), so column order, types and constraints stay byte-identical
    to what colmap wrote. Falls back to the original if that fails.
    """
    dst = workspace / "glomap.db"
    try:
        dst.unlink(missing_ok=True)
        con = sqlite3.connect(dst)
        try:
            con.execute("ATTACH DATABASE ? AS src", (str(database),))
            for kind, name, table, sql in con.execute(
                    "SELECT type, name, tbl_name, sql FROM src.sqlite_master "
                    "WHERE sql IS NOT NULL").fetchall():
                if table in _GLOMAP_DROP or name.startswith("sqlite_"):
                    continue
                con.execute(sql)
                if kind == "table":
                    con.execute(f'INSERT INTO main."{name}" SELECT * FROM src."{name}"')
            version = con.execute("PRAGMA src.user_version").fetchone()[0]
            con.execute(f"PRAGMA main.user_version = {int(version)}")
            con.commit()
        finally:
            con.close()
    except (sqlite3.Error, OSError) as e:
        progress(f"[recon] no glomap-compatible database copy ({e}); using the original")
        return database
    return dst


def _fresh(d: Path) -> Path:
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _model_dir(sparse: Path) -> Path:
    """Mappers write the model to ``<sparse>/0``; some write straight into it."""
    zero = sparse / "0"
    return zero if any(zero.glob("cameras.*")) else sparse


def reconstruct(tools: ReconTools, dataset_dir: str | Path, out_dir: str | Path,
                matcher: str = "exhaustive",
                progress: ProgressCallable = lambda _m: None) -> Path | None:
    """images -> ``<out_dir>/sparse/0`` + ``<out_dir>/images``. None on failure.

    Idempotent: an ``out_dir`` that already holds a sparse model is returned as is.
    """
    out, dataset = Path(out_dir), Path(dataset_dir)
    if classify_dataset(out) == "colmap":
        progress(f"[recon] {out} already reconstructed; skipping")
        return out
    colmap = tools.resolve("colmap")
    if colmap is None:
        progress("[recon] no colmap — press 'Set up recon tools' first")
        return None

    images = dataset / "images" if (dataset / "images").is_dir() else dataset
    work = _fresh(out / "_recon_work")
    database, sparse = work / "database.db", work / "sparse"
    sparse.mkdir()
    progress("[stage] start")
    progress(f"[recon] images={images} out={out} matcher={matcher}; {tools.summary()}")

    progress("[stage] features")
    if not _run_on_gpu_else_cpu(colmap, "feature_extractor",
                                ["--database_path", str(database),
                                 "--image_path", str(images)],
                                tools.recon_cuda, progress):
        progress("[recon] feature extraction FAILED")
        return None

    progress("[stage] matching")
    verb = _MATCH_VERB.get(matcher, _MATCH_VERB["exhaustive"])
    if not _run_on_gpu_else_cpu(colmap, verb, ["--database_path", str(database)],
                                tools.recon_cuda, progress):
        progress(f"[recon] {verb} FAILED")
        return None

    progress("[stage] mapping")
    glomap = tools.resolve("glomap")
    rc = 1
    if glomap is not None:
        rc = _run([glomap, "mapper",
                   "--database_path", str(_glomap_database(database, work, progress)),
                   "--image_path", str(images), "--output_path", str(sparse)], progress)
        if rc != 0:
            progress(f"[recon] glomap mapper failed (rc={rc}); using colmap's mapper")
    if rc != 0:
        rc = _run([colmap, "mapper", "--database_path", str(database),
                   "--image_path", str(images), "--output_path", str(sparse)], progress)
    if rc != 0 or not any(sparse.rglob("cameras.*")):
        progress("[recon] mapping produced no reconstruction")
        return None

    progress("[stage] undistort")
    undist = _fresh(work / "undistort")
    if _run([colmap, "image_undistorter", "--image_path", str(images),
             "--input_path", str(_model_dir(sparse)), "--output_path", str(undist),
             "--output_type", "COLMAP"], progress) != 0:
        progress("[recon] undistortion FAILED")
        return None

    progress("[stage] arrange")
    model = _model_dir(undist / "sparse")
    final_sparse, final_images = _fresh(out / "sparse" / "0"), _fresh(out / "images")
    for f in model.iterdir():
        shutil.copy2(f, final_sparse / f.name)
    n = 0
    for f in (undist / "images").iterdir():
        if f.is_file():
            shutil.copy2(f, final_images / f.name)
            n += 1
    shutil.rmtree(work, ignore_errors=True)
    progress("[stage] done")
    progress(f"[recon] DONE -> {final_sparse} ({n} images)")
    return out
