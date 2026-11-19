# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Release archives must contain the current trainer, shaders and license notices."""
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def _build(source, hook, destination):
    command = "import sys; from setuptools import build_meta; print(getattr(build_meta, sys.argv[1])(sys.argv[2]))"
    result = subprocess.run([sys.executable, "-c", command, hook, str(destination)],
                            cwd=source, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return destination / result.stdout.splitlines()[-1]


def test_wheel_and_sdist_include_shader_resources(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "setup.py", "MANIFEST.in", "LICENSE", "NOTICE.md", "README.md"):
        shutil.copy2(root / name, source / name)
    for name in ("src/trainer", "shaders"):
        shutil.copytree(root / name, source / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (source / "tests").mkdir()
    shutil.copy2(__file__, source / "tests/test_packaging.py")

    clean = _build(source, "build_wheel", tmp_path / "clean")
    for name in ("trainer/__init__.py", "trainer/shaders/common/camera.slang",
                 "trainer/retired.py", "trainer/shaders/obsolete.slang", "scene.ply"):
        cached = source / "build/lib" / name
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"stale build output")
        newer = cached.stat().st_mtime + 3600
        os.utime(cached, (newer, newer))
    cached_wheel = _build(source, "build_wheel", tmp_path / "cached")
    sdist = _build(source, "build_sdist", tmp_path / "sdist")

    modules = {p.relative_to(root / "src").as_posix(): p for p in (root / "src/trainer").rglob("*.py")}
    shaders = {p.relative_to(root / "shaders").as_posix(): p for p in (root / "shaders").rglob("*.slang")}
    notices = {name: (root / name).read_bytes() for name in ("LICENSE", "NOTICE.md")}
    with tarfile.open(sdist) as archive:
        members = {member.name.split("/", 1)[1]: member
                   for member in archive.getmembers() if member.isfile()}
        assert {p for p in members if p.startswith(("src/trainer/", "shaders/"))} == (
            {"src/" + name for name in modules} | {"shaders/" + name for name in shaders})
        assert not any(name.startswith("tests/") for name in members)
        for name, expected_bytes in notices.items():
            stream = archive.extractfile(members[name])
            assert stream is not None
            with stream:
                assert stream.read() == expected_bytes
        archive.extractall(tmp_path / "unpacked")

    unpacked = next((tmp_path / "unpacked").iterdir())
    from_sdist = _build(unpacked, "build_wheel", tmp_path / "from_sdist")
    payload = {**modules, **{"trainer/shaders/" + name: p for name, p in shaders.items()}}
    for artifact in (clean, cached_wheel, from_sdist):
        with zipfile.ZipFile(artifact) as wheel:
            metadata = [p for p in wheel.namelist() if ".dist-info/" in p]
            assert set(wheel.namelist()) - set(metadata) == set(payload)
            for name, path in payload.items():
                assert wheel.read(name) == path.read_bytes(), name
            license_paths = [p for p in metadata if p.rsplit("/", 1)[1] not in
                             ("METADATA", "WHEEL", "RECORD", "top_level.txt")]
            assert len(license_paths) == len(notices)
            assert {p.rsplit("/", 1)[1]: wheel.read(p) for p in license_paths} == notices

    site = tmp_path / "installed"
    with zipfile.ZipFile(from_sdist) as wheel:
        wheel.extractall(site)
    check = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import trainer
from trainer.backend.compute.device import SHADERS_DIR
from trainer.backend.render.stbn_table import default_cache_dir
assert Path(trainer.__file__).is_relative_to(sys.argv[1])
assert SHADERS_DIR == Path(sys.argv[1]) / 'trainer' / 'shaders'
assert (SHADERS_DIR / 'rasterize/tile_binned/forward.slang').is_file()
assert default_cache_dir() == Path.cwd() / '.stbn'
"""
    result = subprocess.run([sys.executable, "-I", "-c", check, str(site)],
                            cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
