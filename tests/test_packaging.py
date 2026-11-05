# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Installed packages must carry their shaders and exact license notices."""
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def test_wheel_and_sdist_include_shader_resources(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "setup.py", "LICENSE", "NOTICE.md", "README.md"):
        shutil.copy2(root / name, source / name)
    for name in ("src/trainer", "shaders"):
        shutil.copytree(root / name, source / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    result = subprocess.run(
        [sys.executable, "-c",
         "from setuptools.build_meta import build_wheel, build_sdist; "
         "build_wheel('dist'); build_sdist('dist')"],
        cwd=source, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    expected = {p.relative_to(root / "shaders").as_posix()
                for p in (root / "shaders").rglob("*.slang")}
    notices = {name: (root / name).read_bytes() for name in ("LICENSE", "NOTICE.md")}
    site = tmp_path / "installed"
    with zipfile.ZipFile(next((source / "dist").glob("*.whl"))) as wheel:
        wheel_names = set(wheel.namelist())
        assert {"trainer/shaders/" + name for name in expected} <= wheel_names
        for name, expected_bytes in notices.items():
            matches = [path for path in wheel_names
                       if ".dist-info/" in path and path.endswith("/" + name)]
            assert len(matches) == 1, matches
            assert wheel.read(matches[0]) == expected_bytes
        wheel.extractall(site)
    with tarfile.open(next((source / "dist").glob("*.tar.gz"))) as archive:
        members = {member.name.split("/", 1)[1]: member
                   for member in archive.getmembers() if "/" in member.name}
        assert {"shaders/" + name for name in expected} <= set(members)
        for name, expected_bytes in notices.items():
            stream = archive.extractfile(members[name])
            assert stream is not None
            with stream:
                assert stream.read() == expected_bytes

    check = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import trainer
from trainer.backend.compute.device import SHADERS_DIR
from trainer.backend.render.stbn import default_cache_dir
from trainer.contracts.codegen import stale_generated_slang
assert Path(trainer.__file__).is_relative_to(sys.argv[1])
assert SHADERS_DIR == Path(sys.argv[1]) / 'trainer' / 'shaders'
assert (SHADERS_DIR / 'rasterize/tile_binned/forward.slang').is_file()
assert stale_generated_slang() == ()
assert default_cache_dir() == Path.cwd() / '.stbn'
"""
    result = subprocess.run([sys.executable, "-I", "-c", check, str(site)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
