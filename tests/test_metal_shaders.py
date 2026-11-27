# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Compile the trainer's Slang entry points to Metal and SPIR-V without a GPU.

slangc (Slang 2026.5.2, the compiler in slangpy 0.42.0) emits Metal source on any
host. Apple's metal compiler is not required for this check and is not run here.
Set SLANGC to the compiler binary, or put slangc on PATH. The test skips otherwise.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHADERS = ROOT / "shaders"
ENTRY = re.compile(r'\[shader\([^\)]*\)\](?:\s*\[[^\]]*\])*\s*void\s+(\w+)', re.DOTALL)

# Official presets compile with bf16 Adam moments and an alpha-over compositor.
# Ultrafast also defines STBN_ENABLED. The third set is the non-default kernel
# and the strided backward, which the CLI can still select.
DEFINE_SETS = [
    ["-DCOMPOSITOR_HARDNESS_ONE=1", "-DADAM_MOMENTS_BF16=1"],
    ["-DCOMPOSITOR_HARDNESS_ONE=1", "-DADAM_MOMENTS_BF16=1", "-DSTBN_ENABLED=1"],
    ["-DCOMPOSITOR_HARDNESS_ONE=1", "-DADAM_MOMENTS_BF16=1", "-DSPLAT_KERNEL=3",
     "-DBACKWARD_PIXEL_STRIDE=4u"],
]


def _slangc() -> str | None:
    override = os.environ.get("SLANGC")
    if override and Path(override).is_file():
        return override
    return shutil.which("slangc")


@pytest.fixture(scope="module")
def slangc() -> str:
    path = _slangc()
    if path is None:
        pytest.skip("slangc not on PATH (set SLANGC). Metal/SPIR-V source check skipped.")
    return path


def _entries() -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for path in sorted(SHADERS.rglob("*.slang")):
        text = path.read_text(encoding="utf-8")
        found.extend((path, name) for name in ENTRY.findall(text))
    assert found, "no compute entry points under shaders/"
    return found


def _compile(slangc: str, path: Path, entry: str, target: str, defines: list[str]) -> str:
    dest = Path(os.environ.get("TMPDIR", "/tmp")) / f"perfgs-{path.stem}-{entry}-{target}.out"
    cmd = [slangc, str(path), "-target", target, "-I", str(SHADERS),
           "-entry", entry, "-stage", "compute", "-o", str(dest), *defines]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"{path.relative_to(ROOT)}:{entry} -target {target} {' '.join(defines)}\n"
        f"{proc.stderr or proc.stdout}"
    )
    return dest.read_text(encoding="utf-8", errors="replace") if target == "metal" else ""


def test_metal_sources_have_no_64bit_atomics_or_oversize_bindings(slangc: str) -> None:
    for path, entry in _entries():
        for defines in DEFINE_SETS:
            metal = _compile(slangc, path, entry, "metal", defines)
            assert "atomic_long" not in metal and "atomic_ulong" not in metal, (
                f"{path.name}:{entry} emitted a 64-bit Metal atomic"
            )
            for part in metal.split("[[kernel]]")[1:]:
                signature = part.split("{", 1)[0]
                slots = [int(n) for n in re.findall(r"buffer\((\d+)\)", signature)]
                assert not slots or max(slots) <= 30, (
                    f"{path.name}:{entry} uses Metal buffer({max(slots)}); the limit is 30"
                )
                # Slang 2026.5.2 passes loose uniforms by value on wave kernels. Apple's
                # metal compiler rejects that (`uint` inside a resource parameter list).
                by_value = re.findall(
                    r"EntryPointParams_\d+\s+entryPointParams_\d+(?!\s*\*)", signature)
                assert not by_value, (
                    f"{path.name}:{entry} passes uniforms by value: {by_value}"
                )


def test_spirv_entries_still_compile(slangc: str) -> None:
    """The Vulkan target still accepts every entry, including the int64 atomic path."""
    for path, entry in _entries():
        _compile(slangc, path, entry, "spirv", DEFINE_SETS[0])


def test_structure_decision_reaches_simd_ops(slangc: str) -> None:
    metal = _compile(
        slangc,
        SHADERS / "densify" / "decide_actions_structure.slang",
        "decideActionsStructure",
        "metal",
        DEFINE_SETS[0],
    )
    kernel = metal.split("[[kernel]]", 1)[1]
    assert kernel.find("simd_") < kernel.find("return")
