# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Python mirrors of shader constants agree with the Slang headers."""

from __future__ import annotations

import re

import pytest

from trainer.backend.compute import execution
from trainer.backend.compute.device import SHADERS_DIR
from trainer.common import sh

_CONSTANT = re.compile(r"^static const (?:uint|int) (\w+) = (\d+);", re.M)


def _slang_constants(relative_path: str) -> dict[str, int]:
    text = (SHADERS_DIR / relative_path).read_text(encoding="utf-8")
    return {name: int(value) for name, value in _CONSTANT.findall(text)}


def test_execution_constants_match_shader() -> None:
    shader = _slang_constants("common/execution.slang")
    python = {name: getattr(execution, name) for name in dir(execution) if name.isupper()}
    assert shader == python


def test_sh_band_limits_match_shader() -> None:
    shader = _slang_constants("primitives/gaussian/sh_layout.slang")
    assert shader == {
        "SH_MAX_DEGREE": sh.SH_MAX_DEGREE,
        "SH_REST_COEFFS": sh.SH_REST_COEFFS_MAX,
        "SH_REST_FLOATS": sh.SH_REST_FLOATS_MAX,
    }


def test_tile_area_is_consistent() -> None:
    assert execution.TILE_AREA == execution.TILE_WIDTH * execution.TILE_HEIGHT
    assert execution.TILE_AREA % execution.RASTER_BUCKET_SIZE == 0


@pytest.mark.parametrize("degree", range(sh.SH_MAX_DEGREE + 1))
def test_sh_degree_round_trips_through_rest_floats(degree: int) -> None:
    assert sh.sh_degree_from_rest_floats(sh.sh_rest_coeffs(degree) * 3) == degree
