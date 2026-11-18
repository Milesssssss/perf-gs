# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

from pathlib import Path
from runpy import run_path
import json

import pytest
from PIL import Image
from _colmap_testkit import write_colmap_scene

_CONVERTER = Path(__file__).resolve().parents[1] / "scripts/running/prepare_graphdeco.py"
_converter = run_path(str(_CONVERTER))
_check_saved_sort = _converter["_check_saved_sort"]


def test_legacy_saved_views_require_strict_spare_capacity():
    _check_saved_sort([{"tile_pairs": 99}], {"max_sort_pairs": 100})
    for view in ({"tile_pairs": 100}, {"tile_pairs": 101}, {}, {"tile_pairs": -1}):
        with pytest.raises(ValueError, match="verify"):
            _check_saved_sort([view], {"max_sort_pairs": 100})


def test_explicit_heldout_overflow_counter_disambiguates_exact_capacity():
    _check_saved_sort([{"tile_pairs": 100}], {"max_sort_pairs": 100, "heldout_sort_overflows": 0})
    with pytest.raises(ValueError, match="exceeded"):
        _check_saved_sort([{"tile_pairs": 100}], {"max_sort_pairs": 100, "heldout_sort_overflows": 1})


def test_training_only_model_uses_registered_split_and_original_dimensions(tmp_path, monkeypatch):
    run, output = tmp_path / "run", tmp_path / "model"
    scene = write_colmap_scene(tmp_path / "bicycle", [(f"{i}.png", 32, 24, 30) for i in reversed(range(9))])
    images = scene / "images_4"
    images.mkdir()
    for idx in (0, 8):
        Image.new("RGB", (8, 6), (idx, 1, 2)).save(images / f"{idx}.png")
    run.mkdir()
    config = {"scene": str(scene), "cli_args": {"iters": 3000, "eval_holdout": 1, "max_splats": 100},
              "resolved": {"n_cams": 9, "n_train": 7, "n_test": 2, "holdout_stride": 8,
                           "downsample": 4, "width": 8, "height": 6}}
    (run / "config.json").write_text(json.dumps(config))
    def log(iters, overflows):
        (run / "log.jsonl").write_text(json.dumps({"event": "train_complete", "iters": iters}) + "\n" +
            json.dumps({"event": "training_memory", "sort_overflows": overflows, "max_sort_pairs": 12000}))
    prepare = _converter["prepare"]
    model = run / "output.ply"
    selected = []
    def make_renderer(ply, args, path, factor, capacity, expected):
        assert (ply, path, factor, capacity, expected) == (model, images, 4, 12000, None)
        def render(idx, name, dest):
            selected.append((idx, name))
            Image.new("RGB", (8, 6), (0, 0, 1)).save(dest)
        return render
    monkeypatch.setitem(prepare.__globals__, "_model_renderer", make_renderer)
    log(2999, 0)
    with pytest.raises(ValueError, match="3,000"):
        prepare(run, output, ply=model)
    log(3000, 1)
    with pytest.raises(ValueError, match="overflows"):
        prepare(run, output, ply=model)
    log(3000, 0)
    method = prepare(run, output, ply=model)
    assert selected == [(0, "0.png"), (8, "8.png")]
    with Image.open(method / "gt" / "00001.png") as gt:
        assert gt.size == (8, 6) and gt.getpixel((0, 0)) == (8, 1, 2)
