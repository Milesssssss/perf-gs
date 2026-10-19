# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The full-run comparison isolates densification and rejects unmatched runs."""

from __future__ import annotations

import json

import pytest

from bench_training_densify import (
    MeasuredController,
    MeasuredReferenceController,
    ReferencePipeline,
    benchmark_implementation,
)
from summarize_training_densify import _MATCHED_FIELDS, summarize
from trainer.frontend import app


@pytest.mark.parametrize("variant", ["reference", "optimized"])
def test_benchmark_factories_restore_on_failure(variant) -> None:
    original_pipeline = app.TrainingPipeline
    original_controller = app.DensificationController
    with pytest.raises(RuntimeError):
        with benchmark_implementation(variant):
            expected_pipeline = ReferencePipeline if variant == "reference" else original_pipeline
            expected_controller = MeasuredReferenceController if variant == "reference" else MeasuredController
            assert app.TrainingPipeline is expected_pipeline
            assert app.DensificationController is expected_controller
            raise RuntimeError("benchmark failed")
    assert app.TrainingPipeline is original_pipeline
    assert app.DensificationController is original_controller


def _write_run(tmp_path, name, variant, seconds, **changes):
    run = dict.fromkeys(_MATCHED_FIELDS, "matched")
    run.update({
        "variant": variant, "train_wall_s": seconds, "train_active_s": seconds,
        "settings": {"log_every": 0, "preview_every": 0, "eval_every": 0,
                     "psnr_curve": False, "stage_timing_every": 0, "profile_run": False},
        "quality_outside_timing": {"n_gaussians": 1234, "mean_sample_psnr_srgb": 25.0},
        "densification_ticks": [{"host_ms": 10.0, "overbudget": False}],
        **changes,
    })
    path = tmp_path / (name + ".json")
    path.write_text(json.dumps(run), encoding="utf-8")
    return path


def test_abba_summary_computes_training_speedup(tmp_path) -> None:
    paths = [
        _write_run(tmp_path, "a1", "reference", 90.0),
        _write_run(tmp_path, "b1", "optimized", 60.0),
        _write_run(tmp_path, "b2", "optimized", 62.0),
        _write_run(tmp_path, "a2", "reference", 92.0),
    ]
    result = summarize(paths)
    assert result["speedup"] == pytest.approx(91 / 61)
    assert result["seconds_saved"] == 30
    assert result["time_reduction_percent"] == pytest.approx(100 * 30 / 91)
    assert result["sample_psnr_delta_db"] == 0
    assert result["order"] == ["reference", "optimized", "optimized", "reference"]


def test_summary_rejects_changed_sources(tmp_path) -> None:
    a = _write_run(tmp_path, "a", "reference", 90.0)
    b = _write_run(tmp_path, "b", "optimized", 60.0, source_fingerprint="different")
    with pytest.raises(ValueError, match="source_fingerprint"):
        summarize([a, b])


def test_summary_rejects_in_loop_evaluation(tmp_path) -> None:
    a = _write_run(tmp_path, "a", "reference", 90.0)
    b = _write_run(tmp_path, "b", "optimized", 60.0, train_active_s=50.0)
    for path in (a, b):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["settings"]["eval_every"] = 100
        path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="observation"):
        summarize([a, b])


def test_summary_excludes_small_cpu_observation_cost(tmp_path) -> None:
    a = _write_run(tmp_path, "a", "reference", 13.0, train_active_s=12.94)
    b = _write_run(tmp_path, "b", "optimized", 12.0, train_active_s=11.94)
    result = summarize([a, b])
    assert result["speedup"] == pytest.approx(12.94 / 11.94)
    assert result["variants"]["reference"]["mean_wall_seconds"] == 13.0
    assert result["variants"]["reference"]["cpu_observation_seconds"] == pytest.approx([0.06])