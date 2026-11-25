# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

import csv
import json
import shutil

import pytest

import score_graphdeco as scorer
from trainer.frontend.settings import preset_values


def test_quality_plus_uses_nine_quality_camera_sets_and_requires_a_real_plateau(tmp_path, monkeypatch):
    source = scorer.ROOT / "docs/mipnerf360_3k_presets.csv"
    (tmp_path / "docs").mkdir()
    shutil.copyfile(source, tmp_path / "docs" / source.name)
    monkeypatch.setattr(scorer, "ROOT", tmp_path)
    monkeypatch.setattr(scorer.subprocess, "check_output", lambda *_a, **_k: scorer.REFERENCE_COMMIT)
    monkeypatch.setattr(scorer.subprocess, "run", lambda *_a, **_k: None)
    monkeypatch.setattr(scorer.filecmp, "cmp", lambda *_a, **_k: True)

    models = []
    with source.open(newline="") as f:
        for record in csv.DictReader(f):
            if record["preset"] != "quality":
                continue
            scene = record["scene"]
            w, h, count = (int(record[k]) for k in ("train_width", "train_height", "heldout_views"))
            views = [dict(cam_idx=i * 8, width=w, height=h, render_png=f"{i}.png") for i in range(count)]
            completed = 10_000 + count
            report = dict(
                run_name=scene, scene_path=scene, iters_requested=0, iters_completed=completed, stop_reason="plateau",
                device=dict(api_name="Vulkan", adapter_name="gpu"),
                cli_args=dict(preset_values("quality+"), preset="quality+", seed=42, no_viewer=True,
                              eval_every=0, log_every=0, preview_every=0, eval_iters=None, psnr_curve=False),
                scene=dict(eval_holdout=True, holdout_stride=8, n_test=count, width=w, height=h,
                           downsample=2 if scene in ("bonsai", "counter", "kitchen", "room") else 4),
                quality=dict(per_view=views), model=dict(sh_degree=3, n_gaussians=1),
                stbn=dict(pixel_fraction=1.0), timing=dict(train_wall_s=1.0))
            run = tmp_path / "outputs/train" / scene
            run.mkdir(parents=True)
            (run / "run_report.json").write_text(json.dumps(report))
            (run / "test_psnr.json").write_text(json.dumps(dict(iters=completed, train_wall_s=1.0,
                sort_overflows=0, heldout_sort_overflows=0, n_test_cams=count)))
            models.append(tmp_path / "models" / scene)

    _, rows = scorer.preset_rows(models, tmp_path, "gpu", quality_plus_only=True)
    assert len(rows) == 9 and sum(int(r["heldout_views"]) for r in rows.values()) == 246
    assert all(r["steps"] == 10_000 + r["heldout_views"] for r in rows.values())
    with pytest.raises(ValueError, match="Expected 27"):
        scorer.preset_rows(models, tmp_path, "gpu")
    for reason, requested in (("iteration_limit", 0), ("interrupted", 0), ("plateau", 30_000)):
        (run / "run_report.json").write_text(json.dumps(dict(report, stop_reason=reason, iters_requested=requested)))
        with pytest.raises(ValueError, match="training length"):
            scorer.preset_rows(models, tmp_path, "gpu", quality_plus_only=True)
