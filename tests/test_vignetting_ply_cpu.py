"""The lens-law opt-in is portable without changing Gaussian vertex properties."""
from types import SimpleNamespace

import numpy as np
from plyfile import PlyData

from trainer.backend.export import save_ply
from trainer.backend.render.pipeline import TrainingPipeline
from trainer.data.ply import load_ply
from trainer.frontend.cli import build_parser


def test_ply_header_roundtrip_preserves_payload_and_restores_pipeline_flag(tmp_path):
    records = np.zeros(2, dtype=[("mean", "f4", 3), ("log_scale", "f4", 3),
                                 ("rot_quat", "f4", 4), ("opacity", "f4"), ("sh0", "f4", 3)])
    records["rot_quat"][:, 0] = 1
    records["mean"][:, 0] = [0.25, -0.5]
    pipe = SimpleNamespace(rasterizer=SimpleNamespace(natural_vignetting=False),
                           buffers=SimpleNamespace(sh_enabled=False),
                           primitives=SimpleNamespace(read=lambda: records),
                           optimizer=SimpleNamespace(clear_state=lambda: None, reset=lambda: None),
                           set_gaussians=lambda *args: None)
    payloads = []
    for enabled in (False, True):
        path = tmp_path / f"{enabled}.ply"
        pipe.rasterizer.natural_vignetting = enabled
        save_ply(pipe, path)
        header, payload = path.read_bytes().split(b"end_header\n", 1)
        assert (b"natural_vignetting_v1" in header) == enabled
        assert b"property float x" in header
        payloads.append(payload)
        loaded = load_ply(path)
        assert ("natural_vignetting" in loaded) == enabled
        pipe.rasterizer.natural_vignetting = not enabled
        TrainingPipeline.load_snapshot(pipe, **loaded)
        assert pipe.rasterizer.natural_vignetting == enabled
        assert np.array_equal(PlyData.read(path)["vertex"].data["x"], records["mean"][:, 0])
    assert payloads[0] == payloads[1]
    assert not build_parser().parse_args([]).natural_vignetting
    assert build_parser().parse_args(["--natural-vignetting"]).natural_vignetting
