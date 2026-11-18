"""CPU tests for a variable-strength response, its scene-color/opacity adjoint and PLY transport."""
from types import SimpleNamespace

import numpy as np
import pytest
from plyfile import PlyData

from trainer.backend.render.rasterizer import Rasterizer
from trainer.common.lens import COS4_SRGB_EXPONENT, parse_strength
from trainer.data.dataset import simple_camera
from trainer.data.ply import load_ply, write_gaussian_ply
from trainer.frontend.cli import build_parser


@pytest.mark.parametrize("strength", [0.0, .17, 0.8, 1.0, 1.65])
def test_upload_gain_and_compositing_adjoint(strength):
    camera = simple_camera(67, 53, eye=(0., 0., 2.))
    camera.proj_matrix[0, 2], camera.proj_matrix[1, 2] = .2, -.15
    payloads = []
    mock = SimpleNamespace(width=67, height=53, natural_vignetting=strength,
                           camera_buffer=SimpleNamespace(copy_from_numpy=lambda blob: payloads.append(blob.copy())))
    Rasterizer.upload_camera(mock, camera)
    data = payloads[0].view(np.float32)
    assert len(data) == 36 and data[35] == 0
    assert data[34] == np.float32(COS4_SRGB_EXPONENT * strength)
    if strength == 1:
        assert data[34].view(np.uint32) == 0xbf68ba2e  # accepted shader's folded floating-point constant
    rng = np.random.default_rng(154)
    for pixel in ([0, 0], [66, 52], [44, 19]):
        extent = np.array(camera.image_size)
        center = (1 + camera.proj_matrix[:2, 2]) * (.5 * extent)
        ray = (np.asarray(pixel) + .5 - center) * (2 / (extent * np.diag(camera.proj_matrix)[:2]))
        gain = max(1e-30, (1 + ray @ ray) ** data[34])
        if strength == 0:
            assert gain == 1
        if strength == 1:
            assert gain == (1 + ray @ ray) ** np.float32(COS4_SRGB_EXPONENT)
        color = rng.uniform(.01, 1.8, (5, 3))
        alpha = rng.uniform(.03, .5, 5)
        cotangent = rng.normal(size=4)
        dc, da = rng.normal(size=color.shape), rng.normal(size=alpha.shape)
        def render(c, a):
            trans = np.r_[1., np.cumprod(1 - a)]
            return np.r_[gain * ((trans[:-1] * a) @ c), 1 - trans[-1]]
        T = np.r_[1., np.cumprod(1 - alpha)]
        prefix = np.cumsum((T[:-1] * alpha)[:, None] * color, axis=0)
        raw_rgb = render(color, alpha)[:3] / gain
        rgb_grad = cotangent[:3] * gain
        grad_c = ((T[:-1] * alpha)[:, None] * rgb_grad)
        grad_a = (T[:-1, None] * color - (raw_rgb-prefix) / (1-alpha)[:, None]) @ rgb_grad
        grad_a += cotangent[3] * T[-1] / (1-alpha)
        eps = 1e-6
        numerical = (render(color+eps*dc, alpha+eps*da) - render(color-eps*dc, alpha-eps*da)) @ cotangent / (2*eps)
        assert np.isclose(grad_a @ da + np.sum(grad_c * dc), numerical, atol=1e-8, rtol=1e-8)


def test_lossless_ply_and_legacy_defaults(tmp_path):
    points = np.zeros(1, dtype=[("mean", "f4", 3), ("log_scale", "f4", 3),
                                ("rot_quat", "f4", 4), ("opacity", "f4"), ("sh0", "f4", 3)])
    points["rot_quat"][:, 0] = 1
    bodies = set()
    for b in (0., .12345678901234566, 1., 1.7):
        path = tmp_path / "gaussians.ply"
        write_gaussian_ply(points, path, natural_vignetting=b)
        assert load_ply(path).get("natural_vignetting", 0) == b
        comments = PlyData.read(path).comments
        assert bool(comments) == (b != 0)
        if b:
            assert ("_v1 " in comments[0]) == (b == 1)
        bodies.add(path.read_bytes().split(b"end_header\n",1)[1])
    assert len(bodies) == 1
    parser = build_parser()
    assert parser.parse_args([]).lens_strength is None
    assert parser.parse_args(["--lens-strength", "0.8"]).lens_strength == .8
    for value in ("nan", "-0.1", "inf"):
        with pytest.raises(ValueError):
            parse_strength(value)
