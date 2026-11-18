# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from trainer.backend.compute.execution import LINEAR_DISPATCH_WIDTH, RADIX_PART_SIZE
from trainer.backend.render.pipeline import TrainingPipeline
from trainer.backend.render.targets import TargetImages
from trainer.config.config_types import DensificationMode
from trainer.config.validation import ConfigError, validate_training_config
from trainer.data.colmap import ColmapImageEntry
from trainer.data.dataset import simple_camera
from trainer.frontend.app import TrainingApplication
from trainer.frontend.cli import build_parser
from trainer.frontend.lattice import (
    LatticeTargets, TrainingLattice, last_lattice_step, lattice_camera,
    needs_lattice_scores, needs_native_scores,
)


@pytest.mark.parametrize("stride,until", [(1, 6), (2, 2), (2, 3), (3, 8), (2, 9), (2, 99)])
def test_score_builders_follow_actual_viewport(stride, until):
    args = SimpleNamespace(iters=10, lattice_until=until, lattice_stride=stride,
                           densify_from=1, densify_to=9, sad_accum_every=3)
    lattice = set(range(1, min(9, until) + 1)) if stride > 1 else set()
    scores = {3, 6, 9}
    native = scores - lattice
    assert last_lattice_step(args) == max(lattice, default=0)
    assert needs_native_scores(args) == bool(native)
    assert needs_lattice_scores(args) == bool(scores - native)
    if not native:
        # It must return before inspecting the scene, pipeline, or any images.
        app = SimpleNamespace(args=args, _densify_mode=DensificationMode.STRUCTURE_AWARE)
        assert TrainingApplication._structure_fields(app) == {}


@pytest.mark.parametrize("flag", [("--lattice-stride", "0"), ("--lattice-until", "-1")])
def test_invalid_lattice_config(flag):
    with pytest.raises(ConfigError, match="Lattice"):
        validate_training_config(build_parser().parse_args(flag), n_cams=3, n_train=2, n_test=1)


@pytest.mark.parametrize("count", ["0", "-1", str(LINEAR_DISPATCH_WIDTH * RADIX_PART_SIZE + 1), "4294967296"])
def test_sort_capacity_fits_portable_radix_dispatch(count):
    args = build_parser().parse_args(["--max-sort-pairs", count])
    with pytest.raises(ConfigError, match="--max-sort-pairs"):
        validate_training_config(args, n_cams=3, n_train=2, n_test=1)


def test_packing_normalized_and_byte_rgb_produces_identical_opaque_targets():
    target = TargetImages(None, None, None, 11, 7)
    pixels = np.random.default_rng(11).integers(0, 256, (7, 11, 3), dtype=np.uint8)
    packed = target.pack(pixels)
    np.testing.assert_array_equal(packed[..., :3], pixels)
    assert np.all(packed[..., 3] == 255)
    np.testing.assert_array_equal(packed, target.pack(pixels.astype(np.float32) / 255))


@pytest.mark.parametrize("stride", [2, 3, 4])
def test_pixel_centers_and_camera_response_match_native(stride):
    native = simple_camera(1237, 822, (0, 0, -1))
    native.proj_matrix[:2, 2] = (0.19, -0.24)
    points = np.array([[0, .17, 1.7, 1], [-.75, .75, 3, 1], [.65, -.35, 1, 1]])

    def project(camera):
        clip = points @ camera.proj_matrix.T
        return (clip[:, :2] / clip[:, 3:] + 1) * np.array(camera.image_size) / 2

    def sensor_ray(camera, pixels):
        size = np.array(camera.image_size)
        center = (1 + camera.proj_matrix[:2, 2].astype(float)) * size / 2
        focal = camera.proj_matrix.diagonal()[:2] * size / 2
        return (pixels + .5 - center) / focal

    pixels = np.array([[0, 0], [10, 50], [127, 15]])
    for y in range(stride):
        for x in range(stride):
            phase = np.array([x, y])
            camera = lattice_camera(native, phase, stride)
            assert camera.image_size == (1237 // stride, 822 // stride)
            np.testing.assert_allclose(stride * (project(camera)-.5) + phase + .5,
                                       project(native), atol=1e-4, rtol=0)
            np.testing.assert_allclose(sensor_ray(camera, pixels),
                                       sensor_ray(native, pixels * stride + phase), atol=1e-7, rtol=0)


def test_camera_visits_cycle_independently():
    lattice = object.__new__(TrainingLattice)
    lattice.targets, lattice.visits, lattice.config = SimpleNamespace(stride=3), {}, None
    lattice.select = Mock()
    camera = simple_camera(111, 81, (0, 0, -1))
    entry = ColmapImageEntry(9, "frame", camera)
    history = {2: [], 8: []}
    for i in [8, 2, 8, 8, 2] + [2, 8] * 12:
        sampled = lattice.activate(entry, i, seed=42)
        assert sampled.image_id == 9 and sampled.camera.image_size == (37, 27)
        history[i].append(lattice.targets.phase)
    for camera_id, phases in history.items():
        assert len(set(phases[:9])) == 9
        expected = [(camera_id * 2654435761 + 42 + j) % 9 for j in range(len(phases))]
        assert [x + 3*y for x,y in phases] == expected
    assert camera.image_size == (111, 81)


def test_stream_and_resident_use_identical_odd_sized_rgba_lattice():
    target = object.__new__(LatticeTargets)
    recorder, owned, resident = Mock(), object(), object()
    TargetImages.__init__(target, None, recorder, owned, 3, 2)
    target.stride, target.phase, target._gather = 3, (2, 1), object()
    packed = np.arange(8*11*4, dtype=np.int32).reshape(8, 11, 4).astype(np.uint8)[:, ::-1]
    target.select(packed)
    target.flush()
    uploaded = recorder.upload_texture.call_args.args[1]
    np.testing.assert_array_equal(uploaded, packed[[1,4],:][:,[2,5,8]])
    assert uploaded.flags.c_contiguous and uploaded.dtype == np.uint8
    target.select(resident)
    target.flush()
    dispatch = recorder.dispatch.call_args.kwargs
    assert dispatch["source"] is resident and dispatch["target"] is owned
    assert tuple(dispatch["phase"]) == (2, 1) and tuple(dispatch["size"]) == (3, 2)
    assert dispatch["stride"] == 3 and dispatch["thread_count"] == [3, 2, 1]
    assert target.active is owned
    target.flush()
    assert recorder.dispatch.call_count == recorder.upload_texture.call_count == 1


@pytest.mark.parametrize("backend", ["vulkan", "d3d12"])
def test_gather_shader_and_stream_match_native_rgba_bytes(backend, request):
    from trainer.backend.compute.device import create_device
    from trainer.backend.compute.gpu import CommandRecorder, GpuRuntime
    from trainer.backend.compute.kernels import KernelLibrary

    try:
        device = create_device(backend=backend)
    except Exception as error:
        pytest.skip(f"{backend} unavailable: {error}")
    request.addfinalizer(device.close)
    commands = CommandRecorder(GpuRuntime(device))
    pipeline = SimpleNamespace(width=35, height=17, device=device, commands=commands, kernels=KernelLibrary(device))
    pixels = np.random.default_rng(72).integers(0, 256, (17, 35, 4), dtype=np.uint8)
    native = TargetImages(device, commands, None, 35, 17).upload(pixels)
    target = LatticeTargets(pipeline, 3)
    for source in (native, pixels):
        for x, y in [(0, 0), (2, 1), (1, 2)]:
            target.phase = (x, y)
            target.select(source)
            with commands.recording("gather_target"):
                target.flush()
            device.wait_for_idle()
            np.testing.assert_array_equal(target.active.to_numpy(), pixels[y:y+15:3, x:x+33:3])


def test_viewport_switch_restores_geometry_loss_and_pruning_state():
    from trainer.backend.compute.execution import TILE_HEIGHT, TILE_WIDTH
    p = SimpleNamespace(**{k: SimpleNamespace() for k in
                          ("rasterizer", "loss", "optimizer", "densify_stats", "simple_densifier", "structure_densifier")})
    for width, height in [(59, 17), (119, 35)]:
        config = object()
        targets = SimpleNamespace(width=width, height=height)
        TrainingPipeline.set_training_viewport(p, targets, config)
        assert p.targets is targets and p.rasterizer.config is config
        assert (p.width, p.height) == (p.loss._width, p.loss._height) == (width, height)
        assert (p.rasterizer.width, p.rasterizer.height) == tuple(p.densify_stats._frame) == (width, height)
        assert (p.simple_densifier.width, p.simple_densifier.height) == (width, height)
        assert (p.structure_densifier._width, p.structure_densifier._height) == (width, height)
        assert p.rasterizer.num_tiles_x == (width+TILE_WIDTH-1) // TILE_WIDTH
        assert p.rasterizer.num_tiles_y == (height+TILE_HEIGHT-1) // TILE_HEIGHT
        assert p.optimizer._pixels == width * height
