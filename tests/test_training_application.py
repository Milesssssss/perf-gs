"""Focused contracts for local training-application configuration objects."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import numpy as np
from PIL import Image

from trainer.config.config import (
    AppearanceConfig,
    CompositorConfig,
    LossConfig,
    RasterConfig,
    RegularizationConfig,
)
from _colmap_testkit import write_colmap_scene

from trainer.frontend.evaluation.evaluation import render_all_cameras
from trainer.frontend import app as app_module
from trainer.frontend.viewer.session import (
    MODE_HUB,
    DatasetState,
    ResultState,
    Session,
)
from trainer.frontend.app import (
    ShaderSpecialization,
    TargetDataset,
    TrainingApplication,
    TrainingLoop,
    _resolve_position_lr_max_steps,
    create_pipeline,
    create_training_device,
)


def test_default_shader_specialization_uses_shader_default() -> None:
    # No stride override and hardness NOT specialized (default) => no defines.
    specialization = ShaderSpecialization(backward_pixel_stride=1)
    assert specialization.compiler_defines() is None


def test_strided_backward_specialization_emits_slang_uint_define() -> None:
    specialization = ShaderSpecialization(backward_pixel_stride=4)
    assert specialization.compiler_defines() == {
        "BACKWARD_PIXEL_STRIDE": "4u",
    }


def test_hardness_one_specialization_emits_define() -> None:
    spec = ShaderSpecialization(backward_pixel_stride=1, hardness_is_one=True)
    assert spec.compiler_defines() == {"COMPOSITOR_HARDNESS_ONE": "1"}


def test_from_args_specializes_hardness_when_target_le_one() -> None:
    on = ShaderSpecialization.from_args(
        Namespace(backward_pixel_stride=1, extinction_hardness=1.0))
    assert on.compiler_defines() == {"COMPOSITOR_HARDNESS_ONE": "1"}
    off = ShaderSpecialization.from_args(
        Namespace(backward_pixel_stride=1, extinction_hardness=2.0))
    assert off.compiler_defines() is None


def test_hub_never_compiles_the_hardness_away() -> None:
    """The define lands in ``compiler_options`` when the DEVICE is created, which
    in hub mode happens before the TRAINING SETUP panel exists. Specializing on
    the launch value there would silently pin a panel-set k > 1 to alpha-over."""
    hub = ShaderSpecialization.from_args(
        Namespace(backward_pixel_stride=1, extinction_hardness=1.0, hub=True))
    assert hub.compiler_defines() is None


def test_position_lr_horizon_follows_iters_unless_overridden() -> None:
    """``--position-lr-max-steps`` defaults to 0 == follow ``--iters``; it used to
    be a hard 3000 that silently disagreed with any other ``--iters``."""
    assert _resolve_position_lr_max_steps(0, 7000) == 7000
    assert _resolve_position_lr_max_steps(-1, 7000) == 7000
    assert _resolve_position_lr_max_steps(1500, 7000) == 1500


def test_device_factory_passes_resolved_specialization(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_create_device(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("trainer.frontend.app.create_device", fake_create_device)
    args = Namespace(
        backend="vulkan",
        profile_run=False,
        backward_pixel_stride=2,
        extinction_hardness=2.0,   # > 1 => hardness path NOT specialized
    )

    assert create_training_device(args) is sentinel
    assert captured == {
        "backend": "vulkan",
        # Resolved from --nsight-debug-names (default None -> off unless
        # --capture-external); absent on this bare Namespace so it's False.
        "enable_debug": False,
        "shader_debug_info": False,
        "defines": {"BACKWARD_PIXEL_STRIDE": "2u"},
    }


def test_pipeline_factory_separates_configuration_ownership(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("trainer.frontend.app.Pipeline", fake_pipeline)
    args = Namespace(
        max_splats=123,
        z_near_cull=0.25,
        cluster_cull=False,
        mip_splatting=True,
        mip_kernel_size=0.1,
        sh_degree=0,
        sh_band_interval=1000,
        extinction_hardness=2.0,
        w_alpha=0.4,
        opacity_reg=0.03,
        scale_reg=0.07,
    )
    scene = Namespace(width=64, height=32)

    result = create_pipeline(
        args=args,
        device="device",
        gpu_lock="lock",
        scene=scene,
        lambda_l2=1.5,
        lambda_ssim=0.2,
    )

    assert result is sentinel
    assert captured["raster_config"] == RasterConfig(
        z_near_cull=0.25,
        cluster_cull=False,
        mip_enabled=True,
        kernel_size=0.1,
    )
    assert captured["compositor_config"] == CompositorConfig(
        extinction_hardness=2.0)
    assert captured["loss_config"] == LossConfig(
        alpha_weight=0.4, ssim_weight=0.2, l2_weight=1.5)
    assert captured["regularization_config"] == RegularizationConfig(
        opacity_weight=0.03, scale_weight=0.07)
    assert captured["appearance_config"] == AppearanceConfig(
        sh_degree=0, band_interval=1000)


def test_training_loop_presents_after_full_iteration() -> None:
    events = []
    loop = object.__new__(TrainingLoop)
    states = iter((True, False))
    context = object()

    loop._begin_training_loop = lambda: events.append("begin")
    loop._should_continue = lambda: next(states)
    loop._next_iteration = lambda: events.append("next") or context
    loop._prepare_iteration = lambda ctx: events.append("prepare")
    loop._capture_if_requested = lambda ctx: events.append("capture")
    loop._execute_training_step = lambda ctx: events.append("train")
    loop._apply_mutations = lambda ctx: events.append("mutations")
    loop._produce_observations = lambda ctx: events.append("observations") or True
    loop._present_profile_frame_if_enabled = lambda: events.append("present")
    loop._complete_training_loop = lambda: events.append("complete") or "result"

    assert TrainingLoop.launch_training_loop(loop) == "result"
    assert events == [
        "begin", "next", "prepare", "capture", "train",
        "mutations", "observations", "present", "complete",
    ]


def test_per_camera_renders_do_not_depend_on_the_ground_truth_dump(tmp_path) -> None:
    """render_all_cameras must create its own cam_XX/ dirs.

    It used to rely on the GT dump having made them, so a run with GT writing
    disabled died with FileNotFoundError at the first render_all iteration —
    thousands of iterations in, after all the training work was already done.
    """
    frame = np.zeros((4, 6, 4), dtype=np.float32)
    pipeline = SimpleNamespace(
        render=lambda camera: None,
        read_output_image=lambda: frame,
    )
    scene = SimpleNamespace(images=[SimpleNamespace(camera=object()) for _ in range(3)])
    cams_dir = tmp_path / "cams"          # deliberately never created
    cams_dir.mkdir()

    render_all_cameras(pipeline, scene, 3000, cams_dir)

    assert sorted(p.name for p in cams_dir.iterdir()) == ["cam_00", "cam_01", "cam_02"]
    for cam in range(3):
        assert (cams_dir / f"cam_{cam:02d}" / "iter_003000.png").is_file()


def test_ground_truths_land_one_per_camera_and_are_not_rewritten(tmp_path) -> None:
    targets = object.__new__(TargetDataset)
    targets._use_alpha = False
    targets._srgb = [np.full((3, 5, 3), i / 4.0, dtype=np.float32) for i in range(4)]
    cams_dir = tmp_path / "cams"
    cams_dir.mkdir()

    targets.write_ground_truths(cams_dir)
    written = sorted(p.relative_to(cams_dir).as_posix() for p in cams_dir.rglob("*.png"))
    assert written == [f"cam_{i:02d}/gt.png" for i in range(4)]

    # Order must survive the threaded write: cam_i's gt is target i, not a neighbour.
    for i in range(4):
        pixels = np.asarray(Image.open(cams_dir / f"cam_{i:02d}" / "gt.png"))
        assert pixels.shape == (3, 5, 3)
        assert np.all(pixels == round(i / 4.0 * 255))

    stamps = {p: p.stat().st_mtime_ns for p in cams_dir.rglob("*.png")}
    targets.write_ground_truths(cams_dir)      # rerun: an existing gt is left alone
    assert {p: p.stat().st_mtime_ns for p in cams_dir.rglob("*.png")} == stamps


class _FakeHubViewer:
    """Records what the hub asks the viewer to show.

    Wraps a REAL :class:`Session` rather than a stub, so what the hub publishes
    (dataset lifecycle, notices) is asserted as the panel would read it.
    """

    def __init__(self) -> None:
        self.state = Session(MODE_HUB)
        self.previewed: dict | None = None
        self.attached: list | None = None
        self.snapped = False
        self.suspend_depth = 0
        self.suspended_during_upload = False

    @property
    def messages(self) -> list[str]:
        return list(self.state.snapshot().log)

    @property
    def notices(self) -> list[str]:
        return [n.text for n in self.state.snapshot().notices]

    def suspend_rendering(self) -> None:
        self.suspend_depth += 1

    def resume_rendering(self) -> None:
        self.suspend_depth -= 1

    def preview_points(self, initial_points: dict) -> int:
        self.previewed = initial_points
        self.suspended_during_upload = self.suspend_depth > 0
        return int(initial_points["means"].shape[0])

    def attach_scene(self, cameras, snap: bool = False) -> None:
        self.attached = list(cameras)
        self.snapped = snap


def _hub_app(viewer: _FakeHubViewer) -> TrainingApplication:
    app = object.__new__(TrainingApplication)
    app.viewer = viewer
    app.args = Namespace(width=None, height=None, downsample=1, init_opacity=0.1)
    return app


def test_dropping_a_colmap_dataset_previews_its_point_cloud(tmp_path) -> None:
    """The SfM cloud has to be on screen BEFORE Train: that is the only chance to
    see a broken reconstruction without paying for a full run first."""
    write_colmap_scene(tmp_path / "recon")
    viewer = _FakeHubViewer()

    _hub_app(viewer)._hub_preview_scene(str(tmp_path / "recon"))

    assert viewer.previewed is not None
    assert viewer.previewed["means"].shape == (8, 3)
    # Uploading writes the buffers the render thread reads.
    assert viewer.suspended_during_upload
    assert viewer.suspend_depth == 0, "rendering left suspended"
    # EVERY camera, not the eval split — before a run you fly the whole capture.
    assert [name for name, _cam in viewer.attached] == ["a.png", "b.png", "c.png",
                                                        "d.png"]
    assert viewer.snapped
    assert any("press Train" in m for m in viewer.messages)
    # ... and the panel can now say WHAT was loaded, not just that something was.
    snap = viewer.state.snapshot()
    assert snap.dataset_state is DatasetState.READY
    assert snap.dataset.images == 4
    assert snap.result_state is ResultState.INITIALIZED   # points, not a result
    assert any("loaded" in n for n in viewer.notices)


def test_the_sparse_model_is_parsed_with_rendering_suspended(
        tmp_path, monkeypatch) -> None:
    """Suspension has to cover the PARSE too, not just the buffer upload.

    ``load_colmap``'s binary readers are per-record Python loops and a painting
    frame holds the GIL through its submit wait, so a 0.6s parse next to a live
    render thread took over 40s and the drop looked hung.
    """
    write_colmap_scene(tmp_path / "recon")
    viewer = _FakeHubViewer()
    real_load = app_module.load_colmap
    depth_during_load: list[int] = []

    def spy(*args, **kwargs):
        depth_during_load.append(viewer.suspend_depth)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(app_module, "load_colmap", spy)

    _hub_app(viewer)._hub_preview_scene(str(tmp_path / "recon"))

    assert depth_during_load == [1], "parsed with the render thread still painting"
    assert viewer.suspend_depth == 0, "rendering left suspended"


def test_a_dataset_the_loader_rejects_leaves_the_hub_alive(tmp_path) -> None:
    """A preview failure must not kill the process — the user still has to be able
    to drop a different folder or reconstruct again."""
    viewer = _FakeHubViewer()

    _hub_app(viewer)._hub_preview_scene(str(tmp_path / "not-a-scene"))

    assert viewer.previewed is None
    assert viewer.suspend_depth == 0
    assert any("cannot preview" in m for m in viewer.messages)
    # The failure belongs to the DATASET, with a reason the panel can show.
    snap = viewer.state.snapshot()
    assert snap.dataset_state is DatasetState.ERROR
    assert snap.dataset_error
    assert snap.can_drop, "the hub must still accept another folder"
