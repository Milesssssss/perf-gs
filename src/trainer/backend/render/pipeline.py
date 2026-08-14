from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import slangpy as spy

from trainer.backend.memory import buffers
from trainer.backend.memory import buffer_io
from trainer.backend.densify import densify as _densify
from trainer.backend.memory.buffers import (
    CLUSTER_SIZE,
    GRAD_SCALAR_COMPONENTS,
    RADIX_GLOBAL_HIST_SIZE,
    RADIX_PART_SIZE,
)
from trainer.config.config import (
    AdamLRs,
    AppearanceConfig,
    CompositorConfig,
    LossConfig,
    RasterConfig,
    RegularizationConfig,
    StbnConfig,
    StbnVariables,
)

from trainer.config.config_types import DensificationMode
from trainer.contracts.codegen import sh_rest_coeffs, TILE_WIDTH, TILE_HEIGHT, TILE_AREA
from trainer.data.dataset import Camera
from trainer.backend.densify.densify import DensifyConfig
from trainer.backend.compute.device import SHADERS_DIR
from trainer.backend.compute.gpu import CommandBatch, GpuRuntime
from trainer.backend.compute.gpu_scan import GpuScan, ScanKernels
from trainer.backend.render import stbn as _stbn
from trainer.backend.render.kernels import load_kernels
from trainer.backend.render.losses import ImageLoss
from trainer.backend.optimize.adam import AdamOptimizer
from trainer.backend.densify.simple_densifier import (
    CommandInterface,
    SimpleDensifier,
    SimpleDensifyKernels,
)

from trainer.config.config import IterationContext, IterationContextualVariables

_RADIX_SCAN_DIM = 128


def _srgb01_to_u8(rgba: np.ndarray) -> np.ndarray:
    """Quantize a float [0,1] sRGB RGBA image to contiguous uint8 for upload
    into an rgba8_unorm texture (round-to-nearest, clamped). If the input is
    already uint8 it's passed through unchanged. The GT originates as uint8/255,
    so this is lossless for un-resized pixels and re-quantizes interpolated ones
    to the source's own 1/255 precision."""
    if rgba.dtype == np.uint8:
        return np.ascontiguousarray(rgba)
    u8 = np.clip(np.rint(rgba.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(u8)


class _RecordingTransaction:
    """A recorded batch plus host-state commits that run after submit succeeds."""

    def __init__(self, batch: CommandBatch) -> None:
        self.batch = batch
        self._commits: list[Callable[[], None]] = []

    def after_submit(self, commit: Callable[[], None]) -> None:
        self._commits.append(commit)

    def commit(self) -> None:
        for commit in self._commits:
            commit()


def _module_for_reflection(device: spy.Device) -> spy.Module:
    """
    Load the NEUTRAL shared-layout module (``common/types.slang``) so
    buffers.py can reflect the primitive/projected/optimizer struct layouts
    (``Gaussian3D`` / ``Projected2DGaussian`` / ``BlendData`` / ``AdamState``)
    """
    return spy.Module.load_from_file(
        device, str(SHADERS_DIR / "common" / "types.slang")
    )


class TrainingPipeline:
    def __init__(
        self,
        device: spy.Device,
        width: int,
        height: int,
        max_splats: int = 500_000,
        max_sort_pairs: int | None = None,
        gpu_lock: "contextlib.AbstractContextManager | None" = None,
        raster_config: RasterConfig | None = None,
        loss_config: LossConfig | None = None,
        regularization_config: RegularizationConfig | None = None,
        appearance_config: AppearanceConfig | None = None,
        stbn_config: StbnConfig | None = None,
        viewer_only: bool = False,
        initial_state: dict | None = None,
        use_persplat_backward: bool = True,
        use_flat_bucket_backward: bool = True,
        densify_mode: DensificationMode | None = None,
    ) -> None:
        self.device = device
        self.gpu_lock = gpu_lock if gpu_lock is not None else contextlib.nullcontext()
        self.runtime = GpuRuntime(device, self.gpu_lock)
        self.width = width
        self.height = height
        self.max_splats = max_splats

        if max_sort_pairs is None:
            max_sort_pairs = max(buffers.DEFAULT_MAX_SORT_PAIRS, max_splats * 4)

        self.max_sort_pairs = max_sort_pairs

        self.shared_data_module = _module_for_reflection(device)
        self.viewer_only = viewer_only
        self.raster_config = raster_config or RasterConfig()

        if viewer_only and self.raster_config.cluster_cull:
            self.raster_config = replace(self.raster_config, cluster_cull=False)

        self.regularization_config = (regularization_config or RegularizationConfig())
        resolved_loss_config = loss_config or LossConfig()

        self.appearance_config = appearance_config or AppearanceConfig()
        self._sh_rest_floats = sh_rest_coeffs(self.appearance_config.sh_degree) * 3

        # STBN pixel subsampling. The table is built/uploaded ONCE (a few MB,
        # content-addressed on disk); only the slice + per-tile budget vary per
        # iteration. A disabled run still gets a 1-element placeholder because
        # the kernels declare the parameter unconditionally.
        self.stbn_config = stbn_config or StbnConfig()
        self._stbn = self._create_stbn_mask()

        load_kernels(device, self, viewer_only=viewer_only)

        allocate_fn = buffers.allocate_viewer if viewer_only else buffers.allocate

        self.buffers = allocate_fn(
            device=device, shared_data_module=self.shared_data_module,
            max_splats=max_splats, width=width, height=height,
            max_sort_pairs=max_sort_pairs,
            sh_rest_floats=self._sh_rest_floats,
        )

        self.image_loss = None if viewer_only else ImageLoss(
            ssim_maps_kernel=self.k_ssim_maps,
            ssim_grad_kernel=self.k_ssim_grad,
            loss_kernel=self.k_loss,
            ssim_m1=self.buffers.ssim_m1,
            ssim_m2=self.buffers.ssim_m2,
            ssim_m3=self.buffers.ssim_m3,
            image_gradient=self.buffers.dL_dC_texture,
            width=width, height=height, runtime=self.runtime,
            config=resolved_loss_config,
        )

        # Camera buffer: raw 144-byte buffer (136 data + 8 padding to 16-byte align)
        self._camera_buffer = device.create_buffer(
            size=144,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label="camera_data",
        )

        self._adam = None if viewer_only else AdamOptimizer(
            dispatch=self._dispatch,
            dispatch_indirect=self._dispatch_indirect,
            prepare_args_kernel=self.k_densify_prepare_args,
            adam_kernel=self.k_adam,
            clear_adam_state=self.k_clear_adam_state,
            adam_rest_kernel=getattr(self, "k_adam_rest", None),
            buffers=self.buffers,
            max_splats=max_splats,
            width=width,
            height=height,
            marker_group=self._marker_group,
        )

        # Active command batch during a recorded step. When set, _dispatch /
        # _marker_group record into it; None => immediate dispatch
        self._batch: CommandBatch | None = None
        self.scan_service = GpuScan(
            resources=self.buffers.scan,
            kernels=ScanKernels(
                scan_block=self.k_scan_block,
                scan_block_sums=self.k_scan_block_sums,
                add_block_offsets=self.k_add_block_offsets,
            ),
            runtime=self.runtime,
            active_batch=lambda: self._batch,
        )
        self.simple_densifier = SimpleDensifier(
            primitive=self.buffers.primitive,
            adam=self.buffers._require_adam(),
            resources=self.buffers._require_densification(),
            eta=self.buffers.eta,
            scan=self.scan_service,
            kernels=SimpleDensifyKernels(
                decide_actions=self.k_decide_actions_simple,
                snapshot_count=self.k_densify_snapshot,
                prepare_dispatch=self.k_densify_prepare_args,
                split_and_clone=self.k_split_and_clone,
                build_prune_mask=self.k_build_prune_mask,
                scatter_compact=self.k_scatter_compact,
                scatter_compact_eta=self.k_scatter_compact_eta,
                split_and_clone_sh=self.k_split_and_clone_sh,
                scatter_compact_sh=self.k_scatter_compact_sh,
                scatter_compact_gaussian=self.k_scatter_compact_gaussian,
                scatter_compact_adam=self.k_scatter_compact_adam,
                scatter_compact_features=self.k_scatter_compact_features,
                scatter_compact_adam_rest=self.k_scatter_compact_adam_rest,
                scatter_compact_eta_only=self.k_scatter_compact_eta_only,
            ),
            commands=CommandInterface(
                dispatch=self.dispatch,
                dispatch_indirect=self.dispatch_indirect,
                recording=self.recording,
                marker_group=self.marker_group,
                copy_arena_back=self.copy_arena_back,
            ),
            width=self.width,
            height=self.height,
            max_primitives=self.max_splats,
            sh=self.buffers.sh,
        ) if not viewer_only else None

        self.last_submit_id: int = 0

        assert use_persplat_backward or not use_flat_bucket_backward, \
            "use_flat_bucket_backward=True requires use_persplat_backward=True"

        self.use_persplat_backward = use_persplat_backward
        self.use_flat_bucket_backward = use_flat_bucket_backward

        if self.raster_config.mip_enabled and not self.use_persplat_backward:
            raise RuntimeError(
                "mip_enabled requires use_persplat_backward=True: the per-pixel "
                "backward does not propagate opacity-compensation gradients.")

        #starts at 0, increments after each bwd step
        self._backward_pixel_phase = 0

        # Applied render-state: the LAST values a train_step applied to the
        # pipeline. The frontend owns the SCHEDULE and injects per-step values
        # via IterationContextualVariables; renders outside a training step
        # (eval, export, viewer, bench) inherit whatever was last applied.
        # Default to 1.0 (standard alpha-over) — the neutral value the
        # ExtinctionHardnessSchedule holds before its warmup ramp starts.
        self._active_sh_degree: int = 0
        self._extinction_hardness: float = 1.0

        self._active_target = None if viewer_only else self.buffers.target_image

        self._sorted_values = self.buffers.sort_values_a

        self._mip_cameras_buffer: spy.Buffer | None = None
        self._mip_num_cameras: int = 0
        self._mip_filter_coef: float = 0.0

        self._step_timer = None

        if initial_state is not None:
            self._load_snapshot(**initial_state)
            self._clear_grads()
            self.clear_tracking()
            if densify_mode == DensificationMode.STRUCTURE_AWARE:
                self.clear_structure_scores()

    def _create_stbn_mask(self) -> "_stbn.StbnMask":
        """Resolve the run's STBN table (or the disabled placeholder).

        The viewer never runs a loss or a backward, so it always takes the
        placeholder — no reason to spend megabytes on a mask nothing reads.
        """
        config = self.stbn_config
        if self.viewer_only or not config.enabled:
            return _stbn.upload_disabled_mask(self.device)
        spec = _stbn.StbnMaskSpec(
            source=config.source, path=config.file_path, seed=config.seed)
        mask = _stbn.upload_mask(
            self.device, spec, _stbn.default_cache_dir() / "cache")
        print(f"[stbn] {spec.source} mask {spec.width}x{spec.height}x{spec.slices}"
              f", {_stbn.pixels_per_tile(config.pixel_fraction, TILE_AREA)}"
              f"/{TILE_AREA} pixels per tile at rho={config.pixel_fraction:g}"
              f"{' (compacted backward)' if config.backward_compacted else ''}")
        return mask

    @property
    def stbn_slices(self) -> int:
        """Temporal period of the mask — the frontend wraps its per-camera visit
        counters against this."""
        return self._stbn.slices

    def _stbn_kwargs(self, contextual: IterationContextualVariables) -> dict:
        """The mask parameters for this iteration's kernels.

        A step with no STBN variables (tests, eval, bench, viewer) resolves to
        the full budget, i.e. every pixel sampled — so an unaware caller gets the
        exact full-pixel behavior rather than a silently sparse gradient.
        """
        variables = contextual.stbn if contextual is not None else None
        if variables is None or variables.pixels_per_tile <= 0:
            return self._stbn.shader_kwargs(0, TILE_AREA)
        # The shaders stage a full tile and clamp to it, so any budget in
        # [1, TILE_AREA] is representable; nothing here can overrun the staging.
        return self._stbn.shader_kwargs(
            variables.slice_index,
            min(int(variables.pixels_per_tile), TILE_AREA))

    def on_hot_reload(self) -> None:
        """Refresh reflection-dependent objects after device.reload_all_programs().
        Camera buffer is raw bytes so it survives reload. Only the module
        reference needs refreshing (for any code that uses reflected types)."""
        self.shared_data_module = _module_for_reflection(self.device)

    @property
    def _encoder(self) -> "spy.CommandEncoder | None":
        """Compat view of the active command batch's raw encoder. External
        encoder owners (densify, viewer) assign a raw encoder here; the setter
        wraps it in a CommandBatch so ALL dispatch routing goes through one
        path. Internal code uses ``self._batch`` directly."""
        return self._batch.encoder if self._batch is not None else None

    @_encoder.setter
    def _encoder(self, encoder: "spy.CommandEncoder | None") -> None:
        self._batch = CommandBatch(encoder) if encoder is not None else None

    @property
    def iteration(self) -> int:
        """Adam bias-correction step count; densification reads it as a
        decorrelation seed. The counter itself lives in the optimizer."""
        return self._adam.step_count if self._adam is not None else 0

    def set_active_sh_degree(self, degree: int) -> None:
        self._active_sh_degree = max(0, min(int(degree), self.appearance_config.sh_degree))

    def set_extinction_hardness(self, hardness: float) -> None:
        self._extinction_hardness = float(hardness)

    def _dispatch(self, kernel, marker: str, **kwargs):
        """Dispatch a kernel, tagging it with a flat debug marker ``marker`` when
        a command batch is active (train_step / densify / viewer); otherwise an
        immediate dispatch. Recording is delegated to the active CommandBatch."""
        if self._batch is not None:
            self._batch.dispatch(kernel, marker, **kwargs)
        else:
            transient = self.runtime.command_batch(marker)
            transient.dispatch(kernel, marker, **kwargs)
            self.last_submit_id = self.runtime.submit(transient).submit_id

    def _dispatch_indirect(self, kernel, marker: str, args_buf: spy.NDBuffer,
                           offset: int = 0, **kwargs) -> None:
        """Indirect compute dispatch (group count read from ``args_buf`` at
        ``offset`` as three uint32s). Records into the active batch; if none is
        active (e.g. from render()) a transient one-shot batch is created and
        submitted so this works from any caller. The write-visibility global
        barrier before the indirect fetch lives in CommandBatch.dispatch_indirect."""
        if self._batch is not None:
            self._batch.dispatch_indirect(kernel, marker, args_buf, offset, **kwargs)
        else:
            transient = self.runtime.command_batch(marker)
            transient.dispatch_indirect(kernel, marker, args_buf, offset, **kwargs)
            self.last_submit_id = self.runtime.submit(transient).submit_id

    def copy_arena_back(self, arena, dst: spy.NDBuffer, nbytes: int) -> None:
        """Record an arena->live buffer byte copy into the active batch (commits
        one resource's compaction/re-sort; see ScratchArena.copy_back). Must be
        called with a batch active (inside a recording), which the densify /
        morton paths always are; sgl auto-synchronizes the preceding scatter
        write and the subsequent arena reuse."""
        assert self._batch is not None, "copy_arena_back requires an active batch"
        arena.copy_back(self._batch.encoder, dst, nbytes)

    @contextlib.contextmanager
    def _marker_group(self, label: str):
        """Nested debug-group scope for profiler captures (Nsight etc.). No-op
        (nullcontext) when no batch is active, so it costs nothing outside a
        recorded step; otherwise delegates to CommandBatch.marker_group.

        When a StepTimer is installed (benchmark only), also brackets the scope
        with GPU timestamps so every marker group is timed with no extra call
        sites. The timing writes go on the active batch's encoder alongside the
        debug-group push/pop."""
        if self._batch is None:
            yield
            return
        timer = self._step_timer
        interval_id = -1
        if timer is not None:
            interval_id = timer.open(label, self._batch.encoder)
        with self._batch.marker_group(label):
            try:
                yield
            finally:
                if timer is not None:
                    timer.close(interval_id, self._batch.encoder)

    # Public command contract for extracted GPU services. The private aliases
    # remain while Pipeline's internal implementation is migrated incrementally.
    def dispatch(self, kernel, marker: str, **kwargs) -> None:
        self._dispatch(kernel, marker, **kwargs)

    def dispatch_indirect(self, kernel, marker: str, args_buf: spy.NDBuffer,
                          offset: int = 0, **kwargs) -> None:
        self._dispatch_indirect(kernel, marker, args_buf, offset, **kwargs)

    def marker_group(self, label: str):
        return self._marker_group(label)

    def recording(self, label: str):
        return self._recording(label)

    @property
    def camera_buffer(self):
        """Read-only uploaded camera resource for specialized GPU services."""
        return self._camera_buffer

    def rasterize(self, camera: Camera,
                  contextual: IterationContextualVariables | None = None) -> None:
        if contextual is None:
            contextual = IterationContextualVariables(
                hardness_value=self._extinction_hardness,
                sh_degree=self._active_sh_degree,
            )
        self._run_forward(camera, contextual)

    def radix_sort_pairs(self) -> tuple[spy.NDBuffer, spy.NDBuffer, spy.NDBuffer]:
        """Record the shared 64-bit-key radix sort service."""
        return self._radix_sort_pairs()

    @contextlib.contextmanager
    def _recording(self, label: str):
        """Record GPU work into a fresh command batch installed as the active
        batch, then submit it once on success.

        The active-batch install is cleared in a ``finally`` so a mid-recording
        exception can't leave the pipeline wedged with a stale batch; the submit
        (and the resulting ``last_submit_id``) happens only if the body completed
        without raising. Used by ``train_step`` and the densify passes, which
        record via ``_dispatch`` / ``_marker_group`` into the installed batch.
        """
        batch = self.runtime.command_batch(label)
        transaction = _RecordingTransaction(batch)
        self._batch = batch
        try:
            yield transaction
            self.last_submit_id = self.runtime.submit(batch).submit_id
            transaction.commit()
        finally:
            self._batch = None

    def _load_snapshot(
        self,
        means: np.ndarray,
        log_scales: np.ndarray,
        rot_quats: np.ndarray,
        opacities: np.ndarray,
        color_logits: np.ndarray,
        features_rest: np.ndarray | None = None,
    ) -> None:
        """Upload one primitive snapshot and reset the state it invalidates
        (Adam moments; SH coefficients + their moments). Internal: the pipeline
        is primed once at construction via ``initial_state``. A fresh snapshot
        wholesale-replaces the parameters, so the stale moments MUST reset too."""
        buffer_io.write_gaussians(
            self.buffers,
            means=means, log_scales=log_scales, rot_quats=rot_quats,
            opacities=opacities, color_logits=color_logits,
        )
        # Zero the Adam moments device-side (one clear kernel over the AdamState
        # buffer). Replaces a full GPU->CPU readback that existed only to size a
        # host zero-array — wasteful (>1 GB at a large --max-splats).
        self._adam.clear_state()
        self._adam.reset()

        # Optional SH: zero the rest coefficients (INRIA features_rest init) and
        # their Adam moments device-side, then upload any provided coefficients
        # (e.g. from an SH .ply). ``features_rest`` is ignored on the SH-off path.
        if self.buffers.sh_enabled:
            self._clear_sh_state()
            self.set_features_rest(features_rest)

    def _clear_sh_state(self) -> None:
        """Zero features_rest + its Adam moments (device-side; buffers are far
        too large to host-fill at a big --max-splats). All fp16-packed uint32."""
        sh = self.buffers.sh
        feat_count = self.max_splats * sh.feat_uints
        # adam_rest is float32 (2 uints/coeff: m, v) -> 2*rest_floats uints.
        adam_count = self.max_splats * sh.rest_floats * 2
        self._dispatch(self.k_clear_uint_buffer, "clear_features_rest",
            thread_count=[feat_count, 1, 1],
            buffer=sh.features_rest.storage, count=feat_count)
        self._dispatch(self.k_clear_uint_buffer, "clear_adam_rest",
            thread_count=[adam_count, 1, 1],
            buffer=sh.adam_rest.storage, count=adam_count)

    def set_gaussians(self, means: np.ndarray, log_scales: np.ndarray,
                      rot_quats: np.ndarray, opacities: np.ndarray,
                      color_logits: np.ndarray) -> None:
        """Upload per-field gaussian parameters without touching optimizer state.

        The render-only sibling of ``_load_snapshot``: same fields, but no Adam
        clear and no SH reset, so it is valid on a ``viewer_only`` pipeline (which
        has no Adam state at all). Used to show a COLMAP point cloud in the viewer
        before any training pipeline exists.
        """
        buffer_io.write_gaussians(
            self.buffers,
            means=means, log_scales=log_scales, rot_quats=rot_quats,
            opacities=opacities, color_logits=color_logits,
        )

    def set_gaussians_raw(self, floats: np.ndarray, count: int) -> None:
        """Upload a snapshot already in the packed 14-float layout (from the
        splat stream). Render-only path — does not touch Adam state."""
        buffer_io.write_gaussians_raw(self.buffers, floats, count)

    def set_features_rest(self, features_rest: np.ndarray | None) -> None:
        """Upload initial SH rest coefficients ``(N, rest_floats)`` (coeff-major
        ``[coeff*3 + channel]``) into the separate features_rest buffer, e.g.
        when initializing from an SH ``.ply``. Runs as part of the snapshot load
        (which zero-clears it first). No-op when SH is off or none given."""
        if (not self.buffers.sh_enabled or features_rest is None
                or features_rest.size == 0):
            return
        rest_floats = self.buffers.sh.rest_floats
        assert features_rest.shape[1] == rest_floats, (
            f"features_rest width {features_rest.shape[1]} != rest_floats {rest_floats}")
        # Partial upload: writes the first N*feat_uints uints; the tail stays
        # zero from _clear_sh_state (no full max_splats-sized host array).
        self.buffers.features_rest.copy_from_numpy(
            buffer_io.pack_features_rest(
                np.ascontiguousarray(features_rest, dtype=np.float32)))

    def set_target_image(self, rgb: np.ndarray) -> None:
        """Upload a target image into the pipeline's owned texture. Call
        this once per iter if you don't want to manage per-camera target
        textures yourself.

        Target is stored rgba8_unorm (the GT is an 8-bit sRGB PNG); shaders
        read the same [0,1] sRGB values (unorm auto-normalizes) and compare in
        sRGB space directly. Padded to RGBA with alpha=1.

        For training loops, prefer :meth:`create_target_texture` to
        pre-upload every target once and pass the texture reference
        into :meth:`set_target_texture` each iter — avoids a re-upload
        per step.
        """
        assert rgb.shape == (self.height, self.width, 3), (
            f"target shape {rgb.shape} != ({self.height}, {self.width}, 3)"
        )
        rgba = np.concatenate(
            [rgb, np.ones((self.height, self.width, 1), rgb.dtype)],
            axis=-1,
        )
        self.buffers.target_image.copy_from_numpy(_srgb01_to_u8(rgba))
        self._active_target = self.buffers.target_image

    def pack_target_texture(self, img: np.ndarray) -> np.ndarray:
        """CPU half of :meth:`create_target_texture`: validate, pad alpha, and
        quantize to the contiguous uint8 the upload wants.

        Split out from the upload because it is pure numpy — and therefore drops
        the GIL — so a caller with a few hundred targets can pack them on a
        thread pool (see :func:`trainer.data.parallel.map_ordered`) while the
        device, which is NOT thread-safe, keeps uploading one at a time.
        """
        if img.ndim == 3 and img.shape[2] == 4:
            assert img.shape[:2] == (self.height, self.width)
            rgba = img
            # Straight sRGB RGB + raw alpha — no premultiply (see docstring).
        else:
            assert img.shape == (self.height, self.width, 3), (
                f"target shape {img.shape} != ({self.height}, {self.width}, 3)"
            )
            rgba = np.concatenate(
                [img, np.ones((self.height, self.width, 1), img.dtype)],
                axis=-1,
            )
        return _srgb01_to_u8(rgba)

    def upload_target_texture(self, packed: np.ndarray) -> spy.Texture:
        """GPU half of :meth:`create_target_texture`: allocate the texture and
        upload an already-packed (H, W, 4) uint8 image into it. Must run on the
        thread that owns the device."""
        tex = self.device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba8_unorm,
            width=self.width, height=self.height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="target_image_cached",
        )
        tex.copy_from_numpy(packed)
        return tex

    def create_target_texture(self, img: np.ndarray) -> spy.Texture:
        """Create a new rgba8_unorm target texture and upload into it.

        Accepts (H, W, 3) RGB or (H, W, 4) RGBA. RGB is padded with
        alpha=1. Stored rgba8_unorm: the GT originates as an 8-bit PNG
        (uint8/255), so 8-bit storage is lossless vs. the former rgba32_float
        (4x smaller) — the only change is resized/interpolated pixels quantize
        back to 1/255, which is the source's own precision.

        """
        return self.upload_target_texture(self.pack_target_texture(img))

    def set_target_texture(self, tex: spy.Texture) -> None:
        """Use a pre-uploaded target texture for subsequent forward+loss
        calls. Cheap: just a reference swap — no copy, no upload."""
        self._active_target = tex

    # ------------------------------------------------------------------
    # Camera + forward pipeline
    # ------------------------------------------------------------------

    def _upload_camera(self, camera: Camera) -> None:
        if camera.image_size != (self.width, self.height):
            raise ValueError(
                f"camera resolution {camera.image_size} does not match pipeline "
                f"resolution {(self.width, self.height)}"
            )
        data = np.empty(36, dtype=np.float32)
        data[0:16] = camera.view_matrix.flatten()
        data[16:32] = camera.proj_matrix.flatten()
        data[32:34] = np.array([self.width, self.height], dtype=np.float32)
        data[34:36] = 0.0  # padding
        self._camera_buffer.copy_from_numpy(np.ascontiguousarray(data.view(np.uint8)))

    def _radix_sort_pairs(self) -> tuple[spy.NDBuffer, spy.NDBuffer, spy.NDBuffer]:
        """DeviceRadixSort (ported from GPUSorting, MIT). Returns the
        (keys_lo, keys_hi, values) triplet holding the sorted result.

        Per-pass structure (see shaders/sort/device_radix_sort.slang):
          upsweep    — per-partition digit histogram + atomic add to
                       b_global_hist for this pass.
          scan       — one workgroup per digit, exclusive scan across
                       active partitions in b_pass_hist.
          downsweep  — wave-multisplit rank + scatter of the 3-word
                       record to the *_out buffers.

        ``n`` (the actual pair count) lives in ``scan_total[0]`` on the
        GPU; Python never reads it back in the hot path.
        """
        b = self.buffers
        src_lo, src_hi, src_v = b.sort_keys_lo_a, b.sort_keys_hi_a, b.sort_values_a
        dst_lo, dst_hi, dst_v = b.sort_keys_lo_b, b.sort_keys_hi_b, b.sort_values_b
        max_blocks = buffers._ceil_div(self.max_sort_pairs, RADIX_PART_SIZE)

        # Prepare indirect-dispatch args from the GPU-resident count
        # scan_total[0]. Upsweep and downsweep are then launched with
        # active_blocks groups instead of max_blocks — dispatch cost scales
        # with the frame's actual pair count, not the max_sort_pairs cap.
        self._dispatch(self.k_radix_prepare_dispatch, "radix_prepare_dispatch",
            thread_count=[1, 1, 1],
            n_buf=b.scan_total.storage,
            dispatch_args=b.radix_dispatch_args.storage,
            max_blocks=max_blocks,
            # Clamp scan_total[0] to buffer capacity in place, so the sort and
            # tile-range stages never read past what bin_emit wrote (which is
            # itself capped at max_sort_pairs). See prepareDispatchArgs.
            max_sort_pairs=self.max_sort_pairs,
        )

        # Clear b_global_hist once at the start of the whole sort;
        # subsequent atomic adds during upsweep accumulate into disjoint
        # per-pass regions (offset = pass_idx * 256).
        self._dispatch(self.k_radix_init_global_hist, "radix_init_global_hist",
            thread_count=[RADIX_GLOBAL_HIST_SIZE, 1, 1],
            b_global_hist=b.radix_global_hist.storage,
        )

        for pass_idx in range(8):
            radix_shift = (pass_idx % 4) * 8
            key_word = 0 if pass_idx < 4 else 1
            with self._marker_group(f"pass_{pass_idx}"):
                self._dispatch_indirect(self.k_radix_upsweep, "radix_upsweep",
                    b.radix_dispatch_args,
                    keys_lo=src_lo.storage, keys_hi=src_hi.storage,
                    b_pass_hist=b.radix_pass_hist.storage,
                    b_global_hist=b.radix_global_hist.storage,
                    n_buf=b.scan_total.storage,
                    radix_shift=radix_shift, key_word=key_word,
                    pass_idx=pass_idx, max_blocks=max_blocks,
                )
                self._dispatch(self.k_radix_scan, "radix_scan",
                    thread_count=[256 * _RADIX_SCAN_DIM, 1, 1],
                    b_pass_hist=b.radix_pass_hist.storage,
                    n_buf=b.scan_total.storage,
                    max_blocks=max_blocks,
                )
                self._dispatch_indirect(self.k_radix_downsweep, "radix_downsweep",
                    b.radix_dispatch_args,
                    keys_lo_in=src_lo.storage, keys_hi_in=src_hi.storage, values_in=src_v.storage,
                    keys_lo_out=dst_lo.storage, keys_hi_out=dst_hi.storage, values_out=dst_v.storage,
                    b_pass_hist=b.radix_pass_hist.storage,
                    b_global_hist=b.radix_global_hist.storage,
                    n_buf=b.scan_total.storage,
                    radix_shift=radix_shift, key_word=key_word,
                    pass_idx=pass_idx, max_blocks=max_blocks,
                )
            src_lo, dst_lo = dst_lo, src_lo
            src_hi, dst_hi = dst_hi, src_hi
            src_v, dst_v = dst_v, src_v
        return src_lo, src_hi, src_v

    def _run_forward(self, camera: Camera, contextual: IterationContextualVariables) -> None:
        # TODO: we should use a const buffer but cest la vie
        self._upload_camera(camera)
        hardness_value = contextual.hardness_value
        buffers = self.buffers
        num_tiles = buffers.num_tiles

        # Clear output textures + tile ranges.
        with self._marker_group("clear"):
            self._dispatch(self.k_clear_textures, "clear_textures",
                thread_count=[self.width, self.height, 1],
                output_image=buffers.output_image,
                gaussian_count_per_pixel=buffers.gaussian_count_per_pixel,
                frame_dim=spy.uint2(self.width, self.height),
            )
            self._dispatch(self.k_clear_tile_ranges, "clear_tile_ranges",
                thread_count=[num_tiles, 1, 1],
                tile_start=buffers.tile_start.storage,
                tile_end=buffers.tile_end.storage,
                num_tiles=num_tiles,
            )

        #TODO: This should ideally also be gpu driven by an indirect dispatch
        if self.raster_config.cluster_cull:
            num_clusters = (self.max_splats + CLUSTER_SIZE - 1) // CLUSTER_SIZE
            with self._marker_group("cluster_cull"):
                self._dispatch(self.k_build_cluster_aabb, "build_cluster_aabb",
                    thread_count=[num_clusters * CLUSTER_SIZE, 1, 1],
                    gaussians=buffers.gaussians.storage,
                    num_gaussians_counter=buffers.num_gaussians_counter.storage,
                    cluster_aabb_min=buffers.cluster_aabb_min.storage,
                    cluster_aabb_max=buffers.cluster_aabb_max.storage,
                    max_splats=self.max_splats,
                    num_clusters=num_clusters,
                )
                self._dispatch(self.k_cull_clusters_frustum, "cull_clusters_frustum",
                    thread_count=[num_clusters, 1, 1],
                    cluster_aabb_min=buffers.cluster_aabb_min.storage,
                    cluster_aabb_max=buffers.cluster_aabb_max.storage,
                    camera=self._camera_buffer,
                    cluster_visible=buffers.cluster_visible.storage,
                    num_clusters=num_clusters,
                )

       
        with self._marker_group("bin_count"):
            self._dispatch(self.k_bin_count, "bin_count",
                thread_count=[self.max_splats, 1, 1],
                gaussians=buffers.gaussians.storage,
                num_gaussians_counter=buffers.num_gaussians_counter.storage,
                camera=self._camera_buffer,
                projected_2d=buffers.projected_2d.storage,
                blend_data=buffers.blend_data.storage,
                tile_count_per_gaussian=buffers.tile_count_per_gaussian.storage,
                cluster_visible=buffers.cluster_visible.storage,
                filter_3D=buffers.filter_3D.storage,
                mip_enabled=1 if self.raster_config.mip_enabled else 0,
                kernel_size=self.raster_config.kernel_size,
                z_near=self.raster_config.z_near_cull,
                max_splats=self.max_splats,
            )

        if buffers.sh_enabled:
            with self._marker_group("sh_forward"):
                self._dispatch(self.k_sh_forward, "sh_forward",
                    thread_count=[self.max_splats, 1, 1],
                    gaussians=buffers.gaussians.storage,
                    num_gaussians_counter=buffers.num_gaussians_counter.storage,
                    camera=self._camera_buffer,
                    features_rest=buffers.features_rest.storage,
                    blend_data=buffers.blend_data.storage,
                    active_sh_degree=contextual.sh_degree,
                    rest_floats=buffers.sh.rest_floats,
                    max_splats=self.max_splats,
                )

        with self._marker_group("scan_tile_offsets"):
            self._scan(
                n=self.max_splats,
                in_buf=buffers.tile_count_per_gaussian,
                out_buf=buffers.tile_offset_per_gaussian,
            )

        with self._marker_group("bin_emit"):
            self._dispatch(self.k_bin_emit, "bin_emit",
                thread_count=[self.max_splats, 1, 1],
                gaussians=buffers.gaussians.storage,
                num_gaussians_counter=buffers.num_gaussians_counter.storage,
                camera=self._camera_buffer,
                projected_2d=buffers.projected_2d.storage,
                tile_offset_per_gaussian=buffers.tile_offset_per_gaussian.storage,
                keys_lo=buffers.sort_keys_lo_a.storage,
                keys_hi=buffers.sort_keys_hi_a.storage,
                values=buffers.sort_values_a.storage,
                cluster_visible=buffers.cluster_visible.storage,
                z_near=self.raster_config.z_near_cull,
                max_sort_pairs=self.max_sort_pairs,
            )

        with self._marker_group("radix_sort"):
            sorted_lo, sorted_hi, sorted_v = self._radix_sort_pairs()

        with self._marker_group("identify_tile_ranges"):
            self._dispatch(self.k_identify_tile_ranges, "identify_tile_ranges",
                thread_count=[self.max_sort_pairs, 1, 1],
                keys_hi=sorted_hi.storage,
                tile_start=buffers.tile_start.storage,
                tile_end=buffers.tile_end.storage,
                n_buf=buffers.scan_total.storage,
                num_tiles=num_tiles,
            )


        self._sorted_values = sorted_v

        with self._marker_group("bucket_scan"):
            self._dispatch(self.k_bucket_count, "bucket_count",
                thread_count=[num_tiles, 1, 1],
                tile_start=buffers.tile_start.storage,
                tile_end=buffers.tile_end.storage,
                bucket_count=buffers.bucket_count.storage,
                num_tiles=num_tiles,
            )
            self._scan(
                n=num_tiles,
                in_buf=buffers.bucket_count,
                out_buf=buffers.bucket_offsets,
                total_buf=buffers.bucket_offset_total,
            )

        with self._marker_group("rasterize"):
            self._dispatch(self.k_forward, "forward",
                # One thread per tile-pixel: [numthreads(TILE_WIDTH,TILE_HEIGHT)].
                thread_count=[buffers.num_tiles_x * TILE_WIDTH,
                              buffers.num_tiles_y * TILE_HEIGHT, 1],
                blend_data=buffers.blend_data.storage,
                tile_start=buffers.tile_start.storage,
                tile_end=buffers.tile_end.storage,
                sorted_gaussian_ids=sorted_v.storage,
                camera=self._camera_buffer,
                bucket_offsets=buffers.bucket_offsets.storage,
                raster_checkpoints=buffers.raster_checkpoints.storage,
                output_image=buffers.output_image,
                gaussian_count_per_pixel=buffers.gaussian_count_per_pixel,
                extinction_hardness=hardness_value,
            )


    def render(self, camera: Camera) -> None:
        with self.recording("render"):
            self.rasterize(camera)
        self.runtime.wait_for_idle()

    def read_output_image(self) -> np.ndarray:
        with self.gpu_lock:
            return self.buffers.output_image.to_numpy()

    def read_bin_debug(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        b = self.buffers
        with self.gpu_lock:
            starts = b.tile_start.to_numpy().view(np.uint32)
            ends = b.tile_end.to_numpy().view(np.uint32)
            overlap = b.gaussian_count_per_pixel.to_numpy().view(np.uint32)
        n = b.num_tiles_x * b.num_tiles_y
        tile_counts = (ends[:n].astype(np.int64) - starts[:n].astype(np.int64))
        tile_counts = np.clip(tile_counts, 0, None).astype(np.uint32)
        overlap = overlap.reshape(self.height, self.width)
        return tile_counts, overlap, b.num_tiles_x, b.num_tiles_y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _clear_grads(self) -> None:
        """Zero the 14 scalar param grads + 5 2D grad seeds. Full clear.

        Used by tests that need a clean grad state before an isolated
        backward() (see tests/test_gradients.py, tests/test_backward_persplat.py).
        Also used at pipeline setup via the training loop before iter 1
        (see :meth:`_clear_2d_grad_seeds` for the per-iter hot path)."""
        b = self.buffers
        for name in GRAD_SCALAR_COMPONENTS:
            self._dispatch(self.k_grad_clear, "grad_clear",
                thread_count=[self.max_splats, 1, 1],
                grad_buffer=b.grad_buffers[name].storage,
                max_splats=self.max_splats,
            )
        for name in b.grad_2d_seed_buffers:
            self._dispatch(self.k_grad_clear, "grad_clear",
                thread_count=[self.max_splats, 1, 1],
                grad_buffer=b.grad_2d_seed_buffers[name].storage,
                max_splats=self.max_splats,
            )

    def _clear_2d_grad_seeds(self) -> None:
        b = self.buffers
        for name in b.grad_2d_seed_buffers:
            self._dispatch(self.k_grad_clear, "grad_clear",
                thread_count=[self.max_splats, 1, 1],
                grad_buffer=b.grad_2d_seed_buffers[name].storage,
                max_splats=self.max_splats,
            )

    def _run_loss(self, contextual: IterationContextualVariables) -> None:
        self.image_loss.record(
            self._batch,
            rendered=self.buffers.output_image,
            target=self._active_target,
            stbn_kwargs=self._stbn_kwargs(contextual),
        )

    def _run_backward(self, iterationContext) -> None:
        iteration_contextual_variables: IterationContextualVariables = iterationContext.iteration_contextual_variables
        b = self.buffers
        gb = b.grad_buffers
        sorted_values = self._sorted_values
        # The mask parameters the loss already applied. tileMaxProcessed always
        # takes them (runtime-gated), the backward only on the compacted build
        # (they are declared under #if STBN_ENABLED there).
        stbn_kwargs = self._stbn_kwargs(iteration_contextual_variables)
        backward_stbn = stbn_kwargs if self.stbn_config.backward_compacted else {}

        with self._marker_group("rasterize_backward"):
            if self.use_persplat_backward:
                num_tiles = b.num_tiles_x * b.num_tiles_y
                if self.use_flat_bucket_backward:
                    with self._marker_group("live_bucket_compaction"):
                        self._dispatch(self.k_tile_max_processed, "tile_max_processed",
                            # TILE_AREA threads/tile (reduces over the tile's pixels).
                            thread_count=[num_tiles * TILE_AREA, 1, 1],
                            gaussian_count_per_pixel=b.gaussian_count_per_pixel,
                            live_bucket_count=b.live_bucket_count.storage,
                            frame_dim_px=spy.uint2(self.width, self.height),
                            num_tiles_x=b.num_tiles_x,
                            num_tiles=num_tiles,
                            **stbn_kwargs,
                        )
                        self._scan(
                            n=num_tiles,
                            in_buf=b.live_bucket_count,
                            out_buf=b.live_bucket_offsets,
                            total_buf=b.live_bucket_total,
                        )
                        self._dispatch(self.k_scatter_live_buckets, "scatter_live_buckets",
                            thread_count=[num_tiles, 1, 1],
                            live_bucket_count=b.live_bucket_count.storage,
                            live_bucket_offsets=b.live_bucket_offsets.storage,
                            live_buckets=b.live_buckets.storage,
                            num_tiles=num_tiles,
                        )
                        max_buckets = (buffers._ceil_div(self.max_sort_pairs, buffers.RASTER_BUCKET_SIZE) + num_tiles)
                        self._dispatch(self.k_bucket_prepare_dispatch, "bucket_prepare_dispatch",
                            thread_count=[1, 1, 1],
                            live_bucket_total=b.live_bucket_total.storage,
                            dispatch_args=b.bucket_dispatch_args.storage,
                            max_buckets=max_buckets,
                        )
                    self._dispatch_indirect(self.k_backward_persbucket, "backward_persbucket",
                        args_buf=b.bucket_dispatch_args,
                        blend_data=b.blend_data.storage,
                        tile_start=b.tile_start.storage,
                        tile_end=b.tile_end.storage,
                        sorted_gaussian_ids=self._sorted_values.storage,
                        bucket_offsets=b.bucket_offsets.storage,
                        live_buckets=b.live_buckets.storage,
                        live_bucket_total=b.live_bucket_total.storage,
                        raster_checkpoints=b.raster_checkpoints.storage,
                        camera=self._camera_buffer,
                        output_image=b.output_image,
                        gaussian_count_per_pixel=b.gaussian_count_per_pixel,
                        dL_dC_texture=b.dL_dC_texture,
                        **b.grad_seed_kwargs(),
                        grad_opacity=gb["opacity"].storage,
                        grad_color_r=gb["color_r"].storage,
                        grad_color_g=gb["color_g"].storage,
                        grad_color_b=gb["color_b"].storage,
                        grad_2d_x_iter=b.grad_2d_x_iter.storage,
                        grad_2d_y_iter=b.grad_2d_y_iter.storage,
                        pixel_phase=self._backward_pixel_phase,
                        **backward_stbn,
                        extinction_hardness=iteration_contextual_variables.hardness_value,
                    )
                else:
                  # Transposed per-splat backward: one workgroup (TILE_AREA
                  # threads) per tile, splat-parallel with checkpoint reconstruction.
                  self._dispatch(self.k_backward_persplat, "backward_persplat",
                    thread_count=[num_tiles * TILE_AREA, 1, 1],
                    blend_data=b.blend_data.storage,
                    tile_start=b.tile_start.storage,
                    tile_end=b.tile_end.storage,
                    sorted_gaussian_ids=sorted_values.storage,
                    bucket_offsets=b.bucket_offsets.storage,
                    raster_checkpoints=b.raster_checkpoints.storage,
                    camera=self._camera_buffer,
                    output_image=b.output_image,
                    gaussian_count_per_pixel=b.gaussian_count_per_pixel,
                    dL_dC_texture=b.dL_dC_texture,
                    **b.grad_seed_kwargs(),
                    grad_opacity=gb["opacity"].storage,
                    grad_color_r=gb["color_r"].storage,
                    grad_color_g=gb["color_g"].storage,
                    grad_color_b=gb["color_b"].storage,
                    grad_2d_x_iter=b.grad_2d_x_iter.storage,
                    grad_2d_y_iter=b.grad_2d_y_iter.storage,
                    # SGD pixel-subsampling phase; rotates each backward so the
                    # visited subset alternates (no-op at BACKWARD_PIXEL_STRIDE=1).
                    pixel_phase=self._backward_pixel_phase,
                    **backward_stbn,
                    extinction_hardness=iteration_contextual_variables.hardness_value,
                )

                self._backward_pixel_phase += 1

                with self._marker_group("projection_backward"):
                    self._dispatch(self.k_projection_backward, "projection_backward",
                        thread_count=[self.max_splats, 1, 1],
                        gaussians=b.gaussians.storage,
                        num_gaussians_counter=b.num_gaussians_counter.storage,
                        camera=self._camera_buffer,
                        **b.grad_seed_kwargs(),
                        grad_mean_x=gb["mean_x"].storage,
                        grad_mean_y=gb["mean_y"].storage,
                        grad_mean_z=gb["mean_z"].storage,
                        grad_log_scale_x=gb["log_scale_x"].storage,
                        grad_log_scale_y=gb["log_scale_y"].storage,
                        grad_log_scale_z=gb["log_scale_z"].storage,
                        grad_rot_w=gb["rot_w"].storage,
                        grad_rot_x=gb["rot_x"].storage,
                        grad_rot_y=gb["rot_y"].storage,
                        grad_rot_z=gb["rot_z"].storage,
                        filter_3D=b.filter_3D.storage,
                        mip_enabled=1 if self.raster_config.mip_enabled else 0,
                        kernel_size=self.raster_config.kernel_size,
                        cluster_visible=b.cluster_visible.storage,
                        max_splats=self.max_splats,
                    )
            else:
                self._dispatch(self.k_backward, "backward",
                    # One thread per tile-pixel: [numthreads(TILE_WIDTH,TILE_HEIGHT)].
                    thread_count=[b.num_tiles_x * TILE_WIDTH,
                                  b.num_tiles_y * TILE_HEIGHT, 1],
                    gaussians=b.gaussians.storage,
                    projected_2d=b.projected_2d.storage,
                    tile_start=b.tile_start.storage,
                    tile_end=b.tile_end.storage,
                    sorted_gaussian_ids=sorted_values.storage,
                    camera=self._camera_buffer,
                    output_image=b.output_image,
                    gaussian_count_per_pixel=b.gaussian_count_per_pixel,
                    dL_dC_texture=b.dL_dC_texture,
                    **b.grad_kwargs(),
                    grad_2d_x_iter=b.grad_2d_x_iter.storage,
                    grad_2d_y_iter=b.grad_2d_y_iter.storage,
                )

        if b.sh_enabled:
            with self._marker_group("sh_backward"):
                self._dispatch(self.k_sh_backward, "sh_backward",
                    thread_count=[self.max_splats, 1, 1],
                    gaussians=b.gaussians.storage,
                    num_gaussians_counter=b.num_gaussians_counter.storage,
                    camera=self._camera_buffer,
                    features_rest=b.features_rest.storage,
                    features_rest_grad=b.features_rest_grad.storage,
                    grad_color_r=gb["color_r"].storage,
                    grad_color_g=gb["color_g"].storage,
                    grad_color_b=gb["color_b"].storage,
                    grad_mean_x=gb["mean_x"].storage,
                    grad_mean_y=gb["mean_y"].storage,
                    grad_mean_z=gb["mean_z"].storage,
                    cluster_visible=b.cluster_visible.storage,
                    active_sh_degree=iteration_contextual_variables.sh_degree,
                    rest_floats=b.sh.rest_floats,
                    max_splats=self.max_splats,
                )

        densification_variables = iteration_contextual_variables.densification_variables
        if (densification_variables is not None
                and densification_variables.structure_tensor_texture is not None):
            with self._marker_group("structure_accum"):
                _densify.accumulate_structure_scores(self,
                    densification_variables.structure_tensor_texture,
                    **(densification_variables.structure_kwargs or {})
                )

        with self._marker_group("accumulate_iter"):
            self._dispatch(self.k_accumulate_iter, "accumulate_iter",
                thread_count=[self.max_splats, 1, 1],
                grad_2d_x_iter=b.grad_2d_x_iter.storage,
                grad_2d_y_iter=b.grad_2d_y_iter.storage,
                grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
                visibility_count_accum=b.visibility_count_accum.storage,
                frame_dim=spy.float2(self.width, self.height),
                max_splats=self.max_splats,
            )

    def _run_regularization(self) -> None:
        config = self.regularization_config
        if config.opacity_weight <= 0 and config.scale_weight <= 0:
            return
        b = self.buffers
        gb = b.grad_buffers
        self._dispatch(self.k_regularization, "regularization",
            thread_count=[self.max_splats, 1, 1],
            gaussians=b.gaussians.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            grad_opacity=gb["opacity"].storage,
            grad_log_scale_x=gb["log_scale_x"].storage,
            grad_log_scale_y=gb["log_scale_y"].storage,
            grad_log_scale_z=gb["log_scale_z"].storage,
            cluster_visible=b.cluster_visible.storage,
            w_opacity=config.opacity_weight,
            w_scale=config.scale_weight,
        )

    def clear_tracking(self) -> None:
        """Zero the v2 densification tracking accumulators. Called at the
        start of each densification window (every N iters)."""
        b = self.buffers
        self._dispatch(self.k_clear_tracking, "clear_tracking",
            thread_count=[self.max_splats, 1, 1],
            grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
            visibility_count_accum=b.visibility_count_accum.storage,
            grad_2d_x_iter=b.grad_2d_x_iter.storage,
            grad_2d_y_iter=b.grad_2d_y_iter.storage,
            max_splats=self.max_splats,
        )

    def read_tracking(self) -> dict[str, np.ndarray]:
        """Read window-level tracking buffers (int64 fixed-point)."""
        b = self.buffers
        raw_mag = b.grad_mean_2d_mag_accum.to_numpy()
        raw_vis = b.visibility_count_accum.to_numpy()
        mag_int = np.frombuffer(raw_mag.tobytes(), dtype=np.int64)
        vis = np.frombuffer(raw_vis.tobytes(), dtype=np.uint32)
        return {
            "grad_mean_2d_mag": mag_int.astype(np.float64) / 65536.0,
            "visibility_count": vis.astype(np.int64),
        }


    def train_step(self, iterationContext: IterationContext) -> None:

        if iterationContext.learning_rates is None:
            raise RuntimeError("train_step requires learning_rates to be set in iterationContext")

        contextual = iterationContext.iteration_contextual_variables
        # Apply this iteration's scheduled values to the pipeline's render state so renders outside the step (eval/export/viewer/bench) inherit them.
        # This is code smell, that we need to have the pipeline state be set from the iterationContext
        # but at the same time we may need to run run_forward from outside the train step, and the train step depends on the iterationContext
        # i may need to think more about how to fix ts
        self.set_extinction_hardness(contextual.hardness_value)
        self.set_active_sh_degree(contextual.sh_degree)

        with self._recording("train_step"):
            with self._marker_group("clear_grads"):
                # TODO this should not be here by itself
                self._clear_2d_grad_seeds()
            with self._marker_group("forward"):
                self._run_forward(iterationContext.camera_entry.camera, contextual)
            with self._marker_group("loss"):
                self._run_loss(contextual)
            with self._marker_group("backward"):
                self._run_backward(iterationContext)
            with self._marker_group("regularization"):
                self._run_regularization()
            with self._marker_group("adam"):
                self._adam.step(iterationContext)

    def densify_simple(self, cfg, grad_thresh=0.0002, scale_thresh=None,
                       prune_big_scale=None, allow_scale_prune=True,
                       carry_eta=False) -> dict:
        if self.simple_densifier is None:
            raise RuntimeError("densification is unavailable in viewer-only mode")
        return self.simple_densifier.run(
            cfg, grad_thresh, scale_thresh, prune_big_scale,
            allow_scale_prune, carry_eta)

    def set_mipsplatting_cameras(self, train_cameras: list[Camera]) -> None:
        """
        Mip-Splatting setup: upload the (static) training-camera array and
        derive the sqrt(0.2)/max_focal coefficient ONCE. Must be called before
        :meth:`compute_3d_filter`.
        """
        n = len(train_cameras)

        # CameraData packs to 36 float32s: view 4x4 (16) + proj 4x4 (16) +
        # image_size (2) + 2 pad-to-16B-alignment. Buffer size derives from the
        # array itself (data.nbytes) — no hardcoded byte count.
        data = np.zeros((n, 36), dtype=np.float32)
        max_focal = 0.0
        for i, cam in enumerate(train_cameras):
            data[i, 0:16] = cam.view_matrix.flatten()
            data[i, 16:32] = cam.proj_matrix.flatten()
            data[i, 32:34] = np.array(cam.image_size, dtype=np.float32)
            # focal_x in pixels: proj[0][0] = 2*fx/W  ->  fx = proj[0][0]*W/2.
            fx = abs(float(cam.proj_matrix[0, 0])) * float(cam.image_size[0]) * 0.5
            max_focal = max(max_focal, fx)
        buf = self.device.create_buffer(
            size=data.nbytes,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label="mip_cameras",
        )
        buf.copy_from_numpy(np.ascontiguousarray(data.reshape(-1).view(np.uint8)))
        self._mip_cameras_buffer = buf
        self._mip_num_cameras = n
        self._mip_filter_coef = float(np.sqrt(0.2) / max(max_focal, 1e-8))

    def compute_3d_filter(self) -> None:
        """Mip-Splatting: recompute the per-gaussian 3D smoothing-filter size
        (buffers.filter_3D) from the cameras uploaded by :meth:`set_mipsplatting_cameras`.
        Run under no_grad — filter_3D is a geometric constant, never a gradient
        target. Call once at init and after every densify+morton so new children
        get a valid filter.

        Only the per-gaussian dispatch runs here (means change every step); the
        static camera array + filter coefficient were built once by
        set_mipsplatting_cameras.
        Heavy (O(n_gaussians * n_cameras)) but only fires on the ~17 densify
        ticks of a run, off the hot path.
        """
        if getattr(self, "_mip_cameras_buffer", None) is None:
            raise RuntimeError(
                "compute_3d_filter() requires set_mipsplatting_cameras(...) to be called first")
        b = self.buffers
        # One threadgroup of 32 threads per gaussian (must match
        # THREADS_PER_SPLAT in shaders/mip/compute_3d_filter.slang). slangpy
        # divides the total thread_count by the shader's numthreads, so
        # max_splats*32 total threads => max_splats groups (one per gaussian).
        MIP_THREADS_PER_SPLAT = 32
        self._dispatch(self.k_compute_3d_filter, "compute_3d_filter",
            thread_count=[self.max_splats * MIP_THREADS_PER_SPLAT, 1, 1],
            gaussians=b.gaussians.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            cameras=self._mip_cameras_buffer,
            filter_3D=b.filter_3D.storage,
            num_cameras=self._mip_num_cameras,
            filter_coef=self._mip_filter_coef,
            max_splats=self.max_splats,
        )
        # No wait_for_idle: filter_3D is a GPU buffer consumed by the next forward
        # (bin_count / project); nothing reads it back to CPU here. Draining the
        # whole queue on every call cost ~650ms in-loop (kernel itself is ~59ms) —
        # a pure host-side sync bubble. Let it flow through the command stream.

    def reset_opacity(self, cfg) -> None:
        return _densify.reset_opacity(self, cfg)

    def morton_sort(self, world_min, world_max, carry_eta=False) -> None:
        return _densify.morton_sort(self, world_min, world_max, carry_eta)

    def compute_structure_scores(self, cameras, structure_textures, cfg) -> None:
        return _densify.compute_structure_scores(self, cameras, structure_textures, cfg)

    def clear_structure_scores(self) -> None:
        return _densify.clear_structure_scores(self)

    def densify_structure_aware(self, cfg) -> dict:
        return _densify.densify_structure_aware(self, cfg)

    def _scan(self, n: int,
              in_buf: spy.NDBuffer | None = None,
              out_buf: spy.NDBuffer | None = None,
              total_buf: spy.NDBuffer | None = None) -> None:
        """Compatibility delegate; new components receive ``scan_service``."""
        if in_buf is None:
            densification = self.buffers._require_densification()
            in_buf = densification.valid_mask
        self.scan_service.scan(n, in_buf, out_buf, total_buf)

