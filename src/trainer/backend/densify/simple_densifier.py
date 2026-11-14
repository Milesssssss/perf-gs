# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Reference 3DGS simple densification as an explicit GPU service.

The service receives narrow capability resources, typed kernels, scan, and a
command-recording interface. It does not receive Pipeline or GaussianBuffers and
therefore cannot access rasterization, loss, viewer, or unrelated resources.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from trainer.backend.memory.buffers import (
    AdamResources,
    DensificationResources,
    EtaResources,
    PrimitiveStorage,
    SphericalHarmonicsResources,
)
from trainer.backend.compute.gpu_scan import GpuScan
from trainer.backend.memory.reorder import reorder_soa_planes


@dataclass(frozen=True)
class SimpleDensifyKernels:
    decide_actions: Any
    snapshot_count: Any
    prepare_dispatch: Any
    split_and_clone: Any
    build_prune_mask: Any
    scatter_compact: Any
    scatter_compact_eta: Any
    split_and_clone_sh: Any = None
    scatter_compact_sh: Any = None
    # Single-output compaction variants for the shared-scratch-arena path.
    scatter_compact_gaussian: Any = None
    scatter_compact_adam: Any = None
    scatter_compact_features: Any = None
    scatter_compact_adam_rest: Any = None
    scatter_compact_eta_only: Any = None


@dataclass(frozen=True)
class CommandInterface:
    dispatch: Callable[..., None]
    dispatch_indirect: Callable[..., None]
    recording: Callable[..., Any]
    marker_group: Callable[..., Any]
    # Records an arena->live byte copy into the active batch (see
    # Pipeline.copy_arena_back / ScratchArena.copy_back).
    copy_arena_back: Callable[..., None] = None


class SimpleDensifier:
    def __init__(
        self,
        *,
        primitive: PrimitiveStorage,
        adam: AdamResources,
        resources: DensificationResources,
        eta: EtaResources | None,
        scan: GpuScan,
        kernels: SimpleDensifyKernels,
        commands: CommandInterface,
        width: int,
        height: int,
        max_primitives: int,
        sh: SphericalHarmonicsResources | None = None,
    ) -> None:
        self.primitive = primitive
        self.adam = adam
        self.resources = resources
        self.eta = eta
        self.scan = scan
        self.kernels = kernels
        self.commands = commands
        self.width = width
        self.height = height
        self.max_primitives = max_primitives
        self.sh = sh

    def run(
        self,
        cfg,
        grad_thresh: float = 0.0002,
        scale_thresh: float | None = None,
        prune_big_scale: float | None = None,
        allow_scale_prune: bool = True,
        carry_eta: bool = False,
        min_blend_weight: float = 0.0,
    ) -> dict:
        """Threshold-based split/clone/prune with a transactional compaction."""
        if carry_eta and self.eta is None:
            raise RuntimeError("carry_eta requires EtaResources")
        if scale_thresh is None:
            scale_thresh = cfg.t_big_scale * 0.1

        channels = 3.0
        grad_thresh_internal = grad_thresh * (self.width * self.height * channels)
        big_scale = prune_big_scale if prune_big_scale is not None else cfg.t_big_scale
        primitive = self.primitive
        adam = self.adam
        resources = self.resources
        kernels = self.kernels
        commands = self.commands

        with commands.recording("densify_simple") as recording, \
                commands.marker_group("densify_simple"):
            commands.dispatch(
                kernels.decide_actions, "decide_actions_simple",
                thread_count=[self.max_primitives, 1, 1],
                gaussians=primitive.values.storage,
                num_gaussians_counter=primitive.active_count.storage,
                grad_mean_2d_mag_accum=resources.grad_mean_2d_mag.storage,
                visibility_count_accum=resources.visibility_count.storage,
                blend_weight_accum=resources.blend_weight.storage,
                action=resources.actions.storage,
                grad_thresh=grad_thresh_internal,
                scale_thresh=scale_thresh,
                t_opacity=cfg.t_opacity,
                t_big_scale=big_scale,
                allow_scale_prune=1 if allow_scale_prune else 0,
                min_blend_weight=float(min_blend_weight),
                max_splats=self.max_primitives,
            )
            commands.dispatch(
                kernels.snapshot_count, "densify_snapshot",
                thread_count=[1, 1, 1],
                src=primitive.active_count.storage,
                dst=resources.pre_count.storage,
            )
            commands.dispatch(
                kernels.prepare_dispatch, "densify_prepare_args",
                thread_count=[1, 1, 1],
                count_buf=resources.pre_count.storage,
                dispatch_args=resources.dispatch_args.storage,
                max_splats=self.max_primitives,
            )
            split_kwargs = dict(
                gaussians=primitive.values.storage,
                adam_state=adam.state.storage,
                num_gaussians_counter=primitive.active_count.storage,
                action=resources.actions.storage,
                pre_count_buf=resources.pre_count.storage,
                max_splats=self.max_primitives,
            )
            if self.sh is not None:
                split_kwargs.update(
                    features_rest=self.sh.features_rest.storage,
                    adam_rest=self.sh.adam_rest.storage,
                    rest_floats=self.sh.rest_floats,
                )
            commands.dispatch_indirect(
                kernels.split_and_clone_sh if self.sh is not None
                else kernels.split_and_clone,
                "split_and_clone", resources.dispatch_args, **split_kwargs)
            commands.dispatch(
                kernels.snapshot_count, "densify_snapshot",
                thread_count=[1, 1, 1],
                src=primitive.active_count.storage,
                dst=resources.post_count.storage,
            )
            commands.dispatch(
                kernels.build_prune_mask, "build_prune_mask",
                thread_count=[self.max_primitives, 1, 1],
                action=resources.actions.storage,
                valid_mask=resources.valid_mask.storage,
                pre_count_buf=resources.pre_count.storage,
                post_count_buf=resources.post_count.storage,
                max_splats=self.max_primitives,
            )
            self.scan.scan(
                self.max_primitives,
                resources.valid_mask,
                total_buf=self.scan.resources.total,
            )

            # Shared-arena compaction: compact each resource into the arena, then
            # copy it back into the live buffer (commits in place; no swap).
            # gaussians + adam always; SH pair when enabled; eta via its
            # dedicated eta-only kernel + scratch swap (kept small, not arena'd).
            arena = resources.scratch_arena
            commands.dispatch(
                kernels.scatter_compact_gaussian, "scatter_compact_gaussian",
                thread_count=[self.max_primitives, 1, 1],
                gaussians_in=primitive.values.storage,
                valid_mask=resources.valid_mask.storage,
                scan=self.scan.resources.work.storage,
                gaussians_out=arena.storage,
                max_splats=self.max_primitives,
            )
            commands.copy_arena_back(arena, primitive.values, primitive.values.storage.size,
                                     count_buf=self.scan.resources.total)
            commands.dispatch(
                kernels.scatter_compact_adam, "scatter_compact_adam",
                thread_count=[self.max_primitives, 1, 1],
                adam_state_in=adam.state.storage,
                valid_mask=resources.valid_mask.storage,
                scan=self.scan.resources.work.storage,
                adam_state_out=arena.storage,
                max_splats=self.max_primitives,
            )
            commands.copy_arena_back(arena, adam.state, adam.state.storage.size,
                                     count_buf=self.scan.resources.total)

            if carry_eta:
                commands.dispatch(
                    kernels.scatter_compact_eta_only, "scatter_compact_eta_only",
                    thread_count=[self.max_primitives, 1, 1],
                    valid_mask=resources.valid_mask.storage,
                    scan=self.scan.resources.work.storage,
                    action=resources.actions.storage,
                    pre_count_buf=resources.pre_count.storage,
                    **self.eta.scatter_kwargs(),
                    max_splats=self.max_primitives,
                )

            # SH: carry features_rest + its Adam moments through the same compaction.
            if self.sh is not None:
                commands.dispatch(
                    kernels.scatter_compact_features, "scatter_compact_features",
                    thread_count=[self.max_primitives, 1, 1],
                    features_rest_in=self.sh.features_rest.storage,
                    valid_mask=resources.valid_mask.storage,
                    scan=self.scan.resources.work.storage,
                    features_rest_out=arena.storage,
                    rest_floats=self.sh.rest_floats,
                    max_splats=self.max_primitives,
                )
                commands.copy_arena_back(arena, self.sh.features_rest, self.sh.features_rest.storage.size,
                                         count_buf=self.scan.resources.total)
                reorder_soa_planes(
                    commands=commands, arena=arena, buffer=self.sh.adam_rest,
                    kernel=kernels.scatter_compact_adam_rest,
                    marker="scatter_compact_adam_rest",
                    plane_count=self.sh.moment_planes, capacity=self.max_primitives,
                    live_count=self.scan.resources.total,
                    valid_mask=resources.valid_mask.storage,
                    scan=self.scan.resources.work.storage,
                )

            commands.dispatch(
                kernels.snapshot_count, "densify_snapshot",
                thread_count=[1, 1, 1],
                src=self.scan.resources.total.storage,
                dst=primitive.active_count.storage,
            )
            # Only eta still uses the pointer-swap commit (its scratch is kept);
            # gaussians / adam / SH already committed in place via copy-back.
            if carry_eta:
                recording.after_submit(lambda: self.eta.swap())

        return {"pre_count": None, "post_split_count": None, "final_count": None}
