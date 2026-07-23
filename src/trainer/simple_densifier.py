"""Reference 3DGS simple densification as an explicit GPU service.

The service receives narrow capability resources, typed kernels, scan, and a
command-recording interface. It does not receive Pipeline or GaussianBuffers and
therefore cannot access rasterization, loss, viewer, or unrelated resources.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from trainer.buffers import (
    AdamResources,
    DensificationResources,
    EtaResources,
    PrimitiveStorage,
)
from trainer.gpu_scan import GpuScan


@dataclass(frozen=True)
class SimpleDensifyKernels:
    decide_actions: Any
    snapshot_count: Any
    prepare_dispatch: Any
    split_and_clone: Any
    build_prune_mask: Any
    scatter_compact: Any
    scatter_compact_eta: Any


@dataclass(frozen=True)
class CommandInterface:
    dispatch: Callable[..., None]
    dispatch_indirect: Callable[..., None]
    recording: Callable[..., Any]
    marker_group: Callable[..., Any]


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

    def run(
        self,
        cfg,
        grad_thresh: float = 0.0002,
        scale_thresh: float | None = None,
        prune_big_scale: float | None = None,
        allow_scale_prune: bool = True,
        carry_eta: bool = False,
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
                action=resources.actions.storage,
                grad_thresh=grad_thresh_internal,
                scale_thresh=scale_thresh,
                t_opacity=cfg.t_opacity,
                t_big_scale=big_scale,
                allow_scale_prune=1 if allow_scale_prune else 0,
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
            commands.dispatch_indirect(
                kernels.split_and_clone, "split_and_clone", resources.dispatch_args,
                gaussians=primitive.values.storage,
                adam_state=adam.state.storage,
                num_gaussians_counter=primitive.active_count.storage,
                action=resources.actions.storage,
                pre_count_buf=resources.pre_count.storage,
                max_splats=self.max_primitives,
            )
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

            if carry_eta:
                commands.dispatch(
                    kernels.scatter_compact_eta, "scatter_compact_eta",
                    thread_count=[self.max_primitives, 1, 1],
                    gaussians_in=primitive.values.storage,
                    adam_state_in=adam.state.storage,
                    valid_mask=resources.valid_mask.storage,
                    scan=self.scan.resources.work.storage,
                    action=resources.actions.storage,
                    pre_count_buf=resources.pre_count.storage,
                    **self.eta.scatter_kwargs(),
                    gaussians_out=resources.primitive_scratch.storage,
                    adam_state_out=resources.adam_scratch.storage,
                    max_splats=self.max_primitives,
                )
            else:
                commands.dispatch(
                    kernels.scatter_compact, "scatter_compact",
                    thread_count=[self.max_primitives, 1, 1],
                    gaussians_in=primitive.values.storage,
                    adam_state_in=adam.state.storage,
                    valid_mask=resources.valid_mask.storage,
                    scan=self.scan.resources.work.storage,
                    gaussians_out=resources.primitive_scratch.storage,
                    adam_state_out=resources.adam_scratch.storage,
                    max_splats=self.max_primitives,
                )

            commands.dispatch(
                kernels.snapshot_count, "densify_snapshot",
                thread_count=[1, 1, 1],
                src=self.scan.resources.total.storage,
                dst=primitive.active_count.storage,
            )
            commit_eta = self.eta if carry_eta else None
            recording.after_submit(
                lambda: resources.commit_compaction(primitive, adam, commit_eta))

        return {"pre_count": None, "post_split_count": None, "final_count": None}
