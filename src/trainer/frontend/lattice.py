# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Train on interleaved native pixel centers within the full-frame allocation."""
from dataclasses import replace

import numpy as np
import slangpy as spy

from trainer.backend.densify.structure_tensor import StructureTensor
from trainer.backend.render.targets import TargetImages
from trainer.config.config_types import DensificationMode


def last_lattice_step(args):
    return min(args.lattice_until, args.iters - 1) if args.lattice_stride > 1 else 0


def needs_native_scores(args):
    if not last_lattice_step(args):
        return True
    last_score = min(args.iters, args.densify_to) // args.sad_accum_every * args.sad_accum_every
    return last_score > max(args.densify_from, last_lattice_step(args))


def needs_lattice_scores(args):
    first_score = (args.densify_from // args.sad_accum_every + 1) * args.sad_accum_every
    return first_score <= min(last_lattice_step(args), args.densify_to)


def lattice_camera(camera, phase, stride):
    """Coarse pixel center j+.5 samples native pixel center stride*j+phase+.5."""
    size = tuple(length // stride for length in camera.image_size)
    proj = camera.proj_matrix.copy()
    for axis, (native, coarse, offset) in enumerate(zip(camera.image_size, size, phase)):
        span = coarse * stride
        proj[axis] *= native / span
        proj[axis] += camera.proj_matrix[3] * ((native - 2 * offset + stride - 1) / span - 1)
    return replace(camera, proj_matrix=proj, image_size=size)


class LatticeTargets(TargetImages):
    def __init__(self, pipeline, stride):
        width, height = pipeline.width // stride, pipeline.height // stride
        if min(width, height) < 1:
            raise ValueError("Lattice stride exceeds the training image size")
        texture = pipeline.device.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba8_unorm, width=width, height=height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access, label="lattice_target")
        super().__init__(pipeline.device, pipeline.commands, texture, width, height)
        self._gather = pipeline.kernels.kernel("losses/lattice_target.slang", "gatherTarget")
        self.stride = stride
        self.phase = (0, 0)
        self._resident = None

    def select(self, target):
        self._resident = None
        self.active = self._owned
        if isinstance(target, np.ndarray):
            x, y = self.phase
            s = self.stride
            rows = slice(y, y + self.height * s, s)
            columns = slice(x, x + self.width * s, s)
            self._pending = np.ascontiguousarray(target[rows, columns])
        else:
            self._resident = target
            self._pending = None

    def flush(self):
        if self._resident is not None:
            self._commands.dispatch(self._gather, "gather_lattice_target", thread_count=[self.width, self.height, 1],
                                    source=self._resident, target=self._owned, size=spy.uint2(self.width, self.height),
                                    phase=spy.uint2(*self.phase), stride=self.stride)
            self._resident = None
        super().flush()


class TrainingLattice:
    def __init__(self, loop):
        self.pipeline = loop.runtime.pipeline
        self.densifier = loop.runtime.densifier
        self.native = self.pipeline.targets
        self.native_config = self.pipeline.rasterizer.config
        self.targets = LatticeTargets(self.pipeline, loop.args.lattice_stride)
        self.config = replace(self.native_config, kernel_size=self.native_config.kernel_size / self.targets.stride**2)
        self.structure = (StructureTensor(self.pipeline.device, self.targets.width, self.targets.height)
                          if loop.densify_mode == DensificationMode.STRUCTURE_AWARE and needs_lattice_scores(loop.args) else None)
        self.visits = {}

    def activate(self, entry, camera_index, seed):
        stride = self.targets.stride
        visit = self.visits.get(camera_index, 0)
        self.visits[camera_index] = visit + 1
        # Stagger cameras' starting phases while visiting every phase once per cycle.
        phase = (camera_index * 2654435761 + seed + visit) % (stride * stride)
        self.targets.phase = phase % stride, phase // stride
        self.select(self.targets, self.config)
        return replace(entry, camera=lattice_camera(entry.camera, self.targets.phase, stride))

    def restore(self):
        self.select(self.native, self.native_config)

    def select(self, targets, config):
        if self.pipeline.targets is not targets:
            self.pipeline.set_training_viewport(targets, config)
            self.densifier.set_image_size(targets.width, targets.height)
