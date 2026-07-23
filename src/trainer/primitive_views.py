"""Non-owning GPU views of a primitive population and its gradients.

These are VIEWS, not owners: they hold references to buffers someone else owns.
Today a view points at the pipeline's persistent explicit-Gaussian buffers;
later it can point at neural-network output buffers with NO host readback, which
is why the rasterizer (a future slice) will consume a ``PrimitiveSetView`` and
write into a caller-supplied ``PrimitiveGradientView`` rather than reaching into
``Pipeline.buffers``. Additive scaffolding — nothing consumes these yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import slangpy as spy

from trainer.primitive_types import DEFAULT_KEY, PrimitiveLayoutKey


@dataclass(frozen=True)
class PrimitiveSetView:
    """A non-owning view of a primitive population: the packed parameter buffer,
    the live-count buffer, the allocated capacity, and any primitive-specific
    auxiliary buffers (e.g. the Gaussian Mip 3D filter)."""
    parameters: spy.NDBuffer
    count: spy.NDBuffer
    capacity: int
    aux_filter_3d: spy.NDBuffer | None = None
    layout: PrimitiveLayoutKey | None = None


@dataclass(frozen=True)
class PrimitiveGradientView:
    """A non-owning view of the per-parameter gradient buffers the backward
    writes into (name -> buffer), tagged with the layout they belong to."""
    buffers: dict[str, spy.NDBuffer]
    layout: PrimitiveLayoutKey


def gaussian_primitive_set_view(buffers,
                                layout: PrimitiveLayoutKey = DEFAULT_KEY
                                ) -> PrimitiveSetView:
    """Zero-copy view over a :class:`~trainer.buffers.GaussianBuffers`'s persistent
    primitive state. The returned view aliases the SAME NDBuffers (no copy)."""
    return PrimitiveSetView(
        parameters=buffers.gaussians,
        count=buffers.num_gaussians_counter,
        capacity=buffers.max_splats,
        aux_filter_3d=getattr(buffers, "filter_3D", None),
        layout=layout,
    )


def gaussian_gradient_view(buffers,
                           layout: PrimitiveLayoutKey = DEFAULT_KEY
                           ) -> PrimitiveGradientView:
    """Zero-copy view over the 14 SoA scalar-gradient buffers (aliases the same
    NDBuffers keyed by :data:`~trainer.buffers.GRAD_SCALAR_COMPONENTS`)."""
    return PrimitiveGradientView(buffers=dict(buffers.grad_buffers), layout=layout)
