"""Optional images -> COLMAP reconstruction, driven by the colmap/glomap binaries.

Produces a ``sparse/0`` + ``images`` dataset that the existing loader
(:func:`trainer.data.colmap.load_colmap`) consumes unchanged. Fully OPTIONAL: the
trainer runs without it, and the binaries are downloaded on demand from the
viewer's hub (:class:`ReconTools`) rather than being dependencies.
"""
from __future__ import annotations

from trainer.frontend.reconstruction.pipeline import classify_dataset, reconstruct
from trainer.frontend.reconstruction.tools import ReconTools

__all__ = ["ReconTools", "classify_dataset", "reconstruct"]
