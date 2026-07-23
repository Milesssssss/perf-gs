"""Training-application layer for the trainer frontend.

Houses the extracted training-orchestration components — starting with the pure
per-iteration :mod:`~trainer.training.schedule`, and (in later refactor slices)
config resolution, resources, outputs, evaluation, and the loop. Kept separate
from the primitive/render layers in ``trainer`` so application concerns (WHEN
things happen) do not bleed into the rasterizer (HOW a primitive renders).
"""
