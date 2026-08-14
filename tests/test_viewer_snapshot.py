"""What the viewer copies out of the trainer, and what it tells its rasterizer.

The snapshot is the trainer's only per-iteration cost for having a viewer open, so
it copies the ACTIVE splat range rather than the buffers' ``--max-splats``
allocation. That makes the published count safety-critical: a count ahead of the
copied range would have the viewer's rasterizer walk uninitialized tail memory.

Device-free — the encoder, the buffers and both pipelines are fakes, so what is
under test is the sizing arithmetic and the count discipline, not the GPU.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np

from trainer.frontend.viewer.viewer import Viewer, _copy_splats

GAUSSIAN_STRIDE = 56        # float3 mean + float3 log_scale + float4 quat + 2 floats
CAPACITY = 10_000_000


class _Buffer:
    def __init__(self, size: int) -> None:
        self.size = size


class _Encoder:
    def __init__(self) -> None:
        self.copies: list[tuple[_Buffer, int]] = []
        self.uploads: list[tuple[_Buffer, np.ndarray]] = []

    def copy_buffer(self, dst, dst_offset, src, src_offset, size):
        assert (dst_offset, src_offset) == (0, 0)
        self.copies.append((dst, size))

    def upload_buffer_data(self, buffer, offset, data):
        assert offset == 0
        self.uploads.append((buffer, np.asarray(data)))


# --------------------------------------------------------------------------
# _copy_splats: bytes, not elements
# --------------------------------------------------------------------------

def test_only_the_active_range_is_copied() -> None:
    encoder = _Encoder()
    src, dst = _Buffer(CAPACITY * GAUSSIAN_STRIDE), _Buffer(CAPACITY * GAUSSIAN_STRIDE)
    _copy_splats(encoder, src, dst, count=150_000, capacity=CAPACITY)
    assert encoder.copies == [(dst, 150_000 * GAUSSIAN_STRIDE)]
    # ... which is what the fix is for: a snapshot at 1.5% occupancy used to move
    # the whole 534 MiB allocation.
    assert encoder.copies[0][1] < src.size // 50


def test_the_stride_comes_from_each_buffers_own_size() -> None:
    """filter_3D is 4 B/splat and features_rest is half-packed — one helper, no
    layout knowledge in it."""
    encoder = _Encoder()
    for stride in (4, 56, 92):
        src, dst = _Buffer(CAPACITY * stride), _Buffer(CAPACITY * stride)
        _copy_splats(encoder, src, dst, count=1_000, capacity=CAPACITY)
        assert encoder.copies[-1] == (dst, 1_000 * stride)


def test_a_smaller_destination_is_never_overrun() -> None:
    encoder = _Encoder()
    src = _Buffer(CAPACITY * GAUSSIAN_STRIDE)
    dst = _Buffer(1_000 * GAUSSIAN_STRIDE)          # viewer configured smaller
    _copy_splats(encoder, src, dst, count=500_000, capacity=CAPACITY)
    assert encoder.copies == [(dst, dst.size)]


def test_an_empty_model_copies_nothing() -> None:
    encoder = _Encoder()
    _copy_splats(encoder, _Buffer(1024), _Buffer(1024), count=0, capacity=CAPACITY)
    assert encoder.copies == []


# --------------------------------------------------------------------------
# update_gaussians: the published count is the copied count
# --------------------------------------------------------------------------

class _Runtime:
    def __init__(self, encoder: _Encoder) -> None:
        self.encoder = encoder
        self.submitted = 0

    def command_batch(self, _label):
        return SimpleNamespace(encoder=self.encoder)

    def submit(self, _batch):
        self.submitted += 1

    def queue_scope(self):
        return contextlib.nullcontext()


def _buffers(*, count: int, capacity: int = CAPACITY, sh: bool = False):
    """A stand-in for GaussianBuffers: the four per-splat buffers + the counter."""
    counter = SimpleNamespace(storage=_Buffer(4),
                              to_numpy=lambda: np.array([count], dtype=np.uint32))
    return SimpleNamespace(
        max_splats=capacity,
        sh_enabled=sh,
        gaussians=SimpleNamespace(storage=_Buffer(capacity * GAUSSIAN_STRIDE)),
        features_rest=SimpleNamespace(storage=_Buffer(capacity * 92)),
        filter_3D=SimpleNamespace(storage=_Buffer(capacity * 4)),
        num_gaussians_counter=counter,
    )


def _snapshot(*, live: int, viewer_capacity: int = CAPACITY, sh: bool = False,
              mip: bool = False, active_sh_degree: int = 0):
    """Run the real ``update_gaussians`` against fakes; return (encoder, viewer)."""
    encoder = _Encoder()
    view_buffers = _buffers(count=0, capacity=viewer_capacity, sh=sh)
    invalidations: list[int] = []
    viewer = SimpleNamespace(
        runtime=_Runtime(encoder),
        max_splats=viewer_capacity,
        _has_data=False,
        _pipeline=SimpleNamespace(
            buffers=view_buffers,
            raster_config=SimpleNamespace(mip_enabled=mip),
            set_active_sh_degree=lambda degree: None,
        ),
        live_splat_count=lambda pipeline: Viewer.live_splat_count(viewer, pipeline),
        # While training and parked on a test camera the render loop re-rasterizes
        # ONLY on this signal (see Viewer._scene_due), so a snapshot that failed to
        # raise it would leave the previous model on screen indefinitely.
        invalidate_scene=lambda: invalidations.append(1),
        invalidations=invalidations,
    )
    train = SimpleNamespace(buffers=_buffers(count=live, sh=sh),
                            _active_sh_degree=active_sh_degree)
    Viewer.update_gaussians(viewer, train)
    return encoder, viewer, view_buffers


def test_the_count_the_viewer_is_given_is_the_count_that_was_copied() -> None:
    """NOT the trainer's counter: the readback can be a densify behind by the time
    the batch is recorded, and a larger count would render uninitialized memory."""
    encoder, viewer, view_buffers = _snapshot(live=150_000)
    assert encoder.uploads == [(view_buffers.num_gaussians_counter.storage,
                                np.array([150_000], dtype=np.uint32))]
    copied_bytes = dict(encoder.copies)[view_buffers.gaussians.storage]
    assert copied_bytes == 150_000 * GAUSSIAN_STRIDE
    assert viewer._has_data is True


def test_the_counter_travels_in_the_same_batch_as_the_data() -> None:
    """One queue, so a submitted batch is atomic against viewer frames — the count
    and the splats it describes must not be two submissions."""
    encoder, viewer, _ = _snapshot(live=1_000)
    assert viewer.runtime.submitted == 1
    assert len(encoder.uploads) == 1 and encoder.copies


def test_a_count_beyond_the_viewers_capacity_is_clamped() -> None:
    encoder, _, view_buffers = _snapshot(live=5_000_000, viewer_capacity=1_000_000)
    assert encoder.uploads[0][1].tolist() == [1_000_000]
    assert dict(encoder.copies)[view_buffers.gaussians.storage] == \
        1_000_000 * GAUSSIAN_STRIDE


def test_sh_and_mip_buffers_follow_the_same_count() -> None:
    encoder, _, view_buffers = _snapshot(live=2_000, sh=True, mip=True)
    sizes = dict(encoder.copies)
    assert sizes[view_buffers.gaussians.storage] == 2_000 * GAUSSIAN_STRIDE
    assert sizes[view_buffers.features_rest.storage] == 2_000 * 92
    assert sizes[view_buffers.filter_3D.storage] == 2_000 * 4


def test_nothing_is_copied_before_the_first_splat_exists() -> None:
    encoder, viewer, _ = _snapshot(live=0)
    assert encoder.copies == []
    assert encoder.uploads[0][1].tolist() == [0]
    assert viewer._has_data is False          # nothing to rasterize yet


def test_a_snapshot_invalidates_the_rendered_scene() -> None:
    """The scene budget (Viewer._scene_due) skips the rasterize while training
    unless something says the model changed. A snapshot IS that something — and
    parked on a test camera it is the only trigger there is, so forgetting it
    would freeze the picture for the whole run."""
    _, viewer, _ = _snapshot(live=1_000)
    assert viewer.invalidations == [1]
