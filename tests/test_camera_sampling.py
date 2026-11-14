# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Characterization test: freeze the per-iteration train-camera draw and STBN slice walk.

NON-NEGOTIABLE: the ``ours`` vs ``sadgs`` sampling order and the exact RNG
consumption must not drift. These tests pin the draw ORDER against an in-test
reference replica of each mode's documented consumption, using the SAME RNG
primitive on both sides (numpy Generator for ``ours``, ``random.Random`` for
``sadgs``) so the freeze tracks the code — not a specific library build.
"""

from __future__ import annotations

import random

import numpy as np

from trainer.backend.compute.execution import TILE_AREA
from trainer.config.config import StbnConfig
from trainer.frontend.sampling import CameraSampler, StbnSampler


def test_ours_matches_reference_rng_order() -> None:
    """`ours` draws one integers(0, n_train) per next(), indexing the name-sorted
    train array, consuming the SHARED numpy Generator in that exact order."""
    train = [1, 2, 3, 4, 5, 6, 7]   # e.g. holdout stride 8 over 0..7 drops cam 0
    test = [0]
    seed = 0
    sampler = CameraSampler("ours", train, test, seed, np.random.default_rng(seed))
    draws = [sampler.next() for _ in range(25)]

    # Reference: identical Generator seed, one integers(0, n) draw per step.
    rng = np.random.default_rng(seed)
    train_arr = np.asarray(train, dtype=np.int64)
    expected = [int(train_arr[rng.integers(0, train_arr.shape[0])]) for _ in range(25)]

    assert draws == expected
    assert all(d in train for d in draws)


def test_ours_deterministic_same_seed() -> None:
    train = list(range(1, 8))
    test = [0]
    a = CameraSampler("ours", train, test, 0, np.random.default_rng(0))
    b = CameraSampler("ours", train, test, 0, np.random.default_rng(0))
    assert [a.next() for _ in range(40)] == [b.next() for _ in range(40)]


def test_sadgs_matches_reference_shuffle_order() -> None:
    """`sadgs` replicates the reference draw: random.Random(seed), shuffle(train)
    then shuffle(test) (the test shuffle advances the stream), first epoch popped
    from the front in load-shuffled order, reshuffle-on-refill afterward."""
    train = [1, 2, 3, 4, 5, 6, 7]
    test = [0, 8, 16]
    seed = 0
    sampler = CameraSampler("sadgs", train, test, seed, np.random.default_rng(seed))
    n = len(train) * 2 + 3   # span two epoch boundaries
    draws = [sampler.next() for _ in range(n)]

    # Reference replica of the reference SADGS consumption.
    py = random.Random(seed)
    tr = list(train)
    te = list(test)
    py.shuffle(tr)          # Scene load-time train shuffle
    py.shuffle(te)          # Scene load-time test shuffle (consumes the stream)
    base = list(tr)
    stack = list(base)
    expected = []
    for _ in range(n):
        if not stack:
            stack.extend(base)
            py.shuffle(stack)
        expected.append(stack.pop(0))

    assert draws == expected
    # First epoch is exactly the load-shuffled train order, in order.
    assert draws[: len(train)] == base
    # Every draw is a train camera; the first epoch is a permutation (no repeats).
    assert set(draws[: len(train)]) == set(train)


def test_sadgs_deterministic_same_seed() -> None:
    train = [1, 2, 3, 4, 5, 6, 7]
    test = [0, 8]
    a = CameraSampler("sadgs", train, test, 3, np.random.default_rng(3))
    b = CameraSampler("sadgs", train, test, 3, np.random.default_rng(3))
    assert [a.next() for _ in range(20)] == [b.next() for _ in range(20)]


# --- STBN: the mask's temporal axis is per-CAMERA, not per-iteration ----------

def _stbn_sampler(target_fraction=0.1, warmup=0, slices=64, seed=0, independent=False):
    return StbnSampler(StbnConfig(pixel_fraction=target_fraction, warmup_iters=warmup,
                                  seed=seed, independent=independent), slices)


def test_stbn_slice_advances_per_camera_visit_not_per_iteration() -> None:
    """The mask distributes a pixel's samples along its TEMPORAL axis, and the
    sequence that axis is meant to track is "successive evaluations of the same
    image". So drawing A, B, A, C, A must walk A's slices 0,1,2 while B and C
    each stay on their own first slice."""
    sampler = _stbn_sampler()
    draws = [7, 3, 7, 9, 7, 3]
    slices = [sampler.variables(cam, 1).slice_index for cam in draws]

    phase = sampler.phases
    assert slices == [
        phase[7] + 0, phase[3] + 0, phase[7] + 1,
        phase[9] + 0, phase[7] + 2, phase[3] + 1,
    ]
    assert sampler.visits == {7: 3, 3: 2, 9: 1}


def test_stbn_independent_draws_random_slices_decoupled_from_visits() -> None:
    """--stbn-independent draws a fresh random slice each step from its own RNG,
    so the camera draw order cannot move and two runs with one seed agree."""
    sampler = _stbn_sampler(slices=64, seed=7, independent=True)
    drawn = [sampler.variables(5, 1).slice_index for _ in range(8)]
    assert all(0 <= s < 64 for s in drawn)
    assert len(set(drawn)) > 1               # not a fixed or per-visit sequence
    assert sampler.visits == {} and sampler.phases == {}

    again = _stbn_sampler(slices=64, seed=7, independent=True)
    assert [again.variables(5, 1).slice_index for _ in range(8)] == drawn


def test_stbn_phases_differ_across_cameras_and_wrap_into_range() -> None:
    """A shared start would couple the whole dataset to one sequence."""
    sampler = _stbn_sampler(slices=64)
    for cam in range(40):
        sampler.variables(cam, 1)
    phases = [sampler.phases[cam] for cam in range(40)]
    assert all(0 <= p < 64 for p in phases)
    assert len(set(phases)) > 20          # spread out, not all-equal


def test_stbn_slice_wraps_at_the_masks_temporal_period() -> None:
    sampler = _stbn_sampler(slices=4)
    slices = [sampler.variables(2, 1).slice_index for _ in range(9)]
    assert all(0 <= s < 4 for s in slices)
    assert slices[:4] == slices[4:8]       # one full cycle, then repeat


def test_stbn_phase_is_reproducible_and_independent_of_the_camera_draw() -> None:
    """Phases mix the camera index rather than drawing from the run's RNG, which
    would shift every later camera draw and break the order frozen above."""
    first = _stbn_sampler(seed=5)
    second = _stbn_sampler(seed=5)
    for cam in (0, 1, 2, 3):
        assert first.variables(cam, 1).slice_index == second.variables(cam, 1).slice_index
    reseeded = _stbn_sampler(seed=6)
    assert reseeded.phases == {}           # no eager fill
    assert any(reseeded.variables(cam, 1).slice_index != first.phases[cam]
               for cam in (0, 1, 2, 3))


def test_stbn_budget_follows_the_density_ramp() -> None:
    """Full tile during warmup, then the steady-state per-tile count."""
    sampler = _stbn_sampler(target_fraction=0.1, warmup=1000)
    assert sampler.variables(0, 1).pixels_per_tile == TILE_AREA
    assert sampler.variables(0, 2000).pixels_per_tile == round(0.1 * TILE_AREA)
