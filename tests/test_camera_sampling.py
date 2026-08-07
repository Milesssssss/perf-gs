"""Characterization test: freeze the per-iteration train-camera draw.

NON-NEGOTIABLE: the ``ours`` vs ``sadgs`` sampling order and the exact RNG
consumption must not drift. These tests pin the draw ORDER against an in-test
reference replica of each mode's documented consumption, using the SAME RNG
primitive on both sides (numpy Generator for ``ours``, ``random.Random`` for
``sadgs``) so the freeze tracks the code — not a specific library build.
"""

from __future__ import annotations

import random

import numpy as np

from trainer.frontend.evaluation.evaluation import CameraSampler


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
