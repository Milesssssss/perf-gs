"""Per-iteration train-camera sampling for the v2 trainer.

Isolates the "which camera do we optimize this step" responsibility behind one
``CameraSampler.next()`` call so the training loop stays free of RNG bookkeeping.
"""

from __future__ import annotations

import random

import numpy as np


class CameraSampler:
    """Draws a train-camera index each iteration. Two modes:

    ``ours``
        Uniform draw (with replacement) over the name-sorted train indices,
        using the SHARED numpy Generator so its stream stays interleaved with
        the densify view-scoring draws (preserving run-to-run determinism).

    ``sadgs``
        Byte-for-byte replica of the reference SADGS trainer's draw so a
        like-for-like PSNR comparison isolates the renderer, not the data order.
        The reference (INRIA lineage) consumes the Python ``random`` stream as:
        ``random.seed(0)``; ``Scene(shuffle=True)`` shuffles the train then the
        test cameras (the test shuffle advances the stream); epoch cycling pops
        the load-shuffled order IN ORDER for the first epoch (no reshuffle),
        then reshuffles on every refill before popping from the front. We mirror
        that with a private ``random.Random(seed)`` (identical sequence to a
        fresh global seed, but isolated), same shuffle-call order over the same
        train/test index lists, ``batch_size=1``, without replacement. Our
        camera order already matches SADGS (both name-sort then drop every
        ``stride``-th as holdout), so index i maps to the same image on both
        sides. Bit-exactness assumes SADGS makes no other ``random.*`` draws in
        the loop (holds for the SH0 bicycle config: random_background off,
        densify uses torch RNG).
    """

    def __init__(self, mode: str, train_indices, test_indices, seed: int,
                 rng: np.random.Generator) -> None:
        self.mode = mode
        self._rng = rng
        self._train_arr = np.asarray(train_indices, dtype=np.int64)
        if mode == "sadgs":
            self._py = random.Random(seed)
            train = list(train_indices)   # name-sorted ascending, minus holdout
            test = list(test_indices)
            self._py.shuffle(train)       # Scene load-time shuffle (train)
            self._py.shuffle(test)        # Scene load-time shuffle (test) — consumes stream
            self._base = list(train)      # getTrainCameras() order
            self._stack = list(self._base)
            print(f"[sample-mode=sadgs] replicating SADGS random-shuffle sampling "
                  f"(seed={seed}, {len(self._base)} train cams, batch_size=1, "
                  f"first epoch in load-shuffled order, reshuffle on refill)")

    def next(self) -> int:
        """Return the train-camera index to optimize this iteration."""
        if self.mode == "sadgs":
            # SADGS pops from the front; on an exhausted epoch it refills from
            # getTrainCameras().copy() and reshuffles before the next pop.
            if not self._stack:
                self._stack.extend(self._base)
                self._py.shuffle(self._stack)
            return self._stack.pop(0)
        return int(self._train_arr[self._rng.integers(0, self._train_arr.shape[0])])
