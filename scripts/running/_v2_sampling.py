# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Per-iteration train-camera sampling for the v2 trainer.

Isolates the "which camera do we optimize this step" responsibility behind one
``CameraSampler.next()`` call so the training loop stays free of RNG bookkeeping.
"""

from __future__ import annotations

import random

import numpy as np

class CameraSampler:
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
