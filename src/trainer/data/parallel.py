from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

MAX_WORKERS = 16

def default_workers_numbers(count: int | None = None) -> int:
    workers = min(MAX_WORKERS, max(1, (os.cpu_count() or 1) - 1))
    return workers if count is None else max(1, min(workers, int(count)))


def map_ordered(fn: Callable[[T], R], items: Iterable[T], *,
                workers: int | None = None, lookahead: int = 2) -> Iterator[R]:
    """Yield ``fn(item)`` for every item, IN ORDER, computed on a thread pool.

    ``workers`` defaults to :func:`default_workers`; at most
    ``workers * lookahead`` results are in flight (see the module docstring). An
    exception from ``fn`` propagates to the consumer on the item that raised,
    exactly as the equivalent list comprehension would.
    """
    items = list(items)
    if workers is None:
        workers = default_workers_numbers(len(items))
    if workers <= 1 or len(items) <= 1:
        yield from map(fn, items)          # no pool at all for a trivial sweep
        return

    limit = max(1, workers * max(1, lookahead))
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="dataload") as pool:
        pending: deque[Future[R]] = deque()
        for item in items:
            pending.append(pool.submit(fn, item))
            if len(pending) >= limit:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
