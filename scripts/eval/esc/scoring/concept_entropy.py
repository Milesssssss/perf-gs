"""Concept entropy — how scattered a cross-cutting concept's implementation is.

For a concept ``c`` spread over modules ``i`` with implementation-site fractions
``p_{c,i}``:

    H_c = - ( sum_i p_{c,i} log p_{c,i} ) / log m_c      in [0, 1]

``H_c ~ 0`` means the concept is localised to one module; ``H_c ~ 1`` means its
responsibility is evenly scattered across many. Locality rewards ``1 - H_c``.

Crucially this is *not* optimised alone: a 20k-line god file has near-zero
dispersion while being awful. :mod:`esc.scoring.locality` balances entropy
against patch spread and the static-quality module-size penalty.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from esc.config import ConceptSpec
from esc.repo import SourceFile


def module_of(rel: str, granularity: str) -> str:
    """Map a file path to its module bucket at the requested granularity."""
    if granularity == "file":
        return rel
    parts = rel.split("/")
    if parts and parts[0] == "src":
        parts = parts[1:]
    # package granularity: the directory that owns the file (or the file itself
    # if it sits at the top level).
    return "/".join(parts[:-1]) or parts[-1]


@dataclass(frozen=True)
class ConceptEntropy:
    name: str
    sites: int
    modules: int
    entropy: float
    localization: float  # 1 - H_c
    present: bool
    top_modules: tuple[tuple[str, int], ...]


def _match_count(patterns: tuple[re.Pattern[str], ...], text: str) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def compute_concept_entropy(
    files: Iterable[SourceFile],
    concept: ConceptSpec,
    granularity: str = "file",
) -> ConceptEntropy:
    """Normalised implementation-dispersion entropy for one concept."""
    patterns = tuple(re.compile(p, re.IGNORECASE) for p in concept.patterns)

    per_module: dict[str, int] = {}
    for source in files:
        count = _match_count(patterns, source.text)
        if count:
            bucket = module_of(source.rel, granularity)
            per_module[bucket] = per_module.get(bucket, 0) + count

    total = sum(per_module.values())
    m_c = len(per_module)
    if total == 0 or m_c == 0:
        return ConceptEntropy(concept.name, 0, 0, 0.0, 1.0, present=False, top_modules=())
    if m_c == 1:
        (only_module, only_count), = per_module.items()
        return ConceptEntropy(concept.name, total, 1, 0.0, 1.0, present=True,
                              top_modules=((only_module, only_count),))

    entropy = 0.0
    for count in per_module.values():
        p = count / total
        entropy -= p * math.log(p)
    normalized = entropy / math.log(m_c)

    top = tuple(sorted(per_module.items(), key=lambda kv: kv[1], reverse=True)[:5])
    return ConceptEntropy(
        name=concept.name,
        sites=total,
        modules=m_c,
        entropy=normalized,
        localization=1.0 - normalized,
        present=True,
        top_modules=top,
    )


def concept_entropy_report(
    files: list[SourceFile],
    concepts: Iterable[ConceptSpec],
    granularity: str = "file",
) -> list[ConceptEntropy]:
    """Concept entropy for every configured concept."""
    return [compute_concept_entropy(files, c, granularity) for c in concepts]
