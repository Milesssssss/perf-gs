# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Concept-entropy math: localisation, even scatter, single-module case."""

from __future__ import annotations

import math
from pathlib import Path

from esc.config import ConceptSpec
from esc.repo import SourceFile
from esc.scoring.concept_entropy import compute_concept_entropy


def _file(rel: str, text: str) -> SourceFile:
    return SourceFile(path=Path(rel), rel=rel, text=text)


_CONCEPT = ConceptSpec("foo", (r"foo",))


def test_single_module_is_fully_localised():
    files = [_file("a.py", "foo foo foo"), _file("b.py", "bar baz")]
    result = compute_concept_entropy(files, _CONCEPT)
    assert result.present
    assert result.modules == 1
    assert result.entropy == 0.0
    assert result.localization == 1.0


def test_even_scatter_is_maximum_entropy():
    files = [_file("a.py", "foo"), _file("b.py", "foo")]
    result = compute_concept_entropy(files, _CONCEPT)
    assert result.modules == 2
    assert math.isclose(result.entropy, 1.0, rel_tol=1e-9)
    assert math.isclose(result.localization, 0.0, abs_tol=1e-9)


def test_skewed_scatter_is_intermediate():
    files = [_file("a.py", "foo foo foo"), _file("b.py", "foo")]
    result = compute_concept_entropy(files, _CONCEPT)
    # p = (0.75, 0.25); H = -(.75 ln.75 + .25 ln.25)/ln2 ~ 0.811.
    assert 0.0 < result.entropy < 1.0
    assert math.isclose(result.entropy, 0.8113, abs_tol=1e-3)


def test_absent_concept_reports_not_present():
    files = [_file("a.py", "nothing here")]
    result = compute_concept_entropy(files, _CONCEPT)
    assert not result.present
    assert result.localization == 1.0


def test_package_granularity_buckets_by_directory():
    files = [_file("src/pkg/a.py", "foo"), _file("src/pkg/b.py", "foo")]
    result = compute_concept_entropy(files, _CONCEPT, granularity="package")
    # Both files live in the same package -> one bucket -> localised.
    assert result.modules == 1
    assert result.localization == 1.0
