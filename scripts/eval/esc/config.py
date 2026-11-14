# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Tunable configuration for the ESC harness, as immutable dataclasses.

Weights, gate tolerances, concepts and static thresholds live here so scoring code has no magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class EscWeights:
    """Exponents of the weighted geometric mean; they must sum to 1.0 to keep the aggregate on the input scale."""

    correctness: float = 0.40
    edit_efficiency: float = 0.15
    understandability: float = 0.15
    locality: float = 0.15
    testability: float = 0.10
    static_quality: float = 0.05

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.correctness,
            self.edit_efficiency,
            self.understandability,
            self.locality,
            self.testability,
            self.static_quality,
        )

    def validate(self) -> None:
        total = sum(self.as_tuple())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"ESC weights must sum to 1.0, got {total:.6f}")


class CostNormalization(str, Enum):
    """How a raw agent cost becomes the non-negative ``r`` term of ``E``.

    RATIO: observed/baseline (baseline -> E=e^-1). EXCESS_RATIO: max(0, observed/baseline - 1) (baseline -> E=1).
    """

    RATIO = "ratio"
    EXCESS_RATIO = "excess_ratio"


@dataclass(frozen=True)
class CostWeights:
    """Weights inside ``E = exp(-median_t[ sum_k w_k r_{t,k} ])``."""

    tokens: float = 0.35
    time: float = 0.25
    retries: float = 0.20
    patch: float = 0.20
    normalization: CostNormalization = CostNormalization.EXCESS_RATIO

    def weight_map(self) -> dict[str, float]:
        return {
            "tokens": self.tokens,
            "time": self.time,
            "retries": self.retries,
            "patch": self.patch,
        }


@dataclass(frozen=True)
class GateThresholds:
    """Behavioural tolerances a candidate must satisfy to be *valid* at all."""

    numerical_tolerance: float = 1e-4
    max_perf_regression: float = 0.05  # +5% wall time is the ceiling
    require_determinism: bool = True
    forbid_abi_break: bool = True
    max_static_blockers: int = 0
    min_mutation_score: float = 0.0  # absolute floor; see max_mutation_drop
    max_mutation_drop: float = 0.05  # candidate may lose at most 5 pts vs champion


@dataclass(frozen=True)
class ConceptSpec:
    """A cross-cutting concept whose implementation dispersion we measure.

    ``patterns`` are case-insensitive regexes; each match counts once as an implementation site.
    """

    name: str
    patterns: tuple[str, ...]


#: The 3DGS trainer's load-bearing concepts: when they scatter, the codebase resists change.
DEFAULT_CONCEPTS: tuple[ConceptSpec, ...] = (
    ConceptSpec("opacity", (r"\bopacit", r"\bsigmoid\b")),
    ConceptSpec("densification", (r"densif", r"\bclone\b", r"\bsplit\b", r"\bprune\b")),
    ConceptSpec("optimizer", (r"\badam\b", r"optimi[sz]er", r"moment", r"learning_rate", r"\blr\b")),
    ConceptSpec("gaussian_field_layout", (r"Gaussian3D", r"GaussianChannel", r"StructLayout", r"float_count")),
    ConceptSpec("gradients", (r"gradient", r"\bgrad_", r"backward")),
    ConceptSpec("serialization", (r"\bply\b", r"serial", r"export", r"checkpoint")),
    ConceptSpec("projection", (r"project", r"\bcov2d\b", r"\bmean2d\b")),
    ConceptSpec("sh_color", (r"\bsh0\b", r"spherical_harmonic", r"\bf_dc\b", r"\bappearance\b")),
)


@dataclass(frozen=True)
class StaticThresholds:
    """"Good"/"bad" anchors per static metric: a decreasing curve maps ``good`` to ~1.0 and ``bad`` to ~0.0."""

    complexity_percentile: float = 0.95
    cyclomatic_good: float = 6.0
    cyclomatic_bad: float = 25.0
    cognitive_good: float = 8.0
    cognitive_bad: float = 40.0
    function_len_good: float = 40.0
    function_len_bad: float = 150.0
    module_len_good: float = 400.0
    module_len_bad: float = 1500.0
    fan_out_good: float = 8.0
    fan_out_bad: float = 30.0
    duplication_good: float = 0.02
    duplication_bad: float = 0.20
    # Relative contribution of each metric to S (normalised internally).
    metric_weights: tuple[tuple[str, float], ...] = (
        ("cyclomatic", 0.25),
        ("cognitive", 0.25),
        ("function_len", 0.15),
        ("module_len", 0.10),
        ("fan_out", 0.10),
        ("duplication", 0.10),
        ("cycles", 0.05),
    )


@dataclass(frozen=True)
class LocalityThresholds:
    """How the static concept-entropy term and the dynamic patch-spread term combine into ``L``."""

    #: Weight of static localisation (mean 1-H_c) vs measured patch spread; static is used alone without change runs.
    static_weight: float = 0.5
    files_good: float = 3.0
    files_bad: float = 15.0
    modules_good: float = 2.0
    modules_bad: float = 10.0
    penalty_unrelated: float = 0.15
    penalty_duplicate: float = 0.10
    penalty_parallel_path: float = 0.30
    penalty_no_extension_point: float = 0.20
    file_vs_module_split: float = 0.6  # weight on file term within patch spread
    #: "file" localises to individual files; "package" to owning directories.
    granularity: str = "file"


@dataclass(frozen=True)
class TestabilityThresholds:
    """How mutation/fault evidence and test hygiene combine into ``T``."""

    __test__ = False  # not a pytest test class despite the leading "Test"

    mutation_weight: float = 0.6
    fault_weight: float = 0.4
    nondeterministic_factor: float = 0.5
    per_flaky_penalty: float = 0.05
    weakened_tests_factor: float = 0.3
    runtime_good_s: float = 30.0
    runtime_bad_s: float = 600.0


@dataclass(frozen=True)
class UnderstandabilityWeights:
    answer: float = 0.45
    files: float = 0.35
    calibration: float = 0.10
    context: float = 0.10
    #: Context-token budget beyond which the context sub-term decays to 0.
    context_token_budget: int = 40_000


@dataclass(frozen=True)
class LoopControls:
    """Guards for the champion/candidate procedure."""

    max_patch_files: int = 12
    max_patch_lines: int = 600
    #: A candidate must clear this ESC gain at the paired-bootstrap confidence, with no subscore drop beyond the margin.
    min_esc_gain: float = 0.5
    bootstrap_confidence: float = 0.90
    max_subscore_regression: float = 0.03
    bootstrap_samples: int = 2000


@dataclass(frozen=True)
class EscConfig:
    """The single configuration object threaded through the harness."""

    weights: EscWeights = field(default_factory=EscWeights)
    cost: CostWeights = field(default_factory=CostWeights)
    gates: GateThresholds = field(default_factory=GateThresholds)
    static: StaticThresholds = field(default_factory=StaticThresholds)
    testability: TestabilityThresholds = field(default_factory=TestabilityThresholds)
    understanding: UnderstandabilityWeights = field(default_factory=UnderstandabilityWeights)
    locality: LocalityThresholds = field(default_factory=LocalityThresholds)
    loop: LoopControls = field(default_factory=LoopControls)
    concepts: tuple[ConceptSpec, ...] = DEFAULT_CONCEPTS
    #: Directories excluded from all repo scanning (heavy / generated / vendored).
    scan_excludes: tuple[str, ...] = (
        ".git", ".venv", "__pycache__", "datasets", "outputs", "nsight",
        "assets", ".pytest_cache", ".ruff_cache", "node_modules",
        "slang_3dgs_trainer.egg-info",
    )
    source_suffixes: tuple[str, ...] = (".py", ".slang")

    def __post_init__(self) -> None:
        self.weights.validate()
