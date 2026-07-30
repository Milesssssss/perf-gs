"""Expected Safe Changeability (ESC) evaluation harness.

ESC scores a codebase by *how well it absorbs unseen change* rather than by how
clean it looks. The headline number in ``[0, 100]`` is a weighted geometric mean
of six behavioural subscores:

    ESC = 100 * C^0.40 * E^0.15 * U^0.15 * L^0.15 * T^0.10 * S^0.05

    C  correctness on unseen future-change tasks   (largest weight)
    E  edit efficiency
    U  understandability
    L  change locality
    T  testability and regression safety
    S  static structural quality                   (smallest weight)

A geometric mean is used deliberately: one excellent dimension cannot mask a
catastrophic one. Hard behavioural gates (tests, numerical parity, performance,
ABI, determinism) run *before* scoring, so cleanliness can never be traded for
correctness.

The package is organised in reading order, top-down:

    config      tunable weights, thresholds and concept definitions
    model       immutable result/telemetry value objects + the ESC math seam
    scoring/    the six subscore reducers and the final aggregation
    gates       hard behavioural gates that gate scoring
    agents/     the pluggable coding-agent boundary (scripted / subprocess)
    tasks/      the hidden change-task and comprehension-task families
    workspace   ephemeral sandboxes for candidate evaluation
    evaluate    paired champion/candidate evaluation over tasks x seeds
    loop        the stochastic champion/candidate iteration procedure
    report      the diagnostic dashboard renderer
    cli         the command-line entry point
"""

from __future__ import annotations

__version__ = "0.1.0"
