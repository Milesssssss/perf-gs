# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Importance-guided sparse backward sampling: the experiment suite.

Answers one question — can the same sparse backward budget be allocated toward
pixels with larger expected gradient contribution, while keeping STBN's
spatial/temporal sample structure AND the original dense training objective? —
in six stages, each independently runnable and resumable:

    sampling    the probability field and the sampling process, no GPU:
                calibration (does the empirical selection frequency match p_i?)
                and variable-density blue-noise diagnostics on the
                variance-normalized residual.
    visuals     per-camera panels: render, GT, s_i, s~_i, p_i, uniform mask,
                adaptive mask, and repeat-visit accumulation. Needs a GPU.
    gradients   frozen-state gradient fidelity against the DENSE gradient, at
                equal expected rho, for uniform and adaptive arms at K = 1..64.
    training    the five-scene pilot: dense + three uniform + three adaptive.
    hyper       alpha and sigma robustness, on two scenes only.
    fractions   rho in {0.10, 0.15, 0.25} for uniform vs adaptive STBN — the
                comparison that would convert better allocation into a smaller
                backward budget at equal quality.
    report      report.md, answering each question explicitly and surfacing the
                negative results.

Stage order matters: ``sampling`` and ``gradients`` are gates. If the estimator
is not calibrated and not unbiased, the training numbers are uninterpretable and
running them is a waste of GPU hours.

Everything is keyed by CONFIGURATION and hashed, so a resumed run re-trains only
what changed, and a cell whose adaptive settings differ can never be served from
a cached directory belonging to different ones.
"""
