# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Experiment F — gradient-quality diagnostic on a REAL trained scene state.

Asks one question: how close is the gradient a sparse mask produces to the one
dense supervision would have produced, and how fast does that close as the mask
is accumulated over K consecutive visits?

Structured so it CANNOT influence training. It is a separate process, it is never
invoked by a training cell, and inside it:

  * the scene is advanced to the target iteration by the normal training loop, so
    the state examined is a genuine mid-training state with real densification
    history — not a synthetic one;
  * from then on NO optimizer step is taken. The probe records forward + loss +
    backward, reads the gradient buffers, and stops. Parameters, Adam moments and
    the densification accumulators are never written;
  * the camera is pinned and the Gaussians are frozen, so the only thing varying
    between measurements is the mask.

Accumulation, and why the mean is the right thing to compare. The sparse loss is
already reweighted by 1/rho, so ONE masked backward is an unbiased estimate of the
dense gradient. K masks therefore accumulate as a MEAN of K unbiased estimates,
not a sum — a sum would be K times too large and every cosine similarity would
still look fine while every relative L2 error looked catastrophic. Because the
backward is linear in the loss, the mean of K individually-read gradients is
exactly the gradient of the averaged loss, so the probe reads each mask's gradient
separately (clearing in between) and averages on the host. That also means a
single pass over 16 masks yields every K in {1,2,4,8,16} as a prefix mean, rather
than needing five passes.

Two comparisons per parameter group:

    cosine similarity   direction agreement — what actually matters to an
                        optimizer, since Adam renormalizes magnitude per-parameter
    relative L2 error   ||sparse - dense|| / ||dense|| — total error including
                        magnitude, the stricter and less flattering number

Both are reported per parameter group (position, scale, rotation, opacity, colour)
because the groups have very different sparsity: a position gradient comes from a
handful of pixels under one splat, while a colour gradient integrates over its
whole footprint, so they should not be expected to degrade alike.

Usage (invoked by the suite; runnable by hand):

    .venv/Scripts/python.exe scripts/running/paper_gradient_probe.py \
        --scene datasets/360_v2/bicycle --state-iters 500 1500 3000 \
        --out outputs/paper_stbn/gradient_analysis/bicycle.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent
for _sub in ("running", "debug", "plots"):
    _p = str(_SCRIPTS / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

from trainer.backend.render import stbn as stbn_module
from trainer.config.config import (DensificationProperties, IterationContextualVariables, StbnVariables)
from trainer.contracts.codegen import TILE_AREA
from trainer.frontend.app import TrainingApplication
from trainer.frontend.cli import build_parser

# The 14 scalar gradient components, grouped as a paper would report them.
# Names must match trainer.backend.render.pipeline.GRAD_SCALAR_COMPONENTS
# exactly — a typo would silently drop that group from every table rather than
# fail, so :func:`_validate_groups` checks the mapping is total at startup.
PARAM_GROUPS: dict[str, tuple[str, ...]] = {
    "position": ("mean_x", "mean_y", "mean_z"),
    # The optimizer parameterizes scale in log space; the gradient buffers are
    # named accordingly.
    "scale": ("log_scale_x", "log_scale_y", "log_scale_z"),
    "rotation": ("rot_x", "rot_y", "rot_z", "rot_w"),
    "opacity": ("opacity",),
    "color_sh0": ("color_r", "color_g", "color_b"),
}
FIXED_POINT_INV = 1.0 / 65536.0

DEFAULT_K_VALUES = (1, 2, 4, 8, 16)
DEFAULT_STATE_ITERS = (500, 1500, 3000)
# Regimes, as (label, stbn source, independent-draw?). Mirrors the trainer's
# --stbn-source / --stbn-independent combinations exactly.
SPARSE_REGIMES = (("iid_white", stbn_module.SOURCE_WHITE, True),
                  ("indep_blue", stbn_module.SOURCE_NVIDIA, True),
                  ("stbn", stbn_module.SOURCE_NVIDIA, False))


def _grad_components() -> tuple[str, ...]:
    from trainer.backend.render.pipeline import GRAD_SCALAR_COMPONENTS
    return tuple(GRAD_SCALAR_COMPONENTS)


def _validate_groups() -> None:
    """Fail loudly if PARAM_GROUPS does not exactly partition the gradient set.

    A misnamed component would otherwise make its group silently disappear from
    the results — the failure mode where a reader concludes "the paper didn't
    report scale gradients" when in fact the key was misspelled.
    """
    actual = set(_grad_components())
    mapped = {c for group in PARAM_GROUPS.values() for c in group}
    missing, extra = actual - mapped, mapped - actual
    if missing or extra:
        raise RuntimeError(
            "PARAM_GROUPS does not partition GRAD_SCALAR_COMPONENTS: "
            f"unmapped={sorted(missing)} unknown={sorted(extra)}")


def read_gradients(pipeline) -> dict[str, np.ndarray]:
    """The 14 int64 fixed-point gradient buffers, as floats, for live splats."""
    pipeline.runtime.wait_for_idle()
    count = int(pipeline.buffers.num_gaussians_counter
                .to_numpy().view(np.uint32)[0])
    out: dict[str, np.ndarray] = {}
    for name in _grad_components():
        raw = pipeline.buffers.grad_buffers[name].to_numpy()
        ints = np.frombuffer(raw.tobytes(), dtype=np.int64)[:count]
        out[name] = ints.astype(np.float64) * FIXED_POINT_INV
    return out


def one_backward(pipeline, camera, *, sh_degree: int, hardness: float,
                 stbn: StbnVariables | None) -> dict[str, np.ndarray]:
    """Forward + loss + backward for ONE mask, then read the gradients.

    A full ``clear_gradients`` precedes it, because the training loop only clears
    the 2D seeds per iteration and relies on the Adam step to consume the scalar
    gradients — and this probe deliberately never runs Adam. Without the explicit
    clear every reading would be a running sum of all previous masks.
    """
    contextual = IterationContextualVariables(
        hardness_value=hardness, sh_degree=sh_degree, contextual_stbn=stbn,
        densification_variables=DensificationProperties(
            structure_tensor_texture=None, structure_kwargs={}),
    )
    with pipeline.recording("grad_probe"):
        pipeline.clear_gradients()
        pipeline.record_forward(camera, contextual)
        pipeline.record_loss(contextual)
        pipeline.record_backward(contextual)
    return read_gradients(pipeline)


def compare(sparse: dict[str, np.ndarray],
            dense: dict[str, np.ndarray]) -> dict[str, dict]:
    """Cosine similarity and relative L2 error, per group and overall."""
    out: dict[str, dict] = {}
    groups = dict(PARAM_GROUPS)
    groups["all"] = tuple(_grad_components())
    for group, components in groups.items():
        present = [c for c in components if c in sparse and c in dense]
        if not present:
            continue
        a = np.concatenate([np.asarray(sparse[c]).ravel() for c in present])
        b = np.concatenate([np.asarray(dense[c]).ravel() for c in present])
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        cosine = (float(np.dot(a, b) / (norm_a * norm_b))
                  if norm_a > 0 and norm_b > 0 else None)
        rel_l2 = (float(np.linalg.norm(a - b) / norm_b) if norm_b > 0 else None)
        out[group] = {
            "cosine_similarity": cosine,
            "relative_l2_error": rel_l2,
            "sparse_norm": norm_a,
            "dense_norm": norm_b,
            # >1 means the sparse estimate is systematically too large. With
            # correct 1/rho reweighting this should sit near 1.
            "norm_ratio": (norm_a / norm_b) if norm_b > 0 else None,
            "n_values": int(a.size),
        }
    return out


def upload_tables(pipeline, stbn_seed: int) -> dict[str, object]:
    """One uploaded mask table per noise source, held for the whole probe.

    Both tables live on the device simultaneously and are bound by swapping
    ``pipeline._stbn`` between measurements. This is what lets ONE process
    measure all three regimes at ONE frozen state — the alternative (a process
    per noise source) would have each regime training its own trajectory to
    "the same" iteration and then compare gradients at genuinely different
    Gaussian configurations, which would not be a controlled comparison at all.
    """
    tables: dict[str, object] = {}
    cache = stbn_module.default_cache_dir() / "cache"
    for source in (stbn_module.SOURCE_WHITE, stbn_module.SOURCE_NVIDIA):
        spec = stbn_module.StbnMaskSpec(source=source, seed=int(stbn_seed))
        tables[source] = stbn_module.upload_mask(pipeline.device, spec, cache)
    return tables


def probe_state(app, loop, state_iter: int, *, k_values, fraction: float,
                stbn_seed: int, tables: dict[str, object]) -> dict:
    """Advance to ``state_iter`` and measure every regime at that frozen state.

    The advance is DENSE (``loop.stbn_density`` is cleared before this runs), so
    the state reached is a property of the scene and the seed alone — not of the
    sampling regime. Every regime is then probed against that one shared state,
    which is the only way the three are comparable.
    """
    context = loop.advance_to_capture(state_iter)
    pipeline = app._pipeline
    camera = context.camera_entry.camera
    sh_degree = context.iteration_contextual_variables.sh_degree
    hardness = context.iteration_contextual_variables.hardness_value

    pipeline.runtime.wait_for_idle()
    n_gaussians = int(pipeline.buffers.num_gaussians_counter
                      .to_numpy().view(np.uint32)[0])

    # Reference: the dense gradient at this exact state, mask off.
    dense = one_backward(pipeline, camera, sh_degree=sh_degree,
                         hardness=hardness, stbn=None)

    budget = stbn_module.pixels_per_tile(fraction, TILE_AREA)
    max_k = max(k_values)
    slices = pipeline.stbn_slices
    record: dict = {
        "state_iter": state_iter,
        "camera_index": context.camera_index,
        "n_gaussians": n_gaussians,
        "sh_degree": sh_degree,
        "pixel_fraction": fraction,
        "pixels_per_tile": budget,
        "tile_area": TILE_AREA,
        "mask_slices": slices,
        "dense_reference_norms": {
            group: values["dense_norm"]
            for group, values in compare(dense, dense).items()},
        "regimes": {},
    }

    for label, source, independent in SPARSE_REGIMES:
        table = tables.get(source)
        if table is None:
            continue
        # Bind this regime's noise source. Everything else — the state, the
        # camera, the per-tile budget, the reweighting, the backward path — is
        # held fixed, so the only difference between regimes is the mask.
        pipeline._stbn = table
        if independent:
            rng = np.random.default_rng(int(stbn_seed))
            indices = [int(rng.integers(slices)) for _ in range(max_k)]
        else:
            phase = (int(context.camera_index) * 2654435761
                     + int(stbn_seed) * 40503) % slices
            indices = [(phase + visit) % slices for visit in range(max_k)]

        per_mask: list[dict[str, np.ndarray]] = []
        for slice_index in indices:
            per_mask.append(one_backward(
                pipeline, camera, sh_degree=sh_degree, hardness=hardness,
                stbn=StbnVariables(slice_index=slice_index,
                                   pixels_per_tile=budget)))

        by_k: dict[str, dict] = {}
        for k in sorted(k_values):
            if k > len(per_mask):
                continue
            # Prefix MEAN: K unbiased 1/rho-reweighted estimates average to an
            # estimate of the dense gradient (a sum would be K times too big).
            averaged = {name: np.mean([m[name] for m in per_mask[:k]], axis=0)
                        for name in per_mask[0]}
            by_k[str(k)] = compare(averaged, dense)
        record["regimes"][label] = {
            "slice_indices": indices,
            "independent_draw": independent,
            "by_k": by_k,
        }
    return record


def main() -> int:
    parser = build_parser()
    parser.add_argument("--state-iters", type=int, nargs="*",
                        default=list(DEFAULT_STATE_ITERS),
                        help="training states at which to probe")
    parser.add_argument("--k-values", type=int, nargs="*",
                        default=list(DEFAULT_K_VALUES),
                        help="consecutive-mask counts to accumulate")
    parser.add_argument("--probe-fraction", type=float, default=0.25)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # Force the headless, observation-free configuration. This process must not
    # write a run report, evaluate, or dump renders: it is a diagnostic, and any
    # artifact it produced would look like a training cell to the suite.
    args.no_viewer = True
    args.profile_run = False
    args.capture_external = False
    args.preview_every = 0
    args.log_every = 0
    args.eval_every = 0
    args.eval_iters = None
    args.psnr_curve = False
    args.render_all_every = 0
    args.final_test_psnr = False
    args.compute_final_metrics = False
    args.run_report = False
    args.save_test_renders = False
    args.stage_timing_every = 0
    args.write_ground_truths = False
    args.iters = max(args.iters, max(args.state_iters))

    _validate_groups()

    app = TrainingApplication(args)
    app.prepare_training_context()
    loop = app.build_training_loop()

    # DENSE advance. The frozen state must not depend on the regime being
    # measured, so the trajectory that reaches it supervises every pixel.
    # Clearing the density schedule makes _stbn_variables return None, which
    # _stbn_kwargs resolves to the full per-tile budget — and a full budget is
    # bit-identical to running with no mask at all (tests/test_stbn.py).
    loop.stbn_density = None
    tables = upload_tables(app._pipeline, int(args.stbn_seed))

    scene_name = Path(args.scene).name
    print(f"[grad-probe] {scene_name}: states={args.state_iters} "
          f"K={args.k_values} rho={args.probe_fraction}; dense advance, "
          f"{len(tables)} mask tables bound per state", flush=True)

    states: list[dict] = []
    for state_iter in sorted(args.state_iters):
        record = probe_state(app, loop, state_iter, k_values=args.k_values,
                             fraction=args.probe_fraction,
                             stbn_seed=int(args.stbn_seed), tables=tables)
        states.append(record)
        for label, payload in record["regimes"].items():
            summary = payload["by_k"]
            first = summary.get(str(min(args.k_values)), {}).get("all", {})
            last = summary.get(str(max(args.k_values)), {}).get("all", {})
            print(f"[grad-probe] iter {state_iter:5d} {label:11s} "
                  f"cos K=1 {first.get('cosine_similarity')} -> "
                  f"K={max(args.k_values)} {last.get('cosine_similarity')}",
                  flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "scene": scene_name,
        "scene_path": str(args.scene),
        "seed": int(args.seed),
        "stbn_source": args.stbn_source,
        "stbn_seed": int(args.stbn_seed),
        "probe_fraction": float(args.probe_fraction),
        "k_values": sorted(args.k_values),
        "state_reached_with": ("dense supervision (regime-independent "
                              "trajectory); masks applied only during the probe"),
        "param_groups": {k: list(v) for k, v in PARAM_GROUPS.items()},
        "accumulation": ("prefix MEAN of K individually-read gradients; the "
                         "1/rho-reweighted sparse loss makes each an unbiased "
                         "estimate of the dense gradient"),
        "no_parameter_updates": True,
        "states": states,
    }, indent=2))
    print(f"[grad-probe] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
