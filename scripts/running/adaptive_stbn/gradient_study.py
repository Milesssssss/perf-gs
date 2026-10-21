# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Frozen-state gradient fidelity: uniform vs importance-guided sparse sampling.

MEASUREMENT ONLY. No optimizer step is ever taken, no parameter is ever written,
and this script is never invoked by a training cell. It advances to a real
mid-training state with DENSE supervision, freezes it, and then asks one question
repeatedly: with the Gaussians, the camera, the rendered prediction and the loss all
held identical, how close is the gradient produced by a sparse pixel mask to the
dense one?

Structured as a direct extension of ``scripts/running/paper_gradient_study.py`` —
same accumulation trick, same three-scalar metric machinery, same sanity checks —
so a number here is comparable with one there. What is added is the adaptive arm.

WHAT IS HELD FIXED, and why each item matters:

  the STATE          reached by dense supervision, so it is a property of the
                     scene and seed and not of the regime being measured.
  the CAMERA         one at a time; p_i depends on the view, so a method's
                     probability map must be built from the same h_i the dense
                     reference was built from.
  the BUDGET         every method spends the same EXPECTED rho, using the same
                     quantized rho the uniform table bakes in
                     (pixels_per_tile(rho)/64). Comparing methods at different
                     expected budgets would measure budget, not allocation.
  the FORWARD        re-run for every single mask. This is not redundant: the
                     backward does not leave the forward's raster checkpoints
                     reusable, so only the FIRST backward after a forward is
                     valid. The existing study documents what happens otherwise
                     (every sparse gradient came out with norm ratio ~1/rho).

NORMALIZATION. Both paths already carry their unbiasing weight — the uniform path
scales each supervised pixel by TILE_AREA/k = 1/rho, the adaptive path by 1/p_i —
so ONE masked backward is already an unbiased estimate of the dense gradient. K
masks therefore combine as a MEAN of K estimates, never a sum, and the norm ratio
of a single estimate must sit near 1 rather than near rho. ``sanity_checks``
verifies the metric machinery before any comparison is reported.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[2]
for _sub in ("running", "debug", "plots"):
    _p = str(_SCRIPTS / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

from paper_gradient_study import (SCALAR_GROUPS, SH_REST_GROUP,  # noqa: E402
                                  GroupStats, StagedGradReader,
                                  StagedShRestReader, _accum_rows,
                                  _validate_groups, compare_scalars,
                                  sanity_checks, state_fingerprint)

from trainer.backend.render import adaptive_stbn as ad_math      # noqa: E402
from trainer.backend.render import stbn as stbn_module           # noqa: E402
from trainer.backend.render.adaptive_sampler import AdaptiveSampler  # noqa: E402
from trainer.config.config import (DensificationProperties, IterationContextualVariables, StbnVariables)
from trainer.contracts.codegen import TILE_AREA                  # noqa: E402
from trainer.frontend.app import TrainingApplication             # noqa: E402
from trainer.frontend.cli import build_parser                    # noqa: E402

DEFAULT_STATE_ITERS = (500, 1500, 3000)
DEFAULT_K_VALUES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_CAMERAS = 4
DEFAULT_SINGLE_MASKS = 4
DEFAULT_SEQUENCES = 1

# The five arms of section 11, at one matched expected budget.
#
#   kind      "uniform" thresholds the pre-built table at a constant rho;
#             "adaptive" builds p_i from the frozen state's own screen gradient.
#   source    which noise the uniform table is ranked from (uniform arms only).
#   traversal how the slice index advances between consecutive masks —
#             "independent" mirrors --stbn-independent, "consecutive" mirrors the
#             trainer's per-camera visit walk.
#
# uniform_iid and uniform_stbn are the two ends of the published uniform ladder;
# including uniform_indep_blue as well would triple the runtime for a contrast the
# existing study already reports. Stated rather than silently dropped.
METHODS: tuple[dict, ...] = (
    {"name": "uniform_iid", "kind": "uniform", "source": stbn_module.SOURCE_WHITE,
     "traversal": "independent"},
    {"name": "uniform_stbn", "kind": "uniform", "source": stbn_module.SOURCE_NVIDIA,
     "traversal": "consecutive"},
    {"name": "adaptive_iid", "kind": "adaptive", "uniform": ad_math.UNIFORM_IID,
     "traversal": "independent"},
    {"name": "adaptive_indep_blue", "kind": "adaptive",
     "uniform": ad_math.UNIFORM_BLUE_INDEPENDENT, "traversal": "independent"},
    {"name": "adaptive_stbn", "kind": "adaptive",
     "uniform": ad_math.UNIFORM_BLUE_TEMPORAL, "traversal": "consecutive"},
)


# ---------------------------------------------------------------------------
# Method binding
# ---------------------------------------------------------------------------

class MethodBinder:
    """Swaps the pipeline between the uniform table and the adaptive samplers.

    All the resources are resident simultaneously and only the BINDING changes, so
    one process compares every method at ONE frozen state. That is what makes the
    comparison controlled — a per-method process would also mean a per-method
    device, shader compile and (worse) a separately-reached state.

    The uniform arms are bound by assigning ``pipeline._stbn`` (exactly what the
    existing study does) with ``pipeline._adaptive`` cleared; the adaptive arms by
    assigning ``pipeline._adaptive``, which routes the loss onto the dense pass
    plus the sampling pass. Nothing else about the pipeline moves.
    """

    def __init__(self, pipeline, *, fraction: float, alpha: float, sigma: float,
                 downsample: int, stbn_seed: int, width: int,
                 height: int) -> None:
        self.pipeline = pipeline
        self.fraction = float(fraction)
        cache = stbn_module.default_cache_dir() / "cache"
        base = stbn_module.StbnMaskSpec(fraction=self.fraction,
                                        select_mode=stbn_module.SELECT_GLOBAL,
                                        seed=int(stbn_seed))
        self.tables = {
            source: stbn_module.upload_mask(
                pipeline.device, dataclasses.replace(base, source=source), cache)
            for source in (stbn_module.SOURCE_WHITE, stbn_module.SOURCE_NVIDIA)
        }
        self.samplers: dict[str, AdaptiveSampler] = {}
        for uniform in (ad_math.UNIFORM_IID, ad_math.UNIFORM_BLUE_INDEPENDENT,
                        ad_math.UNIFORM_BLUE_TEMPORAL):
            source = (stbn_module.SOURCE_WHITE if uniform == ad_math.UNIFORM_IID
                      else stbn_module.SOURCE_NVIDIA)
            spec = ad_math.AdaptiveSpec(
                fraction=self.fraction, alpha=float(alpha), sigma=float(sigma),
                downsample=int(downsample), uniform=uniform, source=source,
                seed=int(stbn_seed))
            self.samplers[uniform] = AdaptiveSampler(
                device=pipeline.device, width=width, height=height, spec=spec,
                kernels=pipeline, cache_dir=cache)
        # The table the loss binds on the ADAPTIVE path. It is never read there
        # (the budget resolves to TILE_AREA, which takes stbnActive's early-out),
        # but a declared shader parameter still has to be bound.
        self._default_table = self.tables[stbn_module.SOURCE_NVIDIA]

    @property
    def slices(self) -> int:
        return self._default_table.slices

    def bind(self, method: dict) -> None:
        if method["kind"] == "uniform":
            self.pipeline._adaptive = None
            self.pipeline._stbn = self.tables[method["source"]]
            return
        self.pipeline._stbn = self._default_table
        self.pipeline._adaptive = self.samplers[method["uniform"]]

    def bind_dense(self) -> None:
        """Dense reference: no adaptive pass, and the mask off."""
        self.pipeline._adaptive = None
        self.pipeline._stbn = self._default_table

    def sampler_for(self, method: dict) -> AdaptiveSampler | None:
        if method["kind"] != "adaptive":
            return None
        return self.samplers[method["uniform"]]


def slice_sequence(traversal: str, *, slices: int, length: int, rng,
                   phase: int) -> list[int]:
    if traversal == "independent":
        return [int(rng.integers(slices)) for _ in range(length)]
    return [(phase + step) % slices for step in range(length)]


def _context(sh_degree: int, hardness: float,
             stbn: StbnVariables | None) -> IterationContextualVariables:
    return IterationContextualVariables(
        hardness_value=hardness, sh_degree=sh_degree, contextual_stbn=stbn,
        densification_variables=DensificationProperties(
            structure_tensor_texture=None, structure_kwargs={}),
    )


def loss_and_backward(pipeline, camera, *, sh_degree: int, hardness: float,
                      stbn: StbnVariables | None, clear: bool) -> None:
    """One masked forward + loss (+ adaptive pass) + backward on the frozen state.

    ``clear=False`` is the accumulation path: the int64 gradient buffers are atomic
    accumulators, so K rounds without a clear leave their SUM on the device. No
    regularization pass is run — its gradient is mask-independent, so including it
    would add an identical constant to every method and inflate every cosine
    toward 1.
    """
    contextual = _context(sh_degree, hardness, stbn)
    with pipeline.recording("adaptive_grad_backward"):
        if clear:
            pipeline.clear_gradients()
        pipeline.record_forward(camera, contextual)
        pipeline.record_loss(contextual)
        pipeline.record_backward(contextual)


# ---------------------------------------------------------------------------
# Probability-map diagnostics at a frozen state
# ---------------------------------------------------------------------------

def probability_diagnostics(binder: MethodBinder, pipeline, camera, *,
                            sh_degree: float, hardness: float, budget: int,
                            cam_idx: int, revisits: int = 4) -> dict:
    """What the adaptive map looks like at this (state, camera), plus how much it
    MOVES across repeat visits of the same camera.

    The temporal-variation number is the input to design section 15: a per-camera
    EMA of the importance field is only worth adding if the instantaneous field
    changes materially between visits to the SAME camera. With the state frozen it
    cannot change at all, so what this measures is the floor — the variation
    attributable to the sampler itself, which should be exactly zero. Any nonzero
    value here would mean the map depends on something it should not.
    """
    sampler = binder.samplers[ad_math.UNIFORM_BLUE_TEMPORAL]
    binder.pipeline._adaptive = sampler
    binder.pipeline._stbn = binder._default_table
    maps = []
    for visit in range(revisits):
        loss_and_backward(pipeline, camera, sh_degree=sh_degree,
                          hardness=hardness,
                          stbn=StbnVariables(slice_index=visit,
                                             pixels_per_tile=budget,
                                             sample_seed=visit + 1),
                          clear=True)
        pipeline.runtime.wait_for_idle()
        maps.append(sampler.read_probabilities())
    stacked = np.stack(maps)
    reference = stacked[0]
    drift = (np.abs(stacked[1:] - reference[None]).max()
             if stacked.shape[0] > 1 else 0.0)
    # Only THIS camera's revisits: the sampler accumulates across every call in the
    # process, so a mean over the whole history would silently pool cameras and
    # checkpoints into a number labelled with one camera.
    stats = sampler.drain_stats()[-revisits:]
    row = {
        "camera_id": cam_idx,
        "p_mean": float(reference.mean()),
        "p_min": float(reference.min()),
        "p_max": float(reference.max()),
        "p_p99": float(np.percentile(reference, 99)),
        "saturated_fraction": float((reference >= 1.0 - 1e-6).mean()),
        "dynamic_range": float(reference.max() / max(reference.min(), 1e-12)),
        "revisit_max_abs_change": float(drift),
        "revisits": int(revisits),
        "measured_p_mean": (float(np.mean([s.p_mean for s in stats]))
                            if stats else None),
        "measured_selected_fraction": (
            float(np.mean([s.selected_fraction for s in stats]))
            if stats else None),
        "measured_budget_error_max": (
            float(np.max([abs(s.p_mean - sampler.spec.budget_fraction())
                          for s in stats])) if stats else None),
    }
    return row


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------

def study_camera(pipeline, binder: MethodBinder, camera, cam_idx: int, *,
                 sh_degree: int, hardness: float, reader: StagedGradReader,
                 sh_reader: StagedShRestReader, fraction: float, budget: int,
                 k_values, single_masks: int, sequences: int, stbn_seed: int,
                 first_camera: bool,
                 measure_sh_rest: bool) -> tuple[list[dict], list[dict],
                                                 list[dict]]:
    """All measurements for one frozen (state, camera)."""
    slices = binder.slices

    binder.bind_dense()
    loss_and_backward(pipeline, camera, sh_degree=sh_degree, hardness=hardness,
                      stbn=None, clear=True)
    dense = reader.read_all()
    dense_rest = sh_reader.read() if measure_sh_rest else None
    sanity = sanity_checks(dense, fraction) if first_camera else []

    single_rows: list[dict] = []
    accum_rows: list[dict] = []

    for method in METHODS:
        name = method["name"]
        binder.bind(method)
        phase = (int(cam_idx) * 2654435761 + int(stbn_seed) * 40503) % slices

        # ---- Pass 1: single masks (K=1), every group INCLUDING SH-rest -----
        rng = np.random.default_rng((stbn_seed, cam_idx, abs(hash(name)) % 10_000))
        indices = slice_sequence(method["traversal"], slices=slices,
                                 length=single_masks, rng=rng, phase=phase)
        for realization, slice_index in enumerate(indices):
            variables = StbnVariables(slice_index=slice_index,
                                      pixels_per_tile=budget,
                                      sample_seed=int(rng.integers(1 << 32)))
            loss_and_backward(pipeline, camera, sh_degree=sh_degree,
                              hardness=hardness, stbn=variables, clear=True)
            sparse = reader.read_all()
            stats = compare_scalars(sparse, dense)
            for group in stats.groups():
                single_rows.append({
                    "sampling_method": name, "camera_id": cam_idx,
                    "pixel_ratio": fraction, "sample": realization,
                    "slice_index": slice_index, "parameter_group": group,
                    **stats.metrics(group)})
            rest = sh_reader.read() if measure_sh_rest else None
            if rest is not None and dense_rest is not None:
                rest_stats = GroupStats()
                rest_stats.add(SH_REST_GROUP, rest.reshape(-1),
                               dense_rest.reshape(-1))
                single_rows.append({
                    "sampling_method": name, "camera_id": cam_idx,
                    "pixel_ratio": fraction, "sample": realization,
                    "slice_index": slice_index,
                    "parameter_group": SH_REST_GROUP,
                    **rest_stats.metrics(SH_REST_GROUP)})

        # ---- Pass 2: accumulation, K = 1..max, scalar groups only ----------
        # features_rest_grad is OVERWRITTEN per backward (its shader assigns rather
        # than accumulates), so it cannot ride the no-clear accumulation. Excluded
        # here and recorded as such in the payload.
        for sequence in range(sequences):
            rng = np.random.default_rng(
                (stbn_seed, cam_idx, abs(hash(name)) % 10_000, 1000 + sequence))
            walk = slice_sequence(method["traversal"], slices=slices,
                                  length=max(k_values), rng=rng,
                                  phase=(phase + sequence) % slices)
            loss_and_backward(
                pipeline, camera, sh_degree=sh_degree, hardness=hardness,
                stbn=StbnVariables(slice_index=walk[0], pixels_per_tile=budget,
                                   sample_seed=int(rng.integers(1 << 32))),
                clear=True)
            for step, slice_index in enumerate(walk[1:], start=2):
                if (step - 1) in k_values:
                    accum_rows += _accum_rows(reader, dense, step - 1, name,
                                              cam_idx, fraction, sequence)
                loss_and_backward(
                    pipeline, camera, sh_degree=sh_degree, hardness=hardness,
                    stbn=StbnVariables(slice_index=slice_index,
                                       pixels_per_tile=budget,
                                       sample_seed=int(rng.integers(1 << 32))),
                    clear=False)
            if max(k_values) in k_values:
                accum_rows += _accum_rows(reader, dense, max(k_values), name,
                                          cam_idx, fraction, sequence)
    return single_rows, accum_rows, sanity


def main() -> int:
    parser = build_parser()
    parser.add_argument("--state-iters", type=int, nargs="*",
                        default=list(DEFAULT_STATE_ITERS))
    parser.add_argument("--k-values", type=int, nargs="*",
                        default=list(DEFAULT_K_VALUES))
    parser.add_argument("--cameras", type=int, default=DEFAULT_CAMERAS)
    parser.add_argument("--single-masks", type=int, default=DEFAULT_SINGLE_MASKS)
    parser.add_argument("--sequences", type=int, default=DEFAULT_SEQUENCES)
    parser.add_argument("--measure-sh-rest", action="store_true", default=False,
                        help="Also measure the SH bands above DC. Off by default: "
                             "not a reported group, cannot ride the "
                             "K-accumulation path (its buffer is overwritten per "
                             "backward), and ~130 MB per readback at a late "
                             "checkpoint — which dominated the wall clock.")
    parser.add_argument("--probe-fraction", type=float, default=0.25)
    parser.add_argument("--probe-alpha", type=float, default=0.75)
    parser.add_argument("--probe-sigma", type=float, default=8.0)
    parser.add_argument("--probe-importance-downsample", type=int, default=4)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # Diagnostic-only configuration: this process must not write a run report,
    # evaluate, dump renders or take a capture. Any artifact it produced would look
    # like a training cell to the suite.
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
    # The adaptive path is bound by hand below, so the trainer must NOT create its
    # own sampler (which would fix one uniform mode for the whole process). The
    # STBN_ENABLED define still needs the sparse fraction, which the caller passes.
    args.stbn_adaptive = False

    _validate_groups()
    app = TrainingApplication(args)
    app.prepare_training_context()
    loop = app.build_training_loop()

    # DENSE advance: the frozen state must be a property of the scene and seed, not
    # of the regime under measurement.
    loop.stbn_density = None
    pipeline = app._pipeline
    scene = app._sceneContext
    binder = MethodBinder(
        pipeline, fraction=args.probe_fraction, alpha=args.probe_alpha,
        sigma=args.probe_sigma, downsample=args.probe_importance_downsample,
        stbn_seed=int(args.stbn_seed), width=scene.width, height=scene.height)
    budget = stbn_module.pixels_per_tile(args.probe_fraction, TILE_AREA)
    enforced = budget / TILE_AREA

    scene_name = Path(args.scene).name
    train_indices = list(scene.train_indices)
    step = max(1, len(train_indices) // max(args.cameras, 1))
    cameras = train_indices[::step][:args.cameras]
    print(f"[adaptive-grad] {scene_name}: states={args.state_iters} "
          f"K<={max(args.k_values)} rho_requested={args.probe_fraction} "
          f"rho_enforced={enforced:g} budget={budget}/{TILE_AREA} "
          f"alpha={args.probe_alpha} sigma={args.probe_sigma} "
          f"cameras={cameras}", flush=True)

    single_rows: list[dict] = []
    accum_rows: list[dict] = []
    sanity_rows: list[dict] = []
    probability_rows: list[dict] = []
    fingerprints: list[dict] = []

    for state_iter in sorted(args.state_iters):
        context = loop.advance_to_capture(state_iter)
        sh_degree = context.iteration_contextual_variables.sh_degree
        hardness = context.iteration_contextual_variables.hardness_value
        before = state_fingerprint(pipeline)
        count = int(before.get("n_gaussians") or 0)
        reader = StagedGradReader(pipeline, count)
        sh_reader = StagedShRestReader(pipeline, count)
        started = time.perf_counter()

        for order, cam_idx in enumerate(cameras):
            pipeline.set_target_texture(app.target_textures[cam_idx])
            camera = scene.colmapScene.images[cam_idx].camera

            probability = probability_diagnostics(
                binder, pipeline, camera, sh_degree=sh_degree,
                hardness=hardness, budget=budget, cam_idx=cam_idx)
            probability.update(scene=scene_name, checkpoint=state_iter,
                               n_gaussians=count)
            probability_rows.append(probability)

            s_rows, a_rows, checks = study_camera(
                pipeline, binder, camera, cam_idx, sh_degree=sh_degree,
                hardness=hardness, reader=reader, sh_reader=sh_reader,
                fraction=enforced, budget=budget,
                k_values=tuple(sorted(args.k_values)),
                single_masks=args.single_masks, sequences=args.sequences,
                stbn_seed=int(args.stbn_seed),
                first_camera=(order == 0 and state_iter == min(args.state_iters)),
                measure_sh_rest=bool(args.measure_sh_rest))
            for row in s_rows + a_rows:
                row.update(scene=scene_name, checkpoint=state_iter,
                           n_gaussians=count)
            single_rows += s_rows
            accum_rows += a_rows
            if checks:
                for row in checks:
                    row.update(scene=scene_name, checkpoint=state_iter,
                               camera_id=cam_idx)
                sanity_rows += checks
                print("[adaptive-grad] sanity checks:", flush=True)
                for row in checks:
                    print(f"    {row['check']:24s} passed={row.get('passed')}",
                          flush=True)

        after = state_fingerprint(pipeline)
        unchanged = (
            before.get("gaussians_checksum") == after.get("gaussians_checksum")
            and before.get("adam_checksum") == after.get("adam_checksum")
            and before.get("n_gaussians") == after.get("n_gaussians"))
        fingerprints.append({"checkpoint": state_iter, "before": before,
                            "after": after, "state_unchanged": bool(unchanged)})
        print(f"[adaptive-grad] iter {state_iter}: {len(cameras)} cameras, "
              f"n_g={count}, {time.perf_counter() - started:.0f}s, "
              f"state_unchanged={unchanged}", flush=True)
        if not unchanged:
            print("[adaptive-grad] WARNING: the frozen state CHANGED during "
                  "analysis; results for this checkpoint are not controlled.",
                  flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "scene": scene_name,
        "scene_path": str(args.scene),
        "seed": int(args.seed),
        "stbn_seed": int(args.stbn_seed),
        "probe_fraction_requested": float(args.probe_fraction),
        "probe_fraction_enforced": float(enforced),
        "pixels_per_tile": budget,
        "tile_area": TILE_AREA,
        "alpha": float(args.probe_alpha),
        "sigma": float(args.probe_sigma),
        "importance_downsample": int(args.probe_importance_downsample),
        "select_mode": stbn_module.SELECT_GLOBAL,
        "k_values": sorted(args.k_values),
        "cameras": [int(c) for c in cameras],
        "state_reached_with": "dense supervision (regime-independent trajectory)",
        "no_parameter_updates": True,
        "regularization_included": False,
        "scalar_groups": {k: list(v) for k, v in SCALAR_GROUPS.items()},
        "sh_rest_group": SH_REST_GROUP,
        "sh_rest_measured": bool(args.measure_sh_rest),
        "sh_rest_limitation": (
            "features_rest_grad is overwritten per backward, so SH-rest could only "
            "ever be measured in the single-mask pass, never in the "
            "K-accumulation. It is OFF by default here: it is not one of the "
            "reported groups (SH0 is the DC term and lives in the scalar "
            "buffers), and reading it costs ~130 MB per measurement at a late "
            "checkpoint, which dominated the study's wall clock for a number that "
            "would then be excluded from every headline."),
        "budget_matching": (
            "every method spends the same EXPECTED fraction "
            f"{enforced:g} = pixels_per_tile({args.probe_fraction})/{TILE_AREA}, "
            "which is the density the uniform table actually realizes. Solving "
            "the adaptive constraint for the requested value instead would give "
            "the adaptive arm more pixels than its control."),
        "normalization": (
            "the uniform path weights each supervised pixel by TILE_AREA/k = "
            "1/rho and the adaptive path by 1/p_i, so a single masked backward is "
            "already unbiased for the dense gradient; K masks combine as a MEAN"),
        "methods": [dict(m) for m in METHODS],
        "methods_omitted": (
            "uniform_indep_blue is not measured here: the existing gradient study "
            "(outputs/paper_stbn) reports the uniform spatial contrast, and adding "
            "a sixth arm would cost ~20% more GPU time for a number already "
            "published."),
        "sanity_checks": sanity_rows,
        "state_fingerprints": fingerprints,
        "probability_maps": probability_rows,
        "single_mask": single_rows,
        "accumulation": accum_rows,
    }, indent=2))
    print(f"[adaptive-grad] wrote {out_path} ({len(single_rows)} single-mask + "
          f"{len(accum_rows)} accumulation observations)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
