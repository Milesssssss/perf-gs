# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Controlled gradient-fidelity study: how faithfully does a sparse backward reproduce the dense 3DGS gradient?

Measurement only: no optimizer step; every method sees the same frozen, densely-reached state and prediction.
Usage: python scripts/running/paper_gradient_study.py --scene datasets/360_v2/bicycle --state-iters 500 1500 3000 --out outputs/paper_stbn/gradient_analysis/study_bicycle.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent
for _sub in ("running", "debug", "plots"):
    _p = str(_SCRIPTS / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

from trainer.backend.render import stbn_table
from trainer.backend.render.stbn_mask import StbnMask
from trainer.config.config import (DensificationProperties, IterationContextualVariables, StbnVariables)
from trainer.backend.compute.execution import TILE_AREA
from trainer.frontend.app import TrainingApplication
from trainer.frontend.cli import build_parser

# Names must exactly partition GRAD_SCALAR_COMPONENTS; _validate_groups checks this at startup.
SCALAR_GROUPS: dict[str, tuple[str, ...]] = {
    "position": ("mean_x", "mean_y", "mean_z"),
    "scale": ("log_scale_x", "log_scale_y", "log_scale_z"),
    "rotation": ("rot_x", "rot_y", "rot_z", "rot_w"),
    "opacity": ("opacity",),
    "color_sh0": ("color_r", "color_g", "color_b"),
}
# Measured in the single-mask pass only: its buffer is overwritten per backward.
SH_REST_GROUP = "color_sh_rest"

DEFAULT_STATE_ITERS = (500, 1500, 3000)
DEFAULT_K_VALUES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_CAMERAS = 8
DEFAULT_SINGLE_MASKS = 4          # per (camera, method), for K=1 spread
DEFAULT_SEQUENCES = 2             # independent K-sequences per (camera, method)

# (name, noise source, slice traversal); temporal_only is diagnostic-only, completing the spatial x temporal 2x2.
METHODS: tuple[tuple[str, str, str], ...] = (
    ("iid_white", "white", "independent"),
    ("spatial_blue", "nvidia", "independent"),
    ("temporal_only", "temporal", "consecutive"),
    ("stbn", "nvidia", "consecutive"),
)


def _grad_components() -> tuple[str, ...]:
    from trainer.backend.memory.buffers import GRAD_SCALAR_COMPONENTS
    return tuple(GRAD_SCALAR_COMPONENTS)


def _validate_groups() -> None:
    actual = set(_grad_components())
    mapped = {c for group in SCALAR_GROUPS.values() for c in group}
    missing, extra = actual - mapped, mapped - actual
    if missing or extra:
        raise RuntimeError(
            "SCALAR_GROUPS does not partition GRAD_SCALAR_COMPONENTS: "
            f"unmapped={sorted(missing)} unknown={sorted(extra)}")


class StagedGradReader:
    """Reads only the live prefix of each gradient buffer via a right-sized device staging buffer.

    ``Buffer.to_numpy()`` has no range argument, so a direct read transfers the whole ``--max-splats`` allocation.
    """

    def __init__(self, pipeline, count: int) -> None:
        import slangpy as spy

        self.pipeline = pipeline
        self.count = int(count)
        self.nbytes = self.count * 8
        self.staging = spy.NDBuffer(
            device=pipeline.device, dtype="int64_t", shape=(max(self.count, 1),),
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access)

    def read(self, name: str) -> np.ndarray:
        source = self.pipeline.buffers.gradients.params[name]
        with self.pipeline.commands.recording("grad_study_stage"):
            self.pipeline.commands.batch.encoder.copy_buffer(
                self.staging.storage, 0, source.storage, 0, self.nbytes)
        self.pipeline.runtime.wait_for_idle()
        raw = self.staging.to_numpy()
        return np.frombuffer(memoryview(raw), dtype=np.int64)[:self.count]

    def read_all(self) -> dict[str, np.ndarray]:
        self.pipeline.runtime.wait_for_idle()
        # copy() because the staging buffer is reused for the next component.
        return {name: self.read(name).copy() for name in _grad_components()}


class StagedShRestReader:
    """The same live-prefix staging for the packed half2 SH-rest gradients; ``read`` returns None without SH-rest bands."""

    def __init__(self, pipeline, count: int) -> None:
        import slangpy as spy

        self.pipeline = pipeline
        self.count = int(count)
        self.available = False
        sh = pipeline.buffers.sh
        if sh is None:
            return
        self.rest_floats = int(sh.rest_floats)
        self.feat_uints = int(sh.feat_uints)
        self.source = sh.features_rest_grad
        if self.rest_floats <= 0 or self.count <= 0:
            return
        self.uints = self.count * self.feat_uints
        self.nbytes = self.uints * 4
        self.staging = spy.NDBuffer(
            device=pipeline.device, dtype="uint", shape=(max(self.uints, 1),),
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access)
        self.available = True

    def read(self) -> np.ndarray | None:
        if not self.available:
            return None
        with self.pipeline.commands.recording("grad_study_stage_sh"):
            self.pipeline.commands.batch.encoder.copy_buffer(
                self.staging.storage, 0, self.source.storage, 0, self.nbytes)
        self.pipeline.runtime.wait_for_idle()
        raw = self.staging.to_numpy()
        halves = np.frombuffer(memoryview(raw), dtype=np.float16)
        halves = halves[:self.uints * 2].reshape(self.count, self.feat_uints * 2)
        return halves[:, :self.rest_floats].astype(np.float32)


def read_sh_rest_grad(pipeline, count: int) -> np.ndarray | None:
    """``(count, rest_floats)`` float32 view of the packed half2 SH-rest grads."""
    sh = pipeline.buffers.sh
    if sh is None or sh.rest_floats <= 0:
        return None
    rest_floats, feat_uints = int(sh.rest_floats), int(sh.feat_uints)
    raw = sh.features_rest_grad.to_numpy()
    halves = np.frombuffer(memoryview(raw), dtype=np.float16)
    halves = halves[:count * feat_uints * 2].reshape(count, feat_uints * 2)
    return halves[:, :rest_floats].astype(np.float32)


class GroupStats:
    """Per-group float64 ``dot(a,b)``, ``|a|^2``, ``|b|^2``: enough for every metric at any accumulation scale."""

    def __init__(self) -> None:
        self.dot: dict[str, float] = {}
        self.aa: dict[str, float] = {}
        self.bb: dict[str, float] = {}

    def add(self, group: str, a: np.ndarray, b: np.ndarray) -> None:
        av = a.astype(np.float64, copy=False)
        bv = b.astype(np.float64, copy=False)
        self.dot[group] = self.dot.get(group, 0.0) + float(np.dot(av, bv))
        self.aa[group] = self.aa.get(group, 0.0) + float(np.dot(av, av))
        self.bb[group] = self.bb.get(group, 0.0) + float(np.dot(bv, bv))

    def groups(self) -> list[str]:
        return list(self.dot)

    def metrics(self, group: str, scale: float = 1.0) -> dict:
        """Metrics for ``a/scale`` against ``b``; ``scale`` is K, turning the device's sum of K estimates into their mean."""
        dot, aa, bb = self.dot[group], self.aa[group], self.bb[group]
        norm_a = math.sqrt(max(aa, 0.0)) / scale
        norm_b = math.sqrt(max(bb, 0.0))
        # ||a/s - b||^2 = |a|^2/s^2 - 2 dot/s + |b|^2, clamped against round-off.
        diff_sq = max(aa / (scale * scale) - 2.0 * dot / scale + bb, 0.0)
        return {
            "relative_gradient_error": (math.sqrt(diff_sq) / norm_b
                                        if norm_b > 0 else None),
            "cosine_similarity": (dot / (math.sqrt(aa) * norm_b)
                                  if aa > 0 and norm_b > 0 else None),
            "gradient_norm_ratio": (norm_a / norm_b if norm_b > 0 else None),
            "dense_norm": norm_b,
            "sparse_norm": norm_a,
        }


def compare_scalars(sparse: dict[str, np.ndarray],
                    dense: dict[str, np.ndarray]) -> GroupStats:
    """Per-group stats over the 14 scalar gradients; the global entry sums the groups (the scalars are additive)."""
    stats = GroupStats()
    for group, components in SCALAR_GROUPS.items():
        for name in components:
            stats.add(group, sparse[name], dense[name])
    stats.dot["global_scalar"] = sum(stats.dot[g] for g in SCALAR_GROUPS)
    stats.aa["global_scalar"] = sum(stats.aa[g] for g in SCALAR_GROUPS)
    stats.bb["global_scalar"] = sum(stats.bb[g] for g in SCALAR_GROUPS)
    return stats


def build_temporal_only_volume(spec: stbn_table.StbnSpec,
                               seed: int) -> np.ndarray:
    """Diagnostic-only mask: perfectly stratified in time, white in space (the 2x2's missing cell)."""
    period = max(1, int(round(1.0 / max(spec.fraction, 1e-9))))
    rng = np.random.default_rng(seed)
    phase = rng.integers(0, period, size=(spec.height, spec.width))
    slices = np.arange(spec.slices)[:, None, None] % period
    return (slices == phase[None, :, :])


def upload_method_tables(pipeline, spec_fraction: float,
                         stbn_seed: int) -> dict[str, object]:
    """One table per noise source, all resident so every method is compared at one frozen state.

    ``spatial_blue`` and ``stbn`` share the NVIDIA table; only their slice traversal differs.
    """
    import dataclasses

    cache = stbn_table.default_cache_dir() / "cache"
    tables: dict[str, object] = {}
    base = stbn_table.StbnSpec(
        fraction=float(spec_fraction),
        select_mode=stbn_table.SELECT_GLOBAL)

    for source in (stbn_table.SOURCE_WHITE, stbn_table.SOURCE_NVIDIA):
        spec = dataclasses.replace(base, source=source, seed=int(stbn_seed))
        tables[source] = StbnMask.build(pipeline.device, spec, cache)

    temporal_spec = dataclasses.replace(base, source=stbn_table.SOURCE_WHITE,
                                        seed=int(stbn_seed))
    packed = stbn_table.pack_selection(
        build_temporal_only_volume(temporal_spec, int(stbn_seed)), temporal_spec)
    tables["temporal"] = StbnMask.from_table(pipeline.device, packed, temporal_spec)
    return tables


def slice_sequence(traversal: str, *, slices: int, length: int, rng,
                   phase: int) -> list[int]:
    """Slice indices in visit order: fresh draws (``independent``) or the per-camera walk (``consecutive``)."""
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
    """One masked forward + loss + backward on the frozen state.

    The forward is re-run each time: only the first backward after a forward is valid (the frozen re-render is deterministic).
    ``clear=False`` accumulates the sum in the atomic int64 buffers; regularization is skipped as mask-independent.
    """
    contextual = _context(sh_degree, hardness, stbn)
    with pipeline.commands.recording("grad_study_backward"):
        if clear:
            pipeline.optimizer.zero_grad()
        pipeline.record_forward(camera, contextual)
        pipeline.record_loss(contextual)
        pipeline.record_backward(contextual)


def state_fingerprint(pipeline) -> dict:
    """A cheap checksum of everything that must not change during the study."""
    pipeline.runtime.wait_for_idle()
    out: dict = {}
    try:
        count = int(pipeline.buffers.primitive.active_count
                    .to_numpy().view(np.uint32)[0])
        out["n_gaussians"] = count
        values = pipeline.buffers.primitive.values.to_numpy()
        view = np.frombuffer(memoryview(values), dtype=np.uint32)
        out["gaussians_checksum"] = int(view.sum(dtype=np.uint64))
        adam = pipeline.buffers.adam.state.to_numpy()
        out["adam_checksum"] = int(
            np.frombuffer(memoryview(adam), dtype=np.uint32).sum(dtype=np.uint64))
    except Exception as error:                             # noqa: BLE001
        out["error"] = f"{type(error).__name__}: {error}"
    return out


def sanity_checks(dense: dict[str, np.ndarray], fraction: float) -> list[dict]:
    """Verify the metric machinery and the sparse normalization before use."""
    checks: list[dict] = []

    identical = compare_scalars(dense, dense)
    m = identical.metrics("global_scalar")
    checks.append({
        "check": "dense_vs_dense",
        "expect": "relative_error ~ 0, cosine ~ 1, norm_ratio ~ 1",
        "relative_gradient_error": m["relative_gradient_error"],
        "cosine_similarity": m["cosine_similarity"],
        "gradient_norm_ratio": m["gradient_norm_ratio"],
        "passed": bool(abs(m["relative_gradient_error"]) < 1e-9
                       and abs(m["cosine_similarity"] - 1.0) < 1e-9),
    })

    doubled = {k: (v.astype(np.int64) * 2) for k, v in dense.items()}
    m = compare_scalars(doubled, dense).metrics("global_scalar")
    checks.append({
        "check": "dense_x2",
        "expect": "cosine ~ 1, norm_ratio ~ 2, relative_error ~ 1",
        "relative_gradient_error": m["relative_gradient_error"],
        "cosine_similarity": m["cosine_similarity"],
        "gradient_norm_ratio": m["gradient_norm_ratio"],
        "passed": bool(abs(m["cosine_similarity"] - 1.0) < 1e-9
                       and abs(m["gradient_norm_ratio"] - 2.0) < 1e-6
                       and abs(m["relative_gradient_error"] - 1.0) < 1e-6),
    })

    negated = {k: (-v.astype(np.int64)) for k, v in dense.items()}
    m = compare_scalars(negated, dense).metrics("global_scalar")
    checks.append({
        "check": "dense_negated",
        "expect": "cosine ~ -1 (guards the sign convention)",
        "cosine_similarity": m["cosine_similarity"],
        "passed": bool(abs(m["cosine_similarity"] + 1.0) < 1e-9),
    })

    checks.append({
        "check": "normalization_reference",
        "expect": (f"a correctly 1/rho-reweighted sparse gradient has norm_ratio "
                   f"NEAR 1, not near rho={fraction:g}. A ratio near {fraction:g} "
                   f"would mean the mask zeroed pixels without reweighting."),
        "rho": fraction,
        "note": "evaluated against the measured sparse runs below",
        "passed": None,
    })
    return checks


def study_camera(pipeline, camera, cam_idx: int, *, sh_degree: int,
                 hardness: float, count: int, reader: "StagedGradReader",
                 sh_reader: "StagedShRestReader",
                 tables: dict, fraction: float,
                 budget: int, k_values, single_masks: int, sequences: int,
                 stbn_seed: int, first_camera: bool) -> tuple[list[dict], list[dict], list[dict]]:
    """All measurements for one frozen (state, camera): (single-mask, accumulation, sanity) rows."""
    slices = pipeline.stbn_slices

    loss_and_backward(pipeline, camera, sh_degree=sh_degree, hardness=hardness,
                      stbn=None, clear=True)
    dense = reader.read_all()
    dense_rest = sh_reader.read()

    sanity = sanity_checks(dense, fraction) if first_camera else []

    single_rows: list[dict] = []
    accum_rows: list[dict] = []

    for method, source, traversal in METHODS:
        table = tables.get(source)
        if table is None:
            continue
        pipeline.stbn = table
        phase = (int(cam_idx) * 2654435761 + int(stbn_seed) * 40503) % slices

        # Pass 1: single masks (K=1), every group including SH-rest.
        rng = np.random.default_rng((stbn_seed, cam_idx, hash(method) % 10_000))
        indices = slice_sequence(traversal, slices=slices, length=single_masks,
                                 rng=rng, phase=phase)
        for realization, slice_index in enumerate(indices):
            variables = StbnVariables(slice_index=slice_index,
                                      pixels_per_tile=budget)
            loss_and_backward(pipeline, camera, sh_degree=sh_degree,
                              hardness=hardness, stbn=variables, clear=True)
            sparse = reader.read_all()
            stats = compare_scalars(sparse, dense)
            for group in stats.groups():
                single_rows.append({
                    "sampling_method": method, "camera_id": cam_idx,
                    "pixel_ratio": fraction, "sample": realization,
                    "slice_index": slice_index, "parameter_group": group,
                    **stats.metrics(group)})
            rest = sh_reader.read()
            if rest is not None and dense_rest is not None:
                rest_stats = GroupStats()
                rest_stats.add(SH_REST_GROUP, rest.reshape(-1),
                               dense_rest.reshape(-1))
                single_rows.append({
                    "sampling_method": method, "camera_id": cam_idx,
                    "pixel_ratio": fraction, "sample": realization,
                    "slice_index": slice_index,
                    "parameter_group": SH_REST_GROUP,
                    **rest_stats.metrics(SH_REST_GROUP)})

        # Pass 2: accumulation over K, scalar groups only (features_rest_grad is overwritten per backward).
        for sequence in range(sequences):
            rng = np.random.default_rng(
                (stbn_seed, cam_idx, hash(method) % 10_000, 1000 + sequence))
            walk = slice_sequence(traversal, slices=slices,
                                  length=max(k_values), rng=rng,
                                  phase=(phase + sequence) % slices)
            loss_and_backward(pipeline, camera, sh_degree=sh_degree,
                              hardness=hardness,
                              stbn=StbnVariables(slice_index=walk[0],
                                                 pixels_per_tile=budget),
                              clear=True)
            for step, slice_index in enumerate(walk[1:], start=2):
                if (step - 1) in k_values:
                    accum_rows += _accum_rows(
                        reader, dense, step - 1, method, cam_idx,
                        fraction, sequence)
                loss_and_backward(pipeline, camera, sh_degree=sh_degree,
                                  hardness=hardness,
                                  stbn=StbnVariables(slice_index=slice_index,
                                                     pixels_per_tile=budget),
                                  clear=False)
            if max(k_values) in k_values:
                accum_rows += _accum_rows(
                    reader, dense, max(k_values), method, cam_idx,
                    fraction, sequence)
    return single_rows, accum_rows, sanity


def _accum_rows(reader, dense, k, method, cam_idx, fraction,
                sequence) -> list[dict]:
    """Metrics for the running sum of k estimates, reported as their mean."""
    summed = reader.read_all()
    stats = compare_scalars(summed, dense)
    rows = []
    for group in stats.groups():
        metrics = stats.metrics(group, scale=float(k))
        rows.append({
            "sampling_method": method, "camera_id": cam_idx,
            "pixel_ratio": fraction, "sequence": sequence,
            "K": k, "parameter_group": group,
            "cumulative_relative_gradient_error":
                metrics["relative_gradient_error"],
            "cumulative_cosine_similarity": metrics["cosine_similarity"],
            "gradient_norm_ratio": metrics["gradient_norm_ratio"],
        })
    return rows


def main() -> int:
    parser = build_parser()
    parser.add_argument("--state-iters", type=int, nargs="*",
                        default=list(DEFAULT_STATE_ITERS))
    parser.add_argument("--k-values", type=int, nargs="*",
                        default=list(DEFAULT_K_VALUES))
    parser.add_argument("--cameras", type=int, default=DEFAULT_CAMERAS)
    parser.add_argument("--single-masks", type=int, default=DEFAULT_SINGLE_MASKS)
    parser.add_argument("--sequences", type=int, default=DEFAULT_SEQUENCES)
    parser.add_argument("--probe-fraction", type=float, default=0.25)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # No reports, evals, renders or captures: any artifact would look like a training cell to the suite.
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

    # Dense advance, so the frozen state depends on scene and seed, not on the measured regime.
    loop.stbn_sampler = None
    pipeline = app._pipeline
    tables = upload_method_tables(pipeline, args.probe_fraction,
                                 int(args.stbn_seed))
    budget = stbn_table.pixels_per_tile(args.probe_fraction, TILE_AREA)

    scene_name = Path(args.scene).name
    train_indices = list(app._sceneContext.train_indices)
    # Spread over the train set: adjacent cameras see nearly the same geometry.
    step = max(1, len(train_indices) // max(args.cameras, 1))
    cameras = train_indices[::step][:args.cameras]
    print(f"[grad-study] {scene_name}: states={args.state_iters} "
          f"K<={max(args.k_values)} rho={args.probe_fraction} "
          f"budget={budget}/{TILE_AREA} cameras={cameras}", flush=True)

    single_rows: list[dict] = []
    accum_rows: list[dict] = []
    sanity_rows: list[dict] = []
    fingerprints: list[dict] = []

    for state_iter in sorted(args.state_iters):
        context = loop.advance_to_capture(state_iter)
        sh_degree = context.iteration_contextual_variables.sh_degree
        hardness = context.iteration_contextual_variables.hardness_value
        before = state_fingerprint(pipeline)
        count = int(before.get("n_gaussians") or 0)
        # Sized to this checkpoint's live count, which is stable since densification never runs here.
        reader = StagedGradReader(pipeline, count)
        sh_reader = StagedShRestReader(pipeline, count)
        started = time.perf_counter()

        for order, cam_idx in enumerate(cameras):
            pipeline.targets.select(app.target_textures[cam_idx])
            camera = app._sceneContext.colmapScene.images[cam_idx].camera
            s_rows, a_rows, checks = study_camera(
                pipeline, camera, cam_idx, sh_degree=sh_degree,
                hardness=hardness, count=count, reader=reader,
                sh_reader=sh_reader, tables=tables,
                fraction=float(args.probe_fraction), budget=budget,
                k_values=tuple(sorted(args.k_values)),
                single_masks=args.single_masks, sequences=args.sequences,
                stbn_seed=int(args.stbn_seed),
                first_camera=(order == 0 and state_iter == min(args.state_iters)))
            for row in s_rows:
                row.update(scene=scene_name, checkpoint=state_iter,
                           n_gaussians=count)
            for row in a_rows:
                row.update(scene=scene_name, checkpoint=state_iter,
                           n_gaussians=count)
            single_rows += s_rows
            accum_rows += a_rows
            if checks:
                for row in checks:
                    row.update(scene=scene_name, checkpoint=state_iter,
                               camera_id=cam_idx)
                sanity_rows += checks
                print("[grad-study] sanity checks:", flush=True)
                for row in checks:
                    print(f"    {row['check']:24s} passed={row.get('passed')}  "
                          f"{ {k: (round(v, 8) if isinstance(v, float) else v) for k, v in row.items() if k in ('relative_gradient_error','cosine_similarity','gradient_norm_ratio')} }",
                          flush=True)

        after = state_fingerprint(pipeline)
        unchanged = (before.get("gaussians_checksum") == after.get("gaussians_checksum")
                     and before.get("adam_checksum") == after.get("adam_checksum")
                     and before.get("n_gaussians") == after.get("n_gaussians"))
        fingerprints.append({"checkpoint": state_iter, "before": before,
                            "after": after, "state_unchanged": bool(unchanged)})
        print(f"[grad-study] iter {state_iter}: {len(cameras)} cameras, "
              f"n_g={count}, {time.perf_counter() - started:.0f}s, "
              f"state_unchanged={unchanged}", flush=True)
        if not unchanged:
            print("[grad-study] WARNING: the frozen state CHANGED during "
                  "analysis; results for this checkpoint are not controlled.",
                  flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "scene": scene_name,
        "scene_path": str(args.scene),
        "seed": int(args.seed),
        "stbn_seed": int(args.stbn_seed),
        "probe_fraction": float(args.probe_fraction),
        "pixels_per_tile": budget,
        "tile_area": TILE_AREA,
        "select_mode": stbn_table.SELECT_GLOBAL,
        "k_values": sorted(args.k_values),
        "cameras": [int(c) for c in cameras],
        "state_reached_with": "dense supervision (regime-independent trajectory)",
        "no_parameter_updates": True,
        "regularization_included": False,
        "scalar_groups": {k: list(v) for k, v in SCALAR_GROUPS.items()},
        "sh_rest_group": SH_REST_GROUP,
        "sh_rest_limitation": (
            "features_rest_grad is overwritten per backward (its shader is the "
            "sole writer and assigns rather than accumulates), so SH-rest is "
            "measured in the single-mask pass only, not in the K-accumulation."),
        "normalization": (
            "the loss weights each supervised pixel by TILE_AREA/k = 1/rho, so a "
            "single masked backward is already an unbiased estimate of the dense "
            "gradient; K masks combine as a MEAN, not a sum"),
        "methods": [{"name": n, "source": s, "traversal": t}
                    for n, s, t in METHODS],
        "temporal_only_definition": (
            "NOT an existing training mode; defined for this diagnostic. Each "
            "pixel draws a phase in 0..P-1 (P = round(1/rho)) and is selected "
            "when t mod P == phase: perfect temporal stratification, white "
            "spatial arrangement."),
        "sanity_checks": sanity_rows,
        "state_fingerprints": fingerprints,
        "single_mask": single_rows,
        "accumulation": accum_rows,
    }, indent=2))
    print(f"[grad-study] wrote {out_path} "
          f"({len(single_rows)} single-mask + {len(accum_rows)} accumulation "
          f"observations)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
