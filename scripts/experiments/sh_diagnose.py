# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""SH3 diagnosis on a finished checkpoint: is the higher-band signal not learned, or learned but useless?

Probes the SH-rest Adam v-floor and a same-checkpoint active-degree ablation of test PSNR.
    .venv/Scripts/python.exe scripts/experiments/sh_diagnose.py --densify-mode structure_aware --mip-splatting --seed 42 --sh-degree 3 --iters 3000
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
for _sub in ("running", "debug", "plots"):
    _p = str(_SCRIPTS / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np

from trainer.frontend.app import TrainingApplication
from trainer.frontend.cli import build_parser
from trainer.frontend.evaluation.evaluation import eval_test_psnr_srgb
from trainer.common.metrics import psnr as psnr_metric


def _build_args():
    p = build_parser()
    a = p.parse_args()
    a.no_viewer = True; a.profile_run = False; a.capture_external = False
    a.preview_every = 0; a.log_every = 0; a.eval_every = 0; a.psnr_curve = False
    if a.sh_degree <= 0:
        a.sh_degree = 3
    return a


# Low half = scalar 0, high half = scalar 1, matching common/half_pack.slang unpackHalf2x.
def _unpack_half2(u32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = (u32 & 0xFFFF).astype(np.uint16)
    hi = ((u32 >> 16) & 0xFFFF).astype(np.uint16)
    return lo.view(np.float16).astype(np.float64), hi.view(np.float16).astype(np.float64)


def _band_slices(sh_degree: int):
    """(name, start, stop) per SH band over a Gaussian's flat rest scalars: 3/5/7 coeffs per channel, times 3."""
    per_ch = [3, 5, 7][:sh_degree]      # coeffs per channel for bands 1..sh_degree
    slices = []
    # The shader iterates coefficient-major up to active*3, so bands use cumulative *3 boundaries.
    start = 0
    for i, c in enumerate(per_ch, start=1):
        stop = start + c * 3
        slices.append((f"deg{i}", start, stop))
        start = stop
    return slices


def instrument_sh_rest(pipeline, n: int, sh_degree: int, floor: float = 1e-15) -> None:
    b = pipeline.buffers
    rest_floats = b.sh.rest_floats
    feat_uints = b.sh.feat_uints
    fr = np.frombuffer(b.sh.features_rest.storage.to_numpy().tobytes(), dtype=np.uint32)
    fr = fr[: n * feat_uints].reshape(n, feat_uints)
    # adam_rest float32 SoA: word k of gaussian g at k*max_splats + g.
    ms = pipeline.max_splats
    ar_flat = np.frombuffer(b.sh.adam_rest.storage.to_numpy().tobytes(), dtype=np.float32)
    ar = np.empty((n, rest_floats, 2), dtype=np.float32)
    gi = np.arange(n)
    for s in range(rest_floats):
        ar[:, s, 0] = ar_flat[(2 * s) * ms + gi]
        ar[:, s, 1] = ar_flat[(2 * s + 1) * ms + gi]
    gr = np.frombuffer(b.sh.features_rest_grad.storage.to_numpy().tobytes(), dtype=np.uint32)
    gr = gr[: n * feat_uints].reshape(n, feat_uints)

    # Coeffs + grads are 2 scalars per uint (feat_uints = ceil(rest_floats/2)).
    coeff_lo, coeff_hi = _unpack_half2(fr)
    coeffs = np.empty((n, feat_uints * 2), dtype=np.float64)
    coeffs[:, 0::2] = coeff_lo; coeffs[:, 1::2] = coeff_hi
    coeffs = coeffs[:, :rest_floats]
    grad_lo, grad_hi = _unpack_half2(gr)
    grads = np.empty((n, feat_uints * 2), dtype=np.float64)
    grads[:, 0::2] = grad_lo; grads[:, 1::2] = grad_hi
    grads = grads[:, :rest_floats]
    m_arr = ar[..., 0].astype(np.float64)     # (n, rest_floats)
    v_arr = ar[..., 1].astype(np.float64)

    print(f"\n=== #2 SH-rest optimizer instrumentation (n={n}, rest_floats={rest_floats}, "
          f"SH_V_FLOOR={floor:.0e}) ===")
    print(f"{'band':>6} {'coeff_rms':>11} {'grad_rms':>11} {'m_rms':>11} "
          f"{'v_rms':>11} {'v@floor%':>9} {'|coeff|<1e-4%':>13}")
    for name, s, e in _band_slices(sh_degree):
        c = coeffs[:, s:e]; g = grads[:, s:e]; m = m_arr[:, s:e]; v = v_arr[:, s:e]
        vfloor_frac = float(np.mean(np.abs(v - floor) <= floor * 0.01 + 1e-12))
        near_zero = float(np.mean(np.abs(c) < 1e-4))
        print(f"{name:>6} {np.sqrt(np.mean(c*c)):>11.3e} {np.sqrt(np.mean(g*g)):>11.3e} "
              f"{np.sqrt(np.mean(m*m)):>11.3e} {np.sqrt(np.mean(v*v)):>11.3e} "
              f"{vfloor_frac*100:>8.1f}% {near_zero*100:>12.1f}%")
    print("  v@floor% ~100 => Adam step collapsed by the v-floor (the suspected bug).")
    print("  coeff_rms ~0 / |coeff|<1e-4 ~100% => higher bands never moved from init.")


def degree_ablation(pipeline, scene, targets, sh_degree: int) -> None:
    print(f"\n=== #1 Same-checkpoint degree ablation (test PSNR at clamped active degree) ===")
    for deg in range(sh_degree + 1):
        pipeline.set_active_sh_degree(deg)
        p, ncount = eval_test_psnr_srgb(pipeline, scene.colmapScene,
                                        scene.test_indices, targets.srgb)
        print(f"  active_degree={deg}:  test PSNR = {p:.4f} dB   (n_g={ncount})")
    print("  deg0 ~ deg3 => learned higher coeffs contribute ~nothing to held-out views.")


def main() -> int:
    args = _build_args()
    app = TrainingApplication(args)
    app.prepare_training_context()
    loop = app.build_training_loop()

    loop.advance_to_capture(args.iters)
    pipeline = app._pipeline
    scene = app._sceneContext
    n = int(pipeline.buffers.primitive.active_count.to_numpy().view(np.uint32)[0])
    print(f"[sh-diag] trained to iter {args.iters}, n_gaussians={n}, sh_degree={args.sh_degree}")

    if not pipeline.buffers.sh_enabled:
        print("[sh-diag] SH not enabled; run with --sh-degree > 0.")
        return 1

    instrument_sh_rest(pipeline, n, args.sh_degree)
    degree_ablation(pipeline, scene, app._targets, args.sh_degree)
    pipeline.runtime.wait_for_idle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
