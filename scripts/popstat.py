"""Population-statistics instrumentation for OUR trainer, byte-compatible with
the SADGS reference's ``_ref_popstat.py`` output schema.

Writes one JSON line per (pre/post) snapshot around each densify tick to a
``ref_popstat.jsonl`` so our trajectory can be diffed line-for-line against the
reference's (outputs/sadgs/<run>/ref_popstat.jsonl). Every field name, the
percentile set, the eta/fan-out math and the rounding all mirror
SADGS/SADGS/_ref_popstat.py::snapshot so the two files are directly comparable.

Schema per line:
  {iter, phase("pre"|"post"), kind("warmup"|"structure"), count,
   world_max_scale{p10,p25,p50,p75,p90,p99,max},   # activated sigma, world units
   opacity{p10,p25,p50,p75,p90,p99,max}}           # activated sigmoid
  plus, ONLY for kind=="structure" & phase=="pre" (eta accumulators live then):
   eta_max_seen{...}, high_ratio{p50,p90,frac_gt_tau},
   fanout{n_split_cand,prod_mean,prod_p50,prod_p90,prod_max,children_requested}

Reuses the read-back layout of scripts/densify_diag.py (gaussian stride = 14
floats: means[0:3], log-scale[3:6], quat[6:10], opacity-logit[10], sh0[11:14]).
"""

from __future__ import annotations

import json

import numpy as np

_GAUSSIAN_STRIDE = 14


def _counter(pipeline) -> int:
    return int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])


def _u32(buf, n: int) -> np.ndarray:
    return np.frombuffer(buf.to_numpy().tobytes(), dtype=np.uint32)[:n]


def _gaussians(pipeline, n: int) -> np.ndarray:
    return np.frombuffer(pipeline.buffers.gaussians.to_numpy().tobytes(),
                         dtype=np.float32).reshape(-1, _GAUSSIAN_STRIDE)[:n]


def _pcts(a: np.ndarray, ps=(10, 25, 50, 75, 90, 99)) -> dict:
    """Percentile dict matching SADGS _ref_popstat._pcts EXACTLY (same ps set,
    same rounding: 5 dp for percentiles, 4 dp for max)."""
    d = {f"p{p}": round(float(np.percentile(a, p)), 5) for p in ps}
    d["max"] = round(float(a.max()), 4)
    return d


class PopStat:
    """One open ``ref_popstat.jsonl`` writer. Call :meth:`snapshot` pre and post
    each densify tick. Line-buffered so a killed run still keeps its trajectory."""

    def __init__(self, path):
        self._fh = open(path, "w", buffering=1)
        self.path = str(path)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def snapshot(self, pipeline, iteration: int, phase: str, kind: str, *,
                 extent: float, structure_cfg, split_ratio_threshold: float) -> None:
        """Write one population snapshot. ``phase`` in {'pre','post'}; ``kind``
        in {'warmup','structure'}. Reads back GPU state (wait_for_idle first),
        so only call it on the ~17 densify ticks of a run, never per-iter."""
        pipeline.device.wait_for_idle()
        n = _counter(pipeline)
        g = _gaussians(pipeline, n)

        # Activated per-gaussian max scale (sigma) in world units == exp(log-scale),
        # max over the 3 axes. SADGS: gaussians.get_scaling.max(dim=1).
        max_scale = np.exp(g[:, 3:6].astype(np.float64)).max(axis=1)
        # Activated opacity == sigmoid(logit). SADGS: gaussians.get_opacity.
        opacity = 1.0 / (1.0 + np.exp(-g[:, 10].astype(np.float64)))

        rec: dict = {
            "iter": int(iteration), "phase": phase, "kind": kind, "count": int(n),
            "world_max_scale": _pcts(max_scale),
            "opacity": _pcts(opacity),
        }

        # eta / fan-out only meaningful pre a STRUCTURE pass (accumulators live).
        if kind == "structure" and phase == "pre":
            b = pipeline.buffers
            vc = _u32(b.eta_n_total, n).astype(np.float64)              # accum_view_count
            n_high = _u32(b.eta_buffers["n_high"], n).astype(np.float64)  # eta_high_count
            max_x = _u32(b.eta_buffers["max_x"], n).astype(np.float64) / 65536.0
            max_y = _u32(b.eta_buffers["max_y"], n).astype(np.float64) / 65536.0
            max_z = _u32(b.eta_buffers["max_z"], n).astype(np.float64) / 65536.0
            me = np.stack([max_x, max_y, max_z], axis=1)                 # max_eta_3ch [N,3]
            eta_max = me.max(axis=1)

            seen = vc > 0
            hr = np.zeros_like(vc)
            hr[seen] = n_high[seen] / vc[seen]
            if seen.any():
                rec["eta_max_seen"] = _pcts(eta_max[seen])
                rec["high_ratio"] = {
                    "p50": round(float(np.quantile(hr[seen], 0.5)), 4),
                    "p90": round(float(np.quantile(hr[seen], 0.9)), 4),
                    "frac_gt_tau": round(float(np.mean(hr[seen] > split_ratio_threshold)), 4),
                }
                # Fan-out the structure pass WOULD apply: ks = ceil(sqrt(clamp(eta,1)))
                # per axis, clamped to [1, max_split_factor]; prod over axes. SADGS
                # hardcodes sqrt (power 0.5); we mirror that for comparability.
                cap = int(structure_cfg.max_split_factor)
                ks = np.clip(np.ceil(np.sqrt(np.clip(me, 1.0, None))), 1, cap)
                prod = ks.prod(axis=1)
                # Split candidates ~ high_ratio > tau & large scale (SADGS: 0.001*extent).
                large = max_scale > (0.001 * extent)
                cand = seen & (hr > split_ratio_threshold) & large
                if cand.any():
                    pc = prod[cand].astype(np.float64)
                    rec["fanout"] = {
                        "n_split_cand": int(cand.sum()),
                        "prod_mean": round(float(pc.mean()), 2),
                        "prod_p50": int(np.quantile(pc, 0.5)),
                        "prod_p90": int(np.quantile(pc, 0.9)),
                        "prod_max": int(pc.max()),
                        "children_requested": int((pc - 1).sum()),
                    }

        self._fh.write(json.dumps(rec) + "\n")
