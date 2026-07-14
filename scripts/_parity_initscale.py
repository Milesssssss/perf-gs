"""PROBE A: init-scale population parity vs reference distCUDA2.

Loads real bicycle COLMAP points through the ACTUAL trainer.colmap code path
(so any hidden multiplier / clamp / normalization is exercised), then compares
the resulting per-point log_scales against:
  (1) an EXACT reference statistic (mean of 3 nearest SQUARED dists, sqrt, log)
      computed with scipy cKDTree on the identical points, and
  (2) an APPROXIMATE replication of the reference CUDA distCUDA2 (boxMeanDist):
      Morton-sort points, then kNN only within +/-3 index window + Morton boxes.
Reports per-point ratio distribution and projects init 1-sigma to pixels at a
typical bicycle camera depth.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from trainer.colmap import load_colmap, _parse_points3d_bin  # noqa: E402


def pcts(x, label):
    ps = [1, 10, 25, 50, 75, 90, 99]
    print(f"  {label:26s} " + " ".join(f"p{p}={np.percentile(x, p):.4f}" for p in ps)
          + f"  min={x.min():.4f} max={x.max():.4f}")


# ---- reference approximate distCUDA2 (Morton box replication) ----
def prep_morton(x):
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249
    return x


def morton_codes(pts):
    minn = pts.min(0); maxx = pts.max(0)
    span = np.where((maxx - minn) == 0, 1.0, (maxx - minn))
    q = ((pts - minn) / span * ((1 << 10) - 1)).astype(np.uint32)
    x = prep_morton(q[:, 0]); y = prep_morton(q[:, 1]); z = prep_morton(q[:, 2])
    return x | (y << 1) | (z << 2)


def approx_distcuda2(pts, box_size=1024):
    """Replicate SADGS simple_knn.cu boxMeanDist: mean of 3 nearest sq dists,
    approximated over Morton-sorted order with a +/-3 window seed + box prune."""
    P = pts.shape[0]
    codes = morton_codes(pts)
    order = np.argsort(codes, kind="stable")
    sp = pts[order]  # points in sorted order
    num_boxes = (P + box_size - 1) // box_size
    # box min/max
    box_min = np.empty((num_boxes, 3)); box_max = np.empty((num_boxes, 3))
    for b in range(num_boxes):
        s = sp[b * box_size: min(P, (b + 1) * box_size)]
        box_min[b] = s.min(0); box_max[b] = s.max(0)

    out = np.empty(P)
    for idx in range(P):
        point = sp[idx]
        # seed with +/-3 window
        best = np.array([np.inf, np.inf, np.inf])
        lo = max(0, idx - 3); hi = min(P - 1, idx + 3)
        for i in range(lo, hi + 1):
            if i == idx:
                continue
            d = sp[i] - point
            dist = float(d @ d)
            best = np.sort(np.concatenate([best, [dist]]))[:3]
        reject = best[2]
        best = np.array([np.inf, np.inf, np.inf])
        # box prune
        for b in range(num_boxes):
            diff = np.zeros(3)
            below = point < box_min[b]; above = point > box_max[b]
            diff = np.where(below | above,
                            np.minimum(np.abs(point - box_min[b]), np.abs(point - box_max[b])),
                            0.0)
            bdist = float(diff @ diff)
            if bdist > reject or bdist > best[2]:
                continue
            s0 = b * box_size; s1 = min(P, (b + 1) * box_size)
            block = sp[s0:s1]
            dd = block - point
            dsq = np.einsum("ij,ij->i", dd, dd)
            # exclude self
            self_local = idx - s0
            if 0 <= self_local < block.shape[0]:
                dsq[self_local] = np.inf
            cand = np.concatenate([best, dsq])
            best = np.sort(cand)[:3]
        out[order[idx]] = (best[0] + best[1] + best[2]) / 3.0
    return out


def main():
    scene_dir = REPO / "datasets/360_v2/bicycle"
    ds = load_colmap(scene_dir, downsample=4)
    our_log = ds.initial_points["log_scales"][:, 0].astype(np.float64)  # isotropic, col 0
    pts, _ = _parse_points3d_bin(scene_dir / "sparse/0/points3D.bin")
    pts = pts.astype(np.float64)
    n = pts.shape[0]
    print(f"points: {n}  scene_extent={ds.scene_extent:.3f}\n")

    # (1) EXACT reference statistic via cKDTree (mean of 3 nearest SQUARED dists)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=4)  # col0 self=0
    dist_sq_exact = (dists[:, 1:4] ** 2).mean(axis=1)
    dist_sq_exact = np.clip(dist_sq_exact, 1e-7, None)
    ref_log_exact = np.log(np.sqrt(dist_sq_exact))

    # (2) APPROXIMATE reference distCUDA2 (Morton box) -- the ACTUAL algo
    dist_sq_approx = np.clip(approx_distcuda2(pts), 1e-7, None)
    ref_log_approx = np.log(np.sqrt(dist_sq_approx))

    print("=== log_scale distributions ===")
    pcts(our_log, "OURS (colmap.py)")
    pcts(ref_log_exact, "REF exact cKDTree")
    pcts(ref_log_approx, "REF approx distCUDA2")

    print("\n=== sigma = exp(log_scale) (world units) ===")
    pcts(np.exp(our_log), "OURS sigma")
    pcts(np.exp(ref_log_exact), "REF exact sigma")
    pcts(np.exp(ref_log_approx), "REF approx sigma")

    # per-point ratio (sigma_ours / sigma_ref) for both references
    for name, ref in (("exact", ref_log_exact), ("approx", ref_log_approx)):
        ratio = np.exp(our_log - ref)  # sigma ratio
        print(f"\n=== sigma ratio OURS/{name} ===")
        for p in (10, 50, 90):
            print(f"  p{p} ratio = {np.percentile(ratio, p):.5f}")
        print(f"  max ratio = {ratio.max():.5f}   median = {np.median(ratio):.5f}")
        cc = np.corrcoef(our_log, ref)[0, 1]
        print(f"  correlation(log_scale ours,{name}) = {cc:.6f}")

    # ---- projection to pixels at typical bicycle depth ----
    cam = ds.images[0].camera
    W, Hpx = cam.image_size
    V = cam.view_matrix.astype(np.float64)
    P = cam.proj_matrix.astype(np.float64)
    fx = P[0, 0] * W * 0.5
    # median depth of points seen by cam 0 (view-space z)
    mv = (V @ np.c_[pts, np.ones(n)].T).T[:, :3]
    front = mv[:, 2] > 1e-6
    med_z = np.median(mv[front, 2])
    print(f"\n=== projection: cam0 {W}x{Hpx}, fx={fx:.1f}px, median depth z={med_z:.3f} ===")
    for name, log_s in (("OURS", our_log), ("REF exact", ref_log_exact),
                        ("REF approx", ref_log_approx)):
        sig = np.exp(log_s)
        L_px = sig * fx / med_z  # 1-sigma projected axis at median depth
        print(f"  {name:11s} 1-sigma projected L(px): "
              f"p10={np.percentile(L_px, 10):.3f} "
              f"p50={np.percentile(L_px, 50):.3f} "
              f"p90={np.percentile(L_px, 90):.3f}")


if __name__ == "__main__":
    main()
