"""Projected-axis + eta parity on REAL gaussians.

Loads a checkpoint PLY (means/scales/quats), a real bicycle camera, and the
cached structure field, then computes — with the SAME projection the shader
uses (verified equivalent to SADGS compute_projected_axes_subset) — the
distribution of:
  - projected 1-sigma axis length L_k (pixels), per axis
  - sqrt(lambda1) sampled at each projected mean
  - eta_k = L_k * sqrt(lambda1)
  - fraction with max-axis eta > 1  (the high vote)

Purpose: field + formula parity already matched the reference; this measures
whether the RUNAWAY comes from L being large (big gaussians / projection) vs the
field. Pure numpy, no GPU, no rerun.

    python scripts/_parity_axis.py [ply_path]
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from trainer.colmap import load_colmap  # noqa: E402


def read_ply(path):
    import struct
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        n = 0
        props = []
        while True:
            line = f.readline().strip().decode()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property float"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(n * len(props) * 4), dtype=np.float32).reshape(n, len(props))
    idx = {name: i for i, name in enumerate(props)}
    return data, idx


def quat_to_R(q):  # q = (w,x,y,z) rows -> (N,3,3), matches matrixFromQuaternion
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def lam1_field(field, px, py):
    """Nearest-texel lambda1 at pixel coords (clamped)."""
    H, W = field.shape[:2]
    xi = np.clip(px.astype(int), 0, W - 1)
    yi = np.clip(py.astype(int), 0, H - 1)
    s = field[yi, xi]  # (N,3)
    a, b, c = s[:, 0], s[:, 1], s[:, 2]
    mid = 0.5 * (a + c)
    disc = np.sqrt(np.maximum((0.5 * (a - c)) ** 2 + b * b, 0.0))
    return (mid + disc).astype(np.float64)


def pcts(x, label):
    ps = [1, 10, 25, 50, 75, 90, 95, 99]
    print(f"  {label:22s} " + " ".join(f"p{p}={np.percentile(x,p):.3f}" for p in ps)
          + f" max={x.max():.2f}")


def main():
    ply = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "outputs/train/diag_run/output.ply")
    scene = load_colmap(REPO / "datasets/360_v2/bicycle", downsample=4)
    cam = scene.images[0].camera
    W, Hpx = cam.image_size
    V = cam.view_matrix.astype(np.float64)
    P = cam.proj_matrix.astype(np.float64)

    fld = np.load(REPO / "datasets/360_v2/bicycle/structure_cache/sad_v6_1237x822_L4_s1.5_r3_g3.npz")
    field = fld["fields"][0].astype(np.float64)  # cam 0 field (H,W,3)

    data, idx = read_ply(ply)
    print(f"ply {Path(ply).name}: {data.shape[0]} gaussians at cam {W}x{Hpx}\n")
    means = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float64)
    log_s = data[:, [idx["scale_0"], idx["scale_1"], idx["scale_2"]]].astype(np.float64)
    quat = data[:, [idx["rot_0"], idx["rot_1"], idx["rot_2"], idx["rot_3"]]].astype(np.float64)
    scale = np.exp(log_s)

    # World 3D sigma (before projection) — is the gaussian itself huge?
    print("=== world-space 1-sigma scale (3D units) ===")
    pcts(scale.max(axis=1), "max axis sigma (world)")

    # Project means to view + pixel.
    mv = (V @ np.c_[means, np.ones(len(means))].T).T[:, :3]  # (N,3) view space
    infront = mv[:, 2] > 1e-6
    clip = (P @ np.c_[mv, np.ones(len(mv))].T).T
    wpos = clip[:, 3] > 1e-6
    ok = infront & wpos
    ndc = clip[ok, :2] / clip[ok, 3:4]
    px = (ndc[:, 0] * 0.5 + 0.5) * W
    py = (ndc[:, 1] * 0.5 + 0.5) * Hpx
    onscreen = (px >= 0) & (px < W) & (py >= 0) & (py < Hpx)

    mvz = mv[ok, 2]
    fx = P[0, 0] * W * 0.5
    fy = P[1, 1] * Hpx * 0.5
    # Jacobian per point (no off-axis clamp, matching reference).
    J00 = fx / mvz; J02 = -fx * (mv[ok, 0] / mvz) / mvz
    J11 = fy / mvz; J12 = -fy * (mv[ok, 1] / mvz) / mvz

    Wr = V[:3, :3]
    R = quat_to_R(quat[ok])
    Rtot = np.einsum("ij,njk->nik", Wr, R)       # R_view @ R_local, (N,3,3)
    axes_cam = Rtot * scale[ok][:, None, :]        # columns = axis vectors, (N,3,3)
    ax_x, ax_y, ax_z = axes_cam[:, 0, :], axes_cam[:, 1, :], axes_cam[:, 2, :]  # each (N,3)
    u = J00[:, None] * ax_x + J02[:, None] * ax_z  # (N,3)
    v = J11[:, None] * ax_y + J12[:, None] * ax_z
    L = np.sqrt(u * u + v * v + 1e-8)              # (N,3) projected 1-sigma axis px

    # Restrict to on-screen for the eta measurement.
    Lon = L[onscreen]
    sq = np.sqrt(lam1_field(field, px[onscreen], py[onscreen]))
    eta = Lon * sq[:, None]
    eta_max = eta.max(axis=1)

    print("\n=== projected 1-sigma axis length L (pixels), on-screen ===")
    pcts(Lon.max(axis=1), "max-axis L (px)")
    pcts(Lon.ravel(), "all-axis L (px)")
    print("\n=== sqrt(lambda1) sampled at projected means ===")
    pcts(sq, "sqrt(lam1)")
    print("\n=== eta = L * sqrt(lambda1) ===")
    pcts(eta_max, "eta_max (max axis)")
    print(f"\n  fraction eta_max > 1.0 (HIGH vote): {np.mean(eta_max > 1.0):.4f}")
    print(f"  fraction eta_max <= 0.1 (LOW/prune):  {np.mean(eta_max <= 0.1):.4f}")
    # If L were the culprit: what L would give median eta ~ 0.5 (selective)?
    print(f"\n  median max-axis L = {np.median(Lon.max(axis=1)):.2f} px; "
          f"median sqrt(lam1) = {np.median(sq):.3f}")
    print(f"  => median eta_max = {np.median(eta_max):.2f}  "
          f"(need L~{0.5/np.median(sq):.2f}px for median eta~0.5)")


if __name__ == "__main__":
    main()
