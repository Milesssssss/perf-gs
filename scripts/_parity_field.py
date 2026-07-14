"""Structure-field parity: reference SADGS torch preprocessing vs our numpy port.

Runs BOTH on the identical decoded bicycle image at the run resolution and
compares the aggregated structure tensor S̄ = (Sxx, Sxy, Syy) and its principal
eigenvalue λ1 pixel-for-pixel. This is the decisive test the diagnosis needs:
eta = L·√λ1, so if λ1 distributions diverge, the field construction is the bug;
if they match, the error is in projection / axis length.

Run with the SYSTEM python (has torch): python scripts/_parity_field.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SADGS = REPO / "SADGS" / "SADGS"
IMG = REPO / "datasets" / "360_v2" / "bicycle" / "images_4" / "_DSC8679.JPG"
# Run resolution (from config.json): 1237x822.
W, H = 1237, 822


def load_img_chw():
    im = Image.open(IMG).convert("RGB").resize((W, H), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0          # (H,W,3) in [0,1]
    return a


def run_reference(img_hwc):
    """Reference get_multiscale_structure_tensor_v1(levels=4) — verbatim."""
    import torch
    sys.path.insert(0, str(SADGS))
    from utils.loss_utils import get_multiscale_structure_tensor_v1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.from_numpy(img_hwc).permute(2, 0, 1).unsqueeze(0).to(dev)  # (1,3,H,W)
    with torch.no_grad():
        st = get_multiscale_structure_tensor_v1(t, levels=4)   # (1,3,H,W)
    st = st.squeeze(0).permute(1, 2, 0).cpu().numpy()          # (H,W,3)
    return st


def run_ours(img_hwc):
    sys.path.insert(0, str(REPO / "src"))
    from trainer.structure_analysis import compute_structure_tensor_field
    # our port takes sRGB [0,1] HxWx3 directly
    return compute_structure_tensor_field(img_hwc)


def lam1(st):
    a, b, c = st[..., 0], st[..., 1], st[..., 2]
    mid = 0.5 * (a + c)
    disc = np.sqrt(np.maximum((0.5 * (a - c)) ** 2 + b * b, 0.0))
    return mid + disc


def pcts(x, label):
    ps = [1, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(x, ps)
    print(f"  {label:24s} " + " ".join(f"p{p}={v:.5f}" for p, v in zip(ps, vals))
          + f" max={x.max():.5f}")


def main():
    img = load_img_chw()
    print(f"image {IMG.name} at {W}x{H}\n")

    ref = run_reference(img)
    ours = run_ours(img)
    print(f"ref shape {ref.shape}  ours shape {ours.shape}\n")

    lr, lo = lam1(ref).astype(np.float64), lam1(ours).astype(np.float64)
    sr, so = np.sqrt(np.maximum(lr, 0)), np.sqrt(np.maximum(lo, 0))

    print("=== lambda1 distribution ===")
    pcts(lr, "reference lambda1")
    pcts(lo, "ours      lambda1")
    print("\n=== sqrt(lambda1)  (the eta multiplier) ===")
    pcts(sr, "reference sqrt(lam1)")
    pcts(so, "ours      sqrt(lam1)")

    print("\n=== per-tensor-component percentiles (median) ===")
    for i, nm in enumerate(("Sxx", "Sxy", "Syy")):
        print(f"  {nm}: ref p50={np.percentile(ref[...,i],50):.6f}  "
              f"ours p50={np.percentile(ours[...,i],50):.6f}")

    print("\n=== pixel-for-pixel agreement on lambda1 ===")
    mae = np.mean(np.abs(lr - lo))
    rmse = np.sqrt(np.mean((lr - lo) ** 2))
    corr = np.corrcoef(lr.ravel(), lo.ravel())[0, 1]
    ratio = np.median(lo[lr > 1e-4] / lr[lr > 1e-4])
    print(f"  MAE={mae:.5f}  RMSE={rmse:.5f}  corr={corr:.4f}  median(ours/ref)={ratio:.4f}")

    # The load-bearing number for the runaway: fraction of pixels where sqrt(lam1)
    # is high enough that a modest projected axis (say 3px) gives eta>1.
    for thr, axpx in [(1/3, 3), (0.5, 2), (1.0, 1)]:
        print(f"  frac sqrt(lam1)>{thr:.3f} (eta>1 at L={axpx}px):  "
              f"ref={np.mean(sr>thr):.3f}  ours={np.mean(so>thr):.3f}")


if __name__ == "__main__":
    main()
