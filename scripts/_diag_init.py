import sys, numpy as np
sys.path.insert(0, "src")
from plyfile import PlyData
from trainer.colmap import load_colmap

np.set_printoptions(suppress=True)

def pctl(a, ps): return {p: float(np.percentile(a, p)) for p in ps}

# ---------- 1. Our init ----------
ds = load_colmap("datasets/360_v2/bicycle", downsample=4)
means = ds.initial_points["means"]
logsc = ds.initial_points["log_scales"]
sc = np.exp(logsc[:,0])
n = means.shape[0]
print("=== OUR INIT ===")
print("n_init_points:", n)
print("scene_extent:", ds.scene_extent)

# camera centroid
import numpy as np
# recompute cam centroid from cameras (view matrix -> cam center = -R^T t)
cam_centers = []
for e in ds.images:
    V = e.camera.view_matrix
    R = V[:3,:3]; t = V[:3,3]
    cam_centers.append(-R.T @ t)
cam_centers = np.array(cam_centers)
centroid = cam_centers.mean(axis=0)
print("cam_centroid:", centroid)
print("n_cams:", len(ds.images))
print("our init scale: median %.5f p90 %.5f p99 %.5f max %.5f" % (
    np.median(sc), np.percentile(sc,90), np.percentile(sc,99), sc.max()))
print("our init scale min:", sc.min())

# ---------- 2. INRIA input.ply ----------
print("\n=== INRIA input.ply ===")
try:
    p = PlyData.read("datasets/3DGS_INRIA_RESULTS/models/bicycle/input.ply")
    v = p['vertex']
    props = [pr.name for pr in v.properties]
    print("n:", v.count)
    print("fields:", props)
    ix = np.array(v['x']); iy = np.array(v['y']); iz = np.array(v['z'])
    inria_pts = np.stack([ix,iy,iz],1)
    print("bbox min:", inria_pts.min(0), "max:", inria_pts.max(0))
    print("our  bbox min:", means.min(0), "max:", means.max(0))
except Exception as ex:
    print("FAILED:", ex)
    inria_pts = None

# ---------- 3. distant region ----------
print("\n=== DISTANT REGION (top 5% dist from cam centroid) ===")
d_ours = np.linalg.norm(means - centroid, axis=1)
thr = np.percentile(d_ours, 95)
print("dist-from-centroid threshold (95pctl):", thr)
mask = d_ours >= thr
print("our init points in distant region:", int(mask.sum()))
sc_d = sc[mask]
print("distant init scale: median %.4f p90 %.4f p99 %.4f max %.4f" % (
    np.median(sc_d), np.percentile(sc_d,90), np.percentile(sc_d,99), sc_d.max()))
print("distant dist range:", d_ours[mask].min(), d_ours[mask].max())
# near region scale for contrast
sc_near = sc[~mask]
print("NEAR init scale: median %.4f p90 %.4f" % (np.median(sc_near), np.percentile(sc_near,90)))

# ---------- 4. INRIA final ply ----------
print("\n=== INRIA FINAL point_cloud.ply ===")
pf = PlyData.read("datasets/3DGS_INRIA_RESULTS/models/bicycle/point_cloud/iteration_30000/point_cloud.ply")
vf = pf['vertex']
propsf = [pr.name for pr in vf.properties]
frest = [x for x in propsf if x.startswith("f_rest")]
scale_props = sorted([x for x in propsf if x.startswith("scale")])
print("INRIA final n:", vf.count)
print("INRIA f_rest count:", len(frest))
print("INRIA scale props:", scale_props)
fx = np.array(vf['x']); fy = np.array(vf['y']); fz = np.array(vf['z'])
fpts = np.stack([fx,fy,fz],1)
fopac_logit = np.array(vf['opacity'])
fopac = 1/(1+np.exp(-fopac_logit))
# scales are log
fsc = np.exp(np.array(vf[scale_props[0]]))
print("INRIA final opacity: frac<0.1 %.3f median %.3f frac>0.3 %.3f" % (
    (fopac<0.1).mean(), np.median(fopac), (fopac>0.3).mean()))
# distant region in INRIA final using SAME centroid & threshold
df = np.linalg.norm(fpts - centroid, axis=1)
fmask = df >= thr
print("INRIA final splats in distant region:", int(fmask.sum()))
print("  vs our init seeds there:", int(mask.sum()))
print("INRIA distant scale (exp scale0): median %.4f p90 %.4f p99 %.4f" % (
    np.median(fsc[fmask]), np.percentile(fsc[fmask],90), np.percentile(fsc[fmask],99)))
print("INRIA distant opacity: median %.3f frac>0.3 %.3f frac<0.1 %.3f" % (
    np.median(fopac[fmask]), (fopac[fmask]>0.3).mean(), (fopac[fmask]<0.1).mean()))
print("INRIA overall scale (exp scale0): median %.4f p90 %.4f max %.4f" % (
    np.median(fsc), np.percentile(fsc,90), fsc.max()))

# ---------- 5. OUR final ply ----------
print("\n=== OUR FINAL output_mine.ply ===")
po = PlyData.read("outputs/train/20260703_141907/output_mine.ply")
vo = po['vertex']
propso = [pr.name for pr in vo.properties]
frest_o = [x for x in propso if x.startswith("f_rest")]
scale_props_o = sorted([x for x in propso if x.startswith("scale")])
print("our final n:", vo.count)
print("our f_rest count:", len(frest_o))
print("our props sample:", propso[:15])
if 'opacity' in propso:
    oop_l = np.array(vo['opacity'])
    # our opacity may be stored as logit or prob — check range
    print("our opacity raw range:", oop_l.min(), oop_l.max())
    oop = 1/(1+np.exp(-oop_l))
    print("our opacity(sigmoid) frac<0.1 %.3f median %.3f frac>0.3 %.3f" % (
        (oop<0.1).mean(), np.median(oop), (oop>0.3).mean()))
if scale_props_o:
    osc = np.exp(np.array(vo[scale_props_o[0]]))
    ox=np.array(vo['x']);oy=np.array(vo['y']);oz=np.array(vo['z'])
    opts=np.stack([ox,oy,oz],1)
    do=np.linalg.norm(opts-centroid,axis=1)
    omask=do>=thr
    print("our final splats in distant region:", int(omask.sum()))
    print("our final distant scale: median %.4f p90 %.4f" % (np.median(osc[omask]), np.percentile(osc[omask],90)))
    print("our final overall scale: median %.4f p90 %.4f max %.4f" % (np.median(osc), np.percentile(osc,90), osc.max()))
