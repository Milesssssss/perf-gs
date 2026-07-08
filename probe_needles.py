import sys, numpy as np
sys.path.insert(0, "src")
from trainer.colmap import load_colmap
from trainer.config import AdamLRs, RenderConfig
from trainer.device import create_device
from trainer.pipeline import Pipeline

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500

scene = load_colmap("datasets/synthetic/suzanne_blender", downsample=4)
n_cams = len(scene.images)
n_init = scene.initial_points["means"].shape[0]
extent = scene.scene_extent
W, H = scene.images[0].camera.image_size
print(f"cams={n_cams} init={n_init} extent={extent:.3f} {W}x{H}")

def srgb_to_linear(c):
    low = c/12.92; high = np.power(np.clip((c+0.055)/1.055,1e-7,None),2.4)
    return np.where(c<=0.04045, low, high).astype(np.float32)

from trainer.colmap import load_image
targets_srgb = [load_image(e, scene.image_dir, W, H, load_alpha=True) for e in scene.images]

device = create_device(backend="vulkan")
pipeline = Pipeline(device=device, width=W, height=H, max_splats=1_000_000,
                    cfg=RenderConfig(z_near_cull=0.01, opacity_reg=0.01,
                                     scale_reg=0.01, w_alpha=0.2, lambda_ssim=0))
pipeline.set_gaussians_from_numpy(**scene.initial_points)
pipeline.clear_tracking()
target_textures = [pipeline.create_target_texture(t) for t in targets_srgb]

lr_mean_init = 1.6e-4 * extent
lr_mean_final = lr_mean_init * 1e-2
decay = (lr_mean_final/lr_mean_init) ** (1.0/max(ITERS,1))

rng = np.random.default_rng(0)
import time; t0=time.time()
for it in range(1, ITERS+1):
    ci = int(rng.integers(0, n_cams))
    pipeline.set_target_texture(target_textures[ci])
    lrs = AdamLRs(mean=lr_mean_init*(decay**(it-1)))
    pipeline.train_step(scene.images[ci].camera, lrs=lrs)
    if it % 250 == 0:
        print(f"iter {it} {it/(time.time()-t0):.1f} it/s")
pipeline.device.wait_for_idle()

n = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
raw = pipeline.buffers.gaussians.to_numpy()
arr = np.frombuffer(raw, dtype=np.float32)
stride = arr.size // pipeline.max_splats
print(f"n_gaussians={n} stride_floats={stride}")
g = arr.reshape(pipeline.max_splats, stride)[:n]

mean = g[:, 0:3]
log_scale = g[:, 3:6]
rot = g[:, 6:10]
opacity_logit = g[:, 10]
color_logit = g[:, 11:14]

scale = np.exp(log_scale)
ratio = scale.max(1) / np.maximum(scale.min(1), 1e-20)
def pct(x,p): return float(np.percentile(x,p))
print("=== ANISOTROPY ratio max/min ===")
print(f"median={np.median(ratio):.3f} p90={pct(ratio,90):.3f} p99={pct(ratio,99):.3f} max={ratio.max():.3f}")
print(f"frac>3={np.mean(ratio>3):.4f} frac>10={np.mean(ratio>10):.4f} frac>50={np.mean(ratio>50):.4f}")
print("=== per-axis scale medians (sorted per-g) ===")
ssort = np.sort(scale,1)
print(f"min-axis median={np.median(ssort[:,0]):.5f} mid={np.median(ssort[:,1]):.5f} max={np.median(ssort[:,2]):.5f}")
print(f"min-axis p01={pct(ssort[:,0],1):.6f} p50={np.median(ssort[:,0]):.6f}  overall scale min={scale.min():.6e}")
op = 1/(1+np.exp(-opacity_logit))
print("=== opacity sigmoid ===")
print(f"min={op.min():.4f} median={np.median(op):.4f} max={op.max():.4f} frac<0.02={np.mean(op<0.02):.4f} frac<0.1={np.mean(op<0.1):.4f}")
qn = np.linalg.norm(rot,axis=1)
print("=== quat norm ===")
print(f"min={qn.min():.4f} median={np.median(qn):.4f} max={qn.max():.4f} frac<0.99={np.mean(qn<0.99):.4f} frac>1.01={np.mean(qn>1.01):.4f}")

# correlation among needles
mask = ratio>10
print(f"=== needles (ratio>10): count={mask.sum()} ===")
if mask.sum()>0:
    print(f"needle opacity median={np.median(op[mask]):.4f} vs bulk={np.median(op[~mask]):.4f}")
    print(f"needle min-axis scale median={np.median(ssort[mask,0]):.6f} max-axis median={np.median(ssort[mask,2]):.5f}")
    ctr = mean.mean(0)
    dist_all = np.linalg.norm(mean-ctr,1 if False else None,axis=1)
    print(f"needle dist-from-centroid median={np.median(dist_all[mask]):.3f} vs bulk={np.median(dist_all[~mask]):.3f} extent={extent:.3f}")
    print(f"needle mean world bbox min={mean[mask].min(0)} max={mean[mask].max(0)}")
    print(f"bulk   mean world bbox min={mean[~mask].min(0)} max={mean[~mask].max(0)}")
print("DONE")
