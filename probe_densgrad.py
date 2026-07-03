import sys, numpy as np
sys.path.insert(0,"src")
from trainer.colmap import load_colmap, load_image
from trainer.device import create_device
from trainer.pipeline import AdamLRs, Pipeline

ITERS = int(sys.argv[1]) if len(sys.argv)>1 else 600
scene = load_colmap("datasets/synthetic/suzanne_blender", downsample=4)
n_cams=len(scene.images); extent=scene.scene_extent
W,H = scene.images[0].camera.image_size
targets=[load_image(e,scene.image_dir,W,H,load_alpha=True) for e in scene.images]
dev=create_device(backend="vulkan")
p=Pipeline(device=dev,width=W,height=H,max_splats=1_000_000)
p.w_alpha=0.2; p.lambda_ssim=0; p.opacity_reg=0.0; p.scale_reg=0.0
p.set_gaussians_from_numpy(**scene.initial_points); p.clear_tracking()
texs=[p.create_target_texture(t) for t in targets]
lr0=1.6e-4*extent; decay=(1e-2)**(1.0/ITERS)
rng=np.random.default_rng(0)
for it in range(1,ITERS+1):
    ci=int(rng.integers(0,n_cams)); p.set_target_texture(texs[ci])
    p.train_step(scene.images[ci].camera, lrs=AdamLRs(mean=lr0*(decay**(it-1))))
p.device.wait_for_idle()
tr=p.read_tracking()
mag=tr["grad_mean_2d_mag"]; vis=tr["visibility_count"]
n=int(p.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
mag=mag[:n]; vis=vis[:n]
seen=vis>0
avg=np.zeros(n); avg[seen]=mag[seen]/vis[seen]
a=avg[seen]
print(f"iters={ITERS} n_gauss={n} W,H={W},{H} extent={extent:.2f}")
print(f"visible gaussians={seen.sum()}  visibility_count median={np.median(vis[seen]):.0f}")
print("avg_grad (our PIXEL-unit densification signal, avg over iters visible):")
for pc in [50,75,90,99]:
    print(f"  p{pc} = {np.percentile(a,pc):.6e}")
print(f"  max = {a.max():.6e}")
gt=0.0002
print(f"gsplat threshold grow_grad2d=0.0002 (NDC units).")
print(f"  frac(our avg_grad > 0.0002) = {np.mean(a>gt):.4f}   (>this would be densified if thresholded naively)")
# If we converted our pixel-grad to gsplat NDC convention: *width/2, /(H*W*3)
ndc_like = a * (W/2.0) / (H*W*3)
print(f"our signal converted toward gsplat NDC-mean convention (*W/2 /(H*W*3)):")
for pc in [50,90,99]:
    print(f"  p{pc} = {np.percentile(ndc_like,pc):.6e}")
print(f"  frac(converted > 0.0002) = {np.mean(ndc_like>gt):.4f}")
print("DONE")
