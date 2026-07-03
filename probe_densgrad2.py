import sys, numpy as np
sys.path.insert(0,"src")
from trainer.colmap import load_colmap, load_image
from trainer.device import create_device
from trainer.pipeline import AdamLRs, Pipeline

ITERS=600
scene=load_colmap("datasets/synthetic/suzanne_blender",downsample=4)
n_cams=len(scene.images); extent=scene.scene_extent
W,H=scene.images[0].camera.image_size
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
n=int(p.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
mag=tr["grad_mean_2d_mag"][:n]; vis=tr["visibility_count"][:n]
seen=vis>0
a=(mag[seen]/vis[seen])
print(f"NDC-scaled avg_grad (accumulator units, post-fix):")
for pc in [50,90,99]: print(f"  p{pc}={np.percentile(a,pc):.4f}")
# host converts gsplat threshold up by W*H*3
gt=0.0002; internal=gt*(W*H*3)
print(f"gsplat grad_thresh=0.0002 -> internal threshold {internal:.2f} (=0.0002*W*H*3)")
print(f"  frac(avg_grad > internal) = {np.mean(a>internal):.4f}  (target: a sane selective fraction, not ~1.0 or ~0)")
print("DONE")
