import sys, time, numpy as np
sys.path.insert(0,"src")
from trainer.device import create_device
from trainer.pipeline import Pipeline
from trainer import buffers as B

W,H=64,64; MAXS=100_000
dev=create_device(backend="vulkan")
p=Pipeline(device=dev,width=W,height=H,max_splats=MAXS); b=p.buffers
n=5000
means=(np.arange(n*3).reshape(n,3)*0.001).astype(np.float32)
log_scales=np.full((n,3),np.log(0.02),np.float32)
quats=np.tile(np.array([1,0,0,0],np.float32),(n,1))
opac=np.full(n,0.1,np.float32)
colors=(np.arange(n*3).reshape(n,3)*0.0001).astype(np.float32)

t0=time.perf_counter()
B.write_gaussians(b,means=means,log_scales=log_scales,rot_quats=quats,opacities=opac,color_logits=colors)
dev.wait_for_idle()
print(f"NEW write_gaussians(n={n}): {(time.perf_counter()-t0)*1000:.1f} ms")

rb=b.gaussians.to_numpy().view(np.float32).reshape(MAXS,-1)[:3,:14]
# Expected first 3 rows (from the earlier OLD-path readback)
expected=np.array([
 [0,0.001,0.002,-3.9120,-3.9120,-3.9120,1,0,0,0,-2.1972,0,0.0001,0.0002],
 [0.003,0.004,0.005,-3.9120,-3.9120,-3.9120,1,0,0,0,-2.1972,0.0003,0.0004,0.0005],
 [0.006,0.007,0.008,-3.9120,-3.9120,-3.9120,1,0,0,0,-2.1972,0.0006,0.0007,0.0008],
],np.float32)
print("matches OLD output:", np.allclose(rb, expected, atol=1e-3))
# count check
cnt=int(b.num_gaussians_counter.to_numpy().view(np.uint32)[0])
print("num_gaussians set to:", cnt, "(expect 5000)")
print("DONE")
