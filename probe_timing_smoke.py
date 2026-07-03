import sys, numpy as np, os
os.environ["TRAINER_DENSIFY_TIMING"]="1"
sys.path.insert(0,"src")
from trainer.device import create_device
from trainer.pipeline import Pipeline, DensifyConfig
from trainer import buffers as B
W,H=480,270; MAXS=5_000_000; N=1_200_000
dev=create_device(backend="vulkan")
p=Pipeline(device=dev,width=W,height=H,max_splats=MAXS); b=p.buffers
B.set_num_gaussians(b,N)
g=np.zeros((MAXS,14),np.float32); rng=np.random.default_rng(0)
g[:N,3:6]=np.log(0.02); g[:N,6]=1.0; g[:N,10]=0.1
b.gaussians.copy_from_numpy(g.reshape(-1))
b.grad_mean_2d_mag_accum.copy_from_numpy((rng.random(MAXS)*1e-3*65536).astype(np.int64))
b.visibility_count_accum.copy_from_numpy(np.ones(MAXS,np.uint32))
dev.wait_for_idle()
cfg=DensifyConfig(t_opacity=0.0,t_big_scale=1e9)
print("run1 (cold):"); p.densify_simple(cfg,grad_thresh=0.0002,scale_thresh=0.1); dev.wait_for_idle()
B.set_num_gaussians(b,N); b.gaussians.copy_from_numpy(g.reshape(-1)); dev.wait_for_idle()
print("run2 (warm):"); p.densify_simple(cfg,grad_thresh=0.0002,scale_thresh=0.1); dev.wait_for_idle()
print("OK")
