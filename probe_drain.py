import sys, time, numpy as np
sys.path.insert(0,"src")
from trainer.device import create_device
from trainer.pipeline import Pipeline, AdamLRs
from trainer.dataset import simple_camera
from trainer import buffers as B

W,H=480,270; MAXS=5_000_000; N=1_200_000
dev=create_device(backend="vulkan")
p=Pipeline(device=dev,width=W,height=H,max_splats=MAXS); b=p.buffers
rng=np.random.default_rng(0)
g=np.zeros((MAXS,14),np.float32)
g[:N,0:3]=(rng.random((N,3))-0.5)*2; g[:N,2]+=3.0
g[:N,3:6]=np.log(0.03); g[:N,6]=1.0; g[:N,10]=0.5; g[:N,11:14]=0.2
b.gaussians.copy_from_numpy(g.reshape(-1)); B.set_num_gaussians(b,N)
tgt=(rng.random((H,W,3))*0.5).astype(np.float32)
p.set_target_image(tgt)
cam=simple_camera(width=W,height=H,eye=(0,0,0),target=(0,0,1),up=(0,1,0),fovy_degrees=60)
dev.wait_for_idle()

# Warm up (compile shaders).
p.train_step(cam); dev.wait_for_idle()

for backlog in [1, 50, 200, 500]:
    # Submit `backlog` train_steps WITHOUT syncing (mimics the real loop between densifies).
    t0=time.perf_counter()
    for _ in range(backlog):
        p.train_step(cam)
    submit=time.perf_counter()-t0
    # Now the first sync of the window — this is densify's wait_for_idle.
    t0=time.perf_counter()
    dev.wait_for_idle()
    drain=time.perf_counter()-t0
    print(f"backlog={backlog:4d}  CPU submit={submit*1000:7.1f} ms   wait_for_idle drain={drain*1000:8.1f} ms   "
          f"per-iter GPU={drain/backlog*1000:.2f} ms")
print("DONE")
