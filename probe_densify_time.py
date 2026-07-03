import sys, time, numpy as np
sys.path.insert(0,"src")
from trainer.device import create_device
from trainer.pipeline import Pipeline, DensifyConfig
import slangpy as spy

W,H = 480,270
MAXS = 5_000_000     # same as the real run's max_splats
N = 1_200_000        # representative live count
dev = create_device(backend="vulkan")
p = Pipeline(device=dev, width=W, height=H, max_splats=MAXS)
b = p.buffers

# Set count directly + fill buffers with copy_from_numpy (fast, no cursor loop).
from trainer import buffers as B
B.set_num_gaussians(b, N)
# gaussians buffer is Gaussian3D[MAXS] = 14 floats each; fill first N with plausible values
g = np.zeros((MAXS,14), np.float32)
rng = np.random.default_rng(0)
g[:N,0:3] = (rng.random((N,3))-0.5)*4
g[:N,3:6] = np.log(0.02)
g[:N,6] = 1.0
g[:N,10] = 0.1   # opacity logit-ish (sigmoid ~0.52) — enough to pass t_opacity
b.gaussians.copy_from_numpy(g.reshape(-1))
magfx = (rng.random(MAXS)*1e-3*65536).astype(np.int64)
b.grad_mean_2d_mag_accum.copy_from_numpy(magfx)
b.visibility_count_accum.copy_from_numpy(np.ones(MAXS,np.uint32))
dev.wait_for_idle()

cfg = DensifyConfig(t_opacity=0.0, t_big_scale=1e9)  # avoid pruning everything

# Warm up shader compilation.
_ = p.densify_simple(cfg, grad_thresh=0.0002, scale_thresh=0.1); dev.wait_for_idle()
B.set_num_gaussians(b, N)   # reset count for a clean second run
b.gaussians.copy_from_numpy(g.reshape(-1))
b.grad_mean_2d_mag_accum.copy_from_numpy(magfx)
b.visibility_count_accum.copy_from_numpy(np.ones(MAXS,np.uint32))
dev.wait_for_idle()

t0=time.perf_counter()
stats = p.densify_simple(cfg, grad_thresh=0.0002, scale_thresh=0.1)
dev.wait_for_idle()
print(f"TOTAL densify_simple (N={N}, max={MAXS}): {(time.perf_counter()-t0)*1000:.1f} ms")
print("counts:", {k:stats[k] for k in ('pre_count','post_split_count','final_count')})

# Now time the _scan alone at post_split_count, since that's the multi-dispatch suspect.
psc = min(stats['post_split_count'], MAXS)
dev.wait_for_idle(); t0=time.perf_counter()
p._scan(psc); dev.wait_for_idle()
print(f"_scan({psc}) alone: {(time.perf_counter()-t0)*1000:.1f} ms")
print("DONE")

# --- Test the queue-drain hypothesis: submit un-synced GPU work, then densify ---
import time as _t
B.set_num_gaussians(b, N)
b.gaussians.copy_from_numpy(g.reshape(-1))
b.grad_mean_2d_mag_accum.copy_from_numpy(magfx)
b.visibility_count_accum.copy_from_numpy(np.ones(MAXS,np.uint32))
dev.wait_for_idle()

# Queue a pile of GPU work WITHOUT syncing (mimics 500 async train steps' backlog).
NQ = 200
t0=_t.perf_counter()
for _ in range(NQ):
    p._clear_grads()          # dispatches over max_splats, no wait
submit_cpu = (_t.perf_counter()-t0)*1000
print(f"CPU time to submit {NQ} clear_grads passes (no sync): {submit_cpu:.1f} ms")

# Now densify — its first wait_for_idle must drain all that queued work.
t0=_t.perf_counter()
stats = p.densify_simple(cfg, grad_thresh=0.0002, scale_thresh=0.1)
dev.wait_for_idle()
print(f"densify_simple AFTER unsynced backlog: {(_t.perf_counter()-t0)*1000:.1f} ms")
print("DONE2")
