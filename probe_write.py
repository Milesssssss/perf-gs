import sys, time, numpy as np
sys.path.insert(0,"src")
from trainer.device import create_device
from trainer.pipeline import Pipeline
from trainer import buffers as B
import slangpy as spy

W,H=64,64; MAXS=100_000
dev=create_device(backend="vulkan")
p=Pipeline(device=dev,width=W,height=H,max_splats=MAXS); b=p.buffers

# element byte size of the Gaussian3D struct
gb = b.gaussians
print("gaussians buffer:", type(gb).__name__)
# infer element stride from to_numpy
raw = gb.to_numpy()
print("to_numpy dtype/size:", raw.dtype, raw.size, "-> floats/elem =", raw.view(np.float32).size / MAXS)

n = 5000
means = (np.arange(n*3).reshape(n,3)*0.001).astype(np.float32)
log_scales = np.full((n,3), np.log(0.02), np.float32)
quats = np.tile(np.array([1,0,0,0],np.float32),(n,1))
opac = np.full(n, 0.1, np.float32)
colors = (np.arange(n*3).reshape(n,3)*0.0001).astype(np.float32)

# --- OLD path timing ---
t0=time.perf_counter()
B.write_gaussians(b, means=means, log_scales=log_scales, rot_quats=quats, opacities=opac, color_logits=colors)
dev.wait_for_idle()
print(f"OLD write_gaussians(n={n}): {(time.perf_counter()-t0)*1000:.0f} ms")

# read back to learn layout + verify old wrote correctly
rb = gb.to_numpy().view(np.float32).reshape(MAXS, -1)[:3]
print("readback rows[0:3] (14 cols expected):")
print(np.round(rb[:, :14], 4))

# --- test copy_from_numpy PARTIAL (n rows only) ---
floats_per = gb.to_numpy().view(np.float32).size // MAXS
arr_partial = np.zeros((n, floats_per), np.float32)
arr_partial[:, 10] = 0.777  # marker in opacity slot
try:
    gb.copy_from_numpy(arr_partial.reshape(-1))
    dev.wait_for_idle()
    chk = gb.to_numpy().view(np.float32).reshape(MAXS,-1)
    print("PARTIAL copy OK. row0 opacity:", chk[0,10], " row(n) opacity (should be untouched):", chk[n,10])
except Exception as e:
    print("PARTIAL copy FAILED:", repr(e)[:120])
print("DONE")
