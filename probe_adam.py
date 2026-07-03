import sys, time, numpy as np
sys.path.insert(0,"src")
from trainer.device import create_device
from trainer.pipeline import Pipeline
from trainer import buffers as B

W,H=64,64; MAXS=50_000  # small so the OLD loop finishes; cost scales linearly
dev=create_device(backend="vulkan")
p=Pipeline(device=dev,width=W,height=H,max_splats=MAXS); b=p.buffers
raw=b.adam_state.to_numpy()
print("adam_state raw bytes:", raw.size, "-> bytes/elem:", raw.size/MAXS, "floats/elem:", raw.view(np.float32).size/MAXS)

t0=time.perf_counter(); B.zero_adam_state(b); dev.wait_for_idle()
print(f"OLD zero_adam_state(max={MAXS}): {(time.perf_counter()-t0)*1000:.0f} ms  -> at 5M would be ~{(time.perf_counter()-t0)*5_000_000/MAXS:.0f} ms")

# New approach: single zeroed copy_from_numpy of the whole raw byte buffer.
t0=time.perf_counter()
zeros=np.zeros(raw.view(np.float32).size, np.float32)
b.adam_state.copy_from_numpy(zeros); dev.wait_for_idle()
print(f"NEW zero via copy_from_numpy: {(time.perf_counter()-t0)*1000:.1f} ms")
# verify all zero
chk=b.adam_state.to_numpy().view(np.float32)
print("all zero:", bool(np.all(chk==0)))
print("DONE")
