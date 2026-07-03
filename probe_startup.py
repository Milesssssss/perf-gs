import sys, time, glob
sys.path.insert(0,"src")
import numpy as np
from PIL import Image

# downsample=4 default, but bicycle full-res is 4946x3286; no images_4 dir?
import os
scene="datasets/360_v2/bicycle"
print("images_4 exists:", os.path.isdir(f"{scene}/images_4"))
print("images_2 exists:", os.path.isdir(f"{scene}/images_2"))
f=sorted(glob.glob(f"{scene}/images/*"))[0]
W,H = 4946//4, 3286//4   # what downsample=4 targets
print(f"target WxH (downsample 4) = {W}x{H}")

t0=time.perf_counter(); img=Image.open(f).convert("RGB"); t_open=time.perf_counter()-t0
t0=time.perf_counter(); img_r=img.resize((W,H), Image.LANCZOS); t_resize=time.perf_counter()-t0
t0=time.perf_counter(); arr=np.asarray(img_r,dtype=np.float32)/255.0; t_arr=time.perf_counter()-t0
# srgb_to_linear
def s2l(c):
    low=c/12.92; high=np.power(np.clip((c+0.055)/1.055,1e-7,None),2.4)
    return np.where(c<=0.04045,low,high).astype(np.float32)
t0=time.perf_counter(); lin=s2l(arr); t_lin=time.perf_counter()-t0
# save gt png
t0=time.perf_counter()
gt=(arr*255+0.5).clip(0,255).astype(np.uint8)
Image.fromarray(gt,mode="RGB").save("/tmp/_gt_test.png")
t_save=time.perf_counter()-t0
print(f"per-image: open+decode={t_open*1000:.0f}ms resize_LANCZOS={t_resize*1000:.0f}ms toarray={t_arr*1000:.0f}ms srgb2lin={t_lin*1000:.0f}ms save_png={t_save*1000:.0f}ms")
tot=(t_open+t_resize+t_arr+t_lin+t_save)
print(f"per-image total ~{tot*1000:.0f}ms  x194 cams = {tot*194:.1f}s")
print("DONE")
