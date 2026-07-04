"""Cross-process validation of the splat stream: spawns a writer subprocess
and a reader subprocess and checks the reader sees consistent snapshots.
Throwaway test harness."""
import sys, time, subprocess, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PY = sys.executable
NAMEFILE = os.path.join(tempfile.gettempdir(), "xproc_splat_name.txt")

WRITER = '''
import sys, time, numpy as np
sys.path.insert(0, r"{src}")
from trainer.splat_stream import SplatStreamWriter, FLOATS_PER_SPLAT
w = SplatStreamWriter(capacity=5000)
open(r"{namefile}", "w").write(w.name)
for i in range(1, 11):
    n = 1000 + (i % 5) * 500
    data = np.full(n * FLOATS_PER_SPLAT, float(i), dtype=np.float32)
    w.write_snapshot(data, n, iteration=i * 100)
    time.sleep(0.4)
w.close()
'''

READER = '''
import sys, time, numpy as np
sys.path.insert(0, r"{src}")
from trainer.splat_stream import SplatStreamReader, FLOATS_PER_SPLAT
name = open(r"{namefile}").read().strip()
r = None
for _ in range(60):
    try:
        r = SplatStreamReader(name); break
    except (FileNotFoundError, ValueError):
        time.sleep(0.1)
assert r, "could not attach"
seen = []; t0 = time.time()
while time.time() - t0 < 7:
    if r.is_closed():
        break
    snap = r.latest()
    if snap is not None:
        floats, count, it = snap
        ok = bool(np.all(floats == float(it // 100)))
        seen.append((it, count, ok))
    time.sleep(0.05)
uniq = [s for s in seen if len(s) == 3]
print("READER distinct snapshots:", len(uniq))
print("all internally consistent:", (len(uniq) > 0 and all(o for _, _, o in uniq)))
print("iters seen:", sorted(set(s[0] for s in uniq)))
r.close()
'''

src = str(Path(__file__).resolve().parent.parent / "src")
if os.path.exists(NAMEFILE):
    os.remove(NAMEFILE)
wp = subprocess.Popen([PY, "-c", WRITER.format(src=src, namefile=NAMEFILE)])
time.sleep(1.2)
rp = subprocess.run([PY, "-c", READER.format(src=src, namefile=NAMEFILE)],
                    capture_output=True, text=True, timeout=25)
print(rp.stdout)
if rp.stderr.strip():
    print("READER STDERR:", rp.stderr[-500:])
wp.wait(timeout=15)
