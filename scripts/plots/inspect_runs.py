# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Tkinter GUI comparing two v2_train.py runs (``outputs/train/<run>/``): metric curves and render | GT | render views.

Reads log.jsonl, densify_log.jsonl, final_psnr.json and cams/cam_NN/*.png, each optional.
    python scripts/plots/inspect_runs.py [--run-a DIR] [--run-b DIR] [--root outputs/train]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

COLOR_A = "#4f9dff"
COLOR_B = "#ff8a3d"
COLOR_GT = "#9aa0a6"
BG = "#1e1f22"
FG = "#e8e8e8"
GRID = "#33353a"
AXIS = "#6b6f76"

ITER_RE = re.compile(r"iter_(\d+)\.png$", re.IGNORECASE)


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    return rows


class Run:
    """Everything we can load from one ``outputs/train/<run>/`` folder."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.name = self.path.name
        self.log = _load_jsonl(self.path / "log.jsonl")
        self.densify = _load_jsonl(self.path / "densify_log.jsonl")
        self.final = _load_json(self.path / "final_psnr.json") or {}

        self._percam: dict[int, dict] = {}
        for row in self.final.get("per_cam", []):
            idx = row.get("cam_idx")
            if idx is not None:
                self._percam[int(idx)] = row

        self.cams: dict[int, dict] = {}
        cams_dir = self.path / "cams"
        if cams_dir.is_dir():
            for cam_dir in sorted(cams_dir.iterdir()):
                if not cam_dir.is_dir():
                    continue
                m = re.search(r"(\d+)$", cam_dir.name)
                if not m:
                    continue
                cam_idx = int(m.group(1))
                gt = cam_dir / "gt.png"
                iters: dict[int, Path] = {}
                for img in cam_dir.glob("iter_*.png"):
                    mi = ITER_RE.search(img.name)
                    if mi:
                        iters[int(mi.group(1))] = img
                self.cams[cam_idx] = {
                    "gt": gt if gt.exists() else None,
                    "iters": iters,
                }

    def log_series(self, key: str, x_key: str = "iter"):
        """(xs, ys) for a log.jsonl metric, dropping rows missing either value."""
        xs, ys = [], []
        for row in self.log:
            if key in row and x_key in row and row[key] is not None:
                xs.append(row[x_key])
                ys.append(row[key])
        return xs, ys

    def percam_series(self, key: str):
        """(cam_idx, value) sorted by camera index, for a final_psnr metric."""
        items = sorted(self._percam.items())
        xs = [i for i, _ in items]
        ys = [row.get(key) for _, row in items if row.get(key) is not None]
        xs = [i for i, (_, row) in zip(xs, items) if row.get(key) is not None]
        return xs, ys

    def percam(self, cam_idx: int) -> dict | None:
        return self._percam.get(cam_idx)

    def densify_iters(self) -> list[int]:
        return [e["iter"] for e in self.densify if "iter" in e]

    @property
    def avg_psnr(self):
        # Older final_psnr.json files only have avg_psnr_srgb (dB of mean MSE over all cams).
        return self.final.get("psnr_srgb", self.final.get("avg_psnr_srgb"))


METRICS = [
    ("loss (rolling)",       "log",    "loss_rolling"),
    ("loss (instant)",       "log",    "loss_instant"),
    ("PSNR (instant)",       "log",    "psnr_instant"),
    ("# gaussians",          "log",    "n_gaussians"),
    ("learning rate (mean)", "log",    "lr_mean"),
    ("iters / s",            "log",    "it_per_s"),
    ("final PSNR / camera",  "percam", "psnr_srgb"),
    ("final MSE / camera",   "percam", "mse_srgb"),
]
METRIC_BY_LABEL = {label: (src, key) for label, src, key in METRICS}


def _nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """A short list of 'round' tick values covering [lo, hi]."""
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    span = hi - lo
    raw = span / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for step in (1, 2, 2.5, 5, 10):
        if raw <= step * mag:
            step *= mag
            break
    else:
        step = 10 * mag
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            ticks.append(round(v, 12))
        v += step
    return ticks


def _fmt_tick(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-3:
        return f"{v:.1e}"
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:.3g}"
    return f"{v:.3g}"


class LineChart(tk.Canvas):
    """Overlaid line chart; ``on_pick(x)``, if set, receives the data-space x of a left click."""

    PAD_L, PAD_R, PAD_T, PAD_B = 64, 16, 34, 40

    def __init__(self, master, **kw):
        super().__init__(master, bg=BG, highlightthickness=0, **kw)
        self._series: list[dict] = []
        self._markers: list[dict] = []
        self._log_y = False
        self._x_label = "iter"
        self._title = ""
        self._plot_box = None
        self._xr = self._yr = None
        self.on_pick = None
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Button-1>", self._on_click)

    def set_series(self, series, *, x_label="iter", log_y=False, title="",
                   markers=None):
        self._series = series or []
        self._x_label = x_label
        self._log_y = log_y
        self._title = title
        self._markers = markers or []
        self.redraw()

    def _ty(self, y, ylo, yhi, y0, y1):
        if self._log_y:
            y = math.log10(max(y, 1e-12))
        t = (y - ylo) / (yhi - ylo) if yhi > ylo else 0.5
        return y1 - t * (y1 - y0)

    def _tx(self, x, xlo, xhi, x0, x1):
        t = (x - xlo) / (xhi - xlo) if xhi > xlo else 0.5
        return x0 + t * (x1 - x0)

    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 60 or h < 60:
            return
        x0, y0 = self.PAD_L, self.PAD_T
        x1, y1 = w - self.PAD_R, h - self.PAD_B
        self._plot_box = (x0, y0, x1, y1)

        drawable = [s for s in self._series if s["xs"] and s["ys"]]
        if not drawable:
            self.create_text((x0 + x1) / 2, (y0 + y1) / 2, fill=AXIS,
                             text="no data for this metric", font=("", 11))
            return

        all_x = [v for s in drawable for v in s["xs"]]
        all_y = [v for s in drawable for v in s["ys"]]
        xlo, xhi = min(all_x), max(all_x)
        ylo_d, yhi_d = min(all_y), max(all_y)

        if self._log_y:
            pos = [v for v in all_y if v > 0]
            if not pos:
                self._log_y = False
            else:
                ylo_d = min(pos)
        if yhi_d <= ylo_d:
            yhi_d = ylo_d + (abs(ylo_d) or 1.0) * 0.1
        if self._log_y:
            ylo, yhi = math.log10(ylo_d), math.log10(yhi_d)
            ylo -= (yhi - ylo) * 0.05
            yhi += (yhi - ylo) * 0.05
        else:
            pad = (yhi_d - ylo_d) * 0.06
            ylo, yhi = ylo_d - pad, yhi_d + pad
        self._xr = (xlo, xhi)
        self._yr = (ylo, yhi)

        if self._log_y:
            lo_e, hi_e = math.floor(ylo), math.ceil(yhi)
            y_ticks = [10 ** e for e in range(int(lo_e), int(hi_e) + 1)]
        else:
            y_ticks = _nice_ticks(ylo, yhi)
        for yt in y_ticks:
            py = self._ty(yt, ylo, yhi, y0, y1)
            if y0 - 1 <= py <= y1 + 1:
                self.create_line(x0, py, x1, py, fill=GRID)
                self.create_text(x0 - 6, py, anchor="e", fill=AXIS,
                                 text=_fmt_tick(yt), font=("", 8))

        for xt in _nice_ticks(xlo, xhi):
            px = self._tx(xt, xlo, xhi, x0, x1)
            if x0 - 1 <= px <= x1 + 1:
                self.create_line(px, y0, px, y1, fill=GRID)
                self.create_text(px, y1 + 6, anchor="n", fill=AXIS,
                                 text=_fmt_tick(xt), font=("", 8))

        self.create_rectangle(x0, y0, x1, y1, outline=AXIS)
        self.create_text(x0, y1 + 22, anchor="w", fill=AXIS,
                         text=self._x_label, font=("", 9))
        if self._title:
            self.create_text((x0 + x1) / 2, y0 - 18, fill=FG, text=self._title,
                             font=("", 10, "bold"))

        for mk in self._markers:
            if xlo <= mk["x"] <= xhi:
                px = self._tx(mk["x"], xlo, xhi, x0, x1)
                self.create_line(px, y0, px, y1, fill=mk.get("color", GRID),
                                 dash=(2, 3))

        for s in drawable:
            pts = []
            for x, y in zip(s["xs"], s["ys"]):
                if self._log_y and y <= 0:
                    continue
                pts.append(self._tx(x, xlo, xhi, x0, x1))
                pts.append(self._ty(y, ylo, yhi, y0, y1))
            if len(pts) >= 4:
                self.create_line(*pts, fill=s["color"], width=2, smooth=False)
            elif len(pts) == 2:
                self.create_oval(pts[0] - 3, pts[1] - 3, pts[0] + 3, pts[1] + 3,
                                 fill=s["color"], outline="")

        ly = y0 + 6
        for s in drawable:
            self.create_line(x1 - 150, ly, x1 - 130, ly, fill=s["color"], width=3)
            self.create_text(x1 - 124, ly, anchor="w", fill=FG,
                             text=s["label"], font=("", 8))
            ly += 16

    def _on_click(self, event):
        if not self.on_pick or not self._plot_box or not self._xr:
            return
        x0, y0, x1, y1 = self._plot_box
        if not (x0 <= event.x <= x1 and y0 <= event.y <= y1):
            return
        xlo, xhi = self._xr
        t = (event.x - x0) / (x1 - x0) if x1 > x0 else 0
        self.on_pick(xlo + t * (xhi - xlo))


class ImagePanel(tk.Frame):
    def __init__(self, master, title, title_color):
        super().__init__(master, bg=BG)
        self.title = title
        self._title_var = tk.StringVar(value=title)
        tk.Label(self, textvariable=self._title_var, fg=title_color, bg=BG,
                 font=("", 10, "bold")).pack(side="top", fill="x")
        self.canvas = tk.Canvas(self, bg="#111214", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.caption = tk.Label(self, text="", fg=FG, bg=BG, font=("", 9))
        self.caption.pack(side="bottom", fill="x")
        self._pil = None
        self._photo = None  # keep a ref so Tk doesn't GC it
        self.canvas.bind("<Configure>", lambda e: self._render())

    def set_title(self, text):
        self._title_var.set(text)

    def show(self, pil_img: Image.Image | None, caption: str = ""):
        self._pil = pil_img
        self.caption.config(text=caption)
        self._render()

    def _render(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 4 or ch < 4:
            return
        if self._pil is None:
            self.canvas.create_text(cw / 2, ch / 2, fill=AXIS, text="—",
                                    font=("", 20))
            self._photo = None
            return
        iw, ih = self._pil.size
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resample = Image.LANCZOS if scale < 1 else Image.NEAREST
        img = self._pil.resize((nw, nh), resample)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(cw / 2, ch / 2, image=self._photo)


class ImageCache:
    """Decodes PNGs once and hands back PIL (RGB) and numpy (float [0,1])."""

    def __init__(self, limit: int = 48):
        self._limit = limit
        self._order: list[str] = []
        self._store: dict[str, tuple[Image.Image, np.ndarray]] = {}

    def get(self, path: Path):
        key = str(path)
        if key in self._store:
            self._order.remove(key)
            self._order.append(key)
            return self._store[key]
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            return None
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        self._store[key] = (pil, arr)
        self._order.append(key)
        while len(self._order) > self._limit:
            self._store.pop(self._order.pop(0), None)
        return self._store[key]


def _psnr(a: np.ndarray, b: np.ndarray) -> float | None:
    """PSNR between two [0,1] RGB arrays; resizes b to a's shape if needed."""
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        try:
            b_img = Image.fromarray((b * 255).astype(np.uint8)).resize(
                (a.shape[1], a.shape[0]), Image.LANCZOS)
            b = np.asarray(b_img, dtype=np.float32) / 255.0
        except Exception:
            return None
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return -10.0 * math.log10(mse)


def _error_image(render: np.ndarray, gt: np.ndarray, amp: float) -> Image.Image:
    if render.shape != gt.shape:
        gt_img = Image.fromarray((gt * 255).astype(np.uint8)).resize(
            (render.shape[1], render.shape[0]), Image.LANCZOS)
        gt = np.asarray(gt_img, dtype=np.float32) / 255.0
    err = np.clip(np.abs(render - gt) * amp, 0.0, 1.0)
    return Image.fromarray((err * 255).astype(np.uint8), mode="RGB")


class InspectorApp(tk.Tk):
    def __init__(self, root_dir: Path, run_a: Path | None, run_b: Path | None):
        super().__init__()
        self.title("3DGS training-run inspector")
        self.geometry("1560x880")
        self.configure(bg=BG)

        self.root_dir = root_dir
        self.run_a: Run | None = Run(run_a) if run_a else None
        self.run_b: Run | None = Run(run_b) if run_b else None
        self.cache = ImageCache()

        self._build_style()
        self._build_toolbar()
        self._build_body()

        self.refresh_all()

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=FG, fieldbackground="#2a2c31")
        style.configure("TButton", padding=4)
        style.configure("TCombobox", padding=2)
        style.map("TCombobox", fieldbackground=[("readonly", "#2a2c31")])

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=BG)
        bar.pack(side="top", fill="x", padx=8, pady=6)

        self.a_var = tk.StringVar(value=self._label_for(self.run_a))
        self.b_var = tk.StringVar(value=self._label_for(self.run_b))

        tk.Label(bar, text="Run A", fg=COLOR_A, bg=BG, font=("", 10, "bold")
                 ).grid(row=0, column=0, sticky="w", padx=(0, 4))
        tk.Label(bar, textvariable=self.a_var, fg=FG, bg="#2a2c31", anchor="w",
                 width=52).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(bar, text="Browse…", command=lambda: self._browse("a")
                   ).grid(row=0, column=2, padx=2)

        tk.Label(bar, text="Run B", fg=COLOR_B, bg=BG, font=("", 10, "bold")
                 ).grid(row=1, column=0, sticky="w", padx=(0, 4))
        tk.Label(bar, textvariable=self.b_var, fg=FG, bg="#2a2c31", anchor="w",
                 width=52).grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(bar, text="Browse…", command=lambda: self._browse("b")
                   ).grid(row=1, column=2, padx=2)

        ttk.Button(bar, text="Swap A/B", command=self._swap
                   ).grid(row=0, column=3, rowspan=2, padx=(12, 4))
        ttk.Button(bar, text="Reload", command=self._reload
                   ).grid(row=0, column=4, rowspan=2, padx=4)

        self.summary = tk.Label(bar, text="", fg=AXIS, bg=BG, anchor="w",
                                justify="left", font=("", 9))
        self.summary.grid(row=0, column=5, rowspan=2, sticky="w", padx=16)
        bar.columnconfigure(1, weight=1)

    def _label_for(self, run: Run | None) -> str:
        return run.name if run else "(none — click Browse…)"

    def _build_body(self):
        paned = tk.PanedWindow(self, orient="horizontal", bg=BG, sashwidth=6,
                               sashrelief="flat", bd=0)
        paned.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 8))

        left = tk.Frame(paned, bg=BG)
        ctl = tk.Frame(left, bg=BG)
        ctl.pack(side="top", fill="x", pady=(0, 4))

        tk.Label(ctl, text="Metric", fg=FG, bg=BG).pack(side="left")
        self.metric_var = tk.StringVar(value=METRICS[0][0])
        mcb = ttk.Combobox(ctl, textvariable=self.metric_var, state="readonly",
                           width=20, values=[m[0] for m in METRICS])
        mcb.pack(side="left", padx=4)
        mcb.bind("<<ComboboxSelected>>", lambda e: self.refresh_chart())

        tk.Label(ctl, text="X", fg=FG, bg=BG).pack(side="left", padx=(10, 0))
        self.xaxis_var = tk.StringVar(value="iter")
        xcb = ttk.Combobox(ctl, textvariable=self.xaxis_var, state="readonly",
                           width=10, values=["iter", "elapsed_s"])
        xcb.pack(side="left", padx=4)
        xcb.bind("<<ComboboxSelected>>", lambda e: self.refresh_chart())

        self.logy_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctl, text="log Y", variable=self.logy_var, fg=FG, bg=BG,
                       selectcolor=BG, activebackground=BG, activeforeground=FG,
                       command=self.refresh_chart).pack(side="left", padx=8)
        self.densify_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctl, text="mark densify", variable=self.densify_var,
                       fg=FG, bg=BG, selectcolor=BG, activebackground=BG,
                       activeforeground=FG, command=self.refresh_chart
                       ).pack(side="left", padx=4)

        self.chart = LineChart(left, width=560)
        self.chart.pack(side="top", fill="both", expand=True)
        self.chart.on_pick = self._chart_picked
        paned.add(left, minsize=420, width=640)

        right = tk.Frame(paned, bg=BG)
        cc = tk.Frame(right, bg=BG)
        cc.pack(side="top", fill="x", pady=(0, 4))

        tk.Label(cc, text="Camera", fg=FG, bg=BG).pack(side="left")
        self.cam_var = tk.StringVar()
        self.cam_cb = ttk.Combobox(cc, textvariable=self.cam_var,
                                   state="readonly", width=16)
        self.cam_cb.pack(side="left", padx=4)
        self.cam_cb.bind("<<ComboboxSelected>>", lambda e: self._on_cam_change())

        self.error_var = tk.BooleanVar(value=False)
        tk.Checkbutton(cc, text="error ×", variable=self.error_var, fg=FG, bg=BG,
                       selectcolor=BG, activebackground=BG, activeforeground=FG,
                       command=self.refresh_images).pack(side="left", padx=(12, 0))
        self.amp_var = tk.IntVar(value=5)
        tk.Scale(cc, from_=1, to=30, orient="horizontal", variable=self.amp_var,
                 bg=BG, fg=FG, highlightthickness=0, troughcolor="#2a2c31",
                 length=90, command=lambda v: self.refresh_images()
                 ).pack(side="left")

        it_frame = tk.Frame(right, bg=BG)
        it_frame.pack(side="top", fill="x")
        tk.Label(it_frame, text="Iteration", fg=FG, bg=BG).pack(side="left")
        self.iter_label = tk.Label(it_frame, text="—", fg=FG, bg=BG, width=10)
        self.iter_label.pack(side="right")
        self._iters: list[int] = []
        self.iter_scale = tk.Scale(it_frame, from_=0, to=0, orient="horizontal",
                                   showvalue=False, bg=BG, fg=FG,
                                   highlightthickness=0, troughcolor="#2a2c31",
                                   command=self._on_iter_scale)
        self.iter_scale.pack(side="left", fill="x", expand=True, padx=6)

        panels = tk.Frame(right, bg=BG)
        panels.pack(side="top", fill="both", expand=True)
        self.panel_a = ImagePanel(panels, "Run A", COLOR_A)
        self.panel_gt = ImagePanel(panels, "Ground truth", COLOR_GT)
        self.panel_b = ImagePanel(panels, "Run B", COLOR_B)
        # Uniform grid columns give exact thirds; pack/expand splits only leftover space and squeezes panels.
        panels.rowconfigure(0, weight=1)
        for col, p in enumerate((self.panel_a, self.panel_gt, self.panel_b)):
            p.grid(row=0, column=col, sticky="nsew", padx=2)
            panels.columnconfigure(col, weight=1, uniform="imgcol")

        paned.add(right, minsize=520)

    def _browse(self, which: str):
        init = str(self.root_dir if self.root_dir.exists() else Path.cwd())
        chosen = filedialog.askdirectory(title=f"Select run folder ({which.upper()})",
                                         initialdir=init)
        if not chosen:
            return
        run = Run(Path(chosen))
        if which == "a":
            self.run_a = run
            self.a_var.set(self._label_for(run))
        else:
            self.run_b = run
            self.b_var.set(self._label_for(run))
        self.refresh_all()

    def _swap(self):
        self.run_a, self.run_b = self.run_b, self.run_a
        self.a_var.set(self._label_for(self.run_a))
        self.b_var.set(self._label_for(self.run_b))
        self.refresh_all()

    def _reload(self):
        if self.run_a:
            self.run_a = Run(self.run_a.path)
        if self.run_b:
            self.run_b = Run(self.run_b.path)
        self.cache = ImageCache()
        self.refresh_all()

    def refresh_all(self):
        self._update_summary()
        self._rebuild_camera_list()
        self.refresh_chart()
        self.refresh_images()

    def _update_summary(self):
        parts = []
        for tag, run in (("A", self.run_a), ("B", self.run_b)):
            if not run:
                continue
            bits = [f"{tag}: {run.name}"]
            if run.avg_psnr is not None:
                bits.append(f"avg PSNR {run.avg_psnr:.2f} dB")
            if run.log:
                bits.append(f"{run.log[-1].get('n_gaussians', '?')} splats @ last log")
            parts.append("   ".join(bits))
        self.summary.config(text="\n".join(parts))

    def _runs(self):
        return [r for r in (self.run_a, self.run_b) if r]

    def refresh_chart(self):
        src, key = METRIC_BY_LABEL[self.metric_var.get()]
        series = []
        markers = []
        if src == "log":
            x_key = self.xaxis_var.get()
            x_label = "elapsed (s)" if x_key == "elapsed_s" else "iteration"
            for run, color in ((self.run_a, COLOR_A), (self.run_b, COLOR_B)):
                if not run:
                    continue
                xs, ys = run.log_series(key, x_key)
                series.append({"label": run.name, "xs": xs, "ys": ys,
                               "color": color})
            if self.densify_var.get() and x_key == "iter":
                for run, color in ((self.run_a, COLOR_A), (self.run_b, COLOR_B)):
                    if not run:
                        continue
                    for it in run.densify_iters():
                        markers.append({"x": it, "color": color})
        else:
            x_label = "camera index"
            for run, color in ((self.run_a, COLOR_A), (self.run_b, COLOR_B)):
                if not run:
                    continue
                xs, ys = run.percam_series(key)
                series.append({"label": run.name, "xs": xs, "ys": ys,
                               "color": color})

        self.chart.set_series(series, x_label=x_label,
                              log_y=self.logy_var.get(),
                              title=self.metric_var.get(), markers=markers)

    def _chart_picked(self, x_value):
        src, _ = METRIC_BY_LABEL[self.metric_var.get()]
        if src != "percam" or not self._cam_indices:
            return
        nearest = min(self._cam_indices, key=lambda c: abs(c - x_value))
        self.cam_var.set(self._cam_label(nearest))
        self._on_cam_change()

    def _rebuild_camera_list(self):
        idxs = set()
        for run in self._runs():
            idxs.update(run.cams.keys())
        self._cam_indices = sorted(idxs)
        labels = [self._cam_label(i) for i in self._cam_indices]
        self.cam_cb.config(values=labels)
        if labels:
            if self.cam_var.get() not in labels:
                self.cam_var.set(labels[0])
        else:
            self.cam_var.set("")
        self._rebuild_iter_list()

    def _cam_label(self, cam_idx: int) -> str:
        for run in self._runs():
            row = run.percam(cam_idx)
            if row and row.get("image_name"):
                return f"cam {cam_idx:03d}  {row['image_name']}"
        return f"cam {cam_idx:03d}"

    def _selected_cam_idx(self) -> int | None:
        label = self.cam_var.get()
        if not label:
            return None
        m = re.search(r"cam\s+(\d+)", label)
        return int(m.group(1)) if m else None

    def _on_cam_change(self):
        self._rebuild_iter_list()
        self.refresh_images()

    def _rebuild_iter_list(self):
        cam_idx = self._selected_cam_idx()
        iters = set()
        if cam_idx is not None:
            for run in self._runs():
                cam = run.cams.get(cam_idx)
                if cam:
                    iters.update(cam["iters"].keys())
        self._iters = sorted(iters)
        n = len(self._iters)
        if n == 0:
            self.iter_scale.config(from_=0, to=0, state="disabled")
            self.iter_label.config(text="—")
            return
        self.iter_scale.config(from_=0, to=n - 1, state="normal")
        self.iter_scale.set(n - 1)
        self.iter_label.config(text=f"{self._iters[-1]:,}")

    def _on_iter_scale(self, _value):
        if self._iters:
            idx = int(float(_value))
            idx = max(0, min(idx, len(self._iters) - 1))
            self.iter_label.config(text=f"{self._iters[idx]:,}")
        self.refresh_images()

    def _current_iter(self) -> int | None:
        if not self._iters:
            return None
        idx = int(self.iter_scale.get())
        idx = max(0, min(idx, len(self._iters) - 1))
        return self._iters[idx]

    def refresh_images(self):
        cam_idx = self._selected_cam_idx()
        it = self._current_iter()

        gt_entry = None
        for run in self._runs():
            cam = run.cams.get(cam_idx) if cam_idx is not None else None
            if cam and cam.get("gt"):
                gt_entry = self.cache.get(cam["gt"])
                if gt_entry:
                    break
        gt_arr = gt_entry[1] if gt_entry else None
        self.panel_gt.show(gt_entry[0] if gt_entry else None,
                           "" if gt_entry else "no gt.png")

        amp = float(self.amp_var.get())
        show_err = self.error_var.get()

        for run, panel, tag in ((self.run_a, self.panel_a, "Run A"),
                                (self.run_b, self.panel_b, "Run B")):
            panel.set_title(tag if not run else f"{tag} — {run.name}")
            if not run or cam_idx is None:
                panel.show(None, "")
                continue
            cam = run.cams.get(cam_idx)
            path = cam["iters"].get(it) if cam else None
            if path is None:
                panel.show(None, "no render at this iter")
                continue
            entry = self.cache.get(path)
            if entry is None:
                panel.show(None, "failed to load")
                continue
            pil, arr = entry
            psnr = _psnr(arr, gt_arr)
            cap = f"iter {it:,}"
            if psnr is not None:
                cap += f"   PSNR {psnr:.2f} dB" if math.isfinite(psnr) else "   PSNR ∞"
            if show_err and gt_arr is not None:
                panel.show(_error_image(arr, gt_arr, amp), cap + f"   (error ×{int(amp)})")
            else:
                panel.show(pil, cap)


def _auto_pick_recent(root: Path, n: int = 2) -> list[Path]:
    """Newest ``n`` run dirs under ``root`` that have a log.jsonl or cams/ folder."""
    if not root.is_dir():
        return []
    cand = [d for d in root.iterdir() if d.is_dir()
            and ((d / "log.jsonl").exists() or (d / "cams").is_dir())]
    cand.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return cand[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect & compare 3DGS training runs")
    ap.add_argument("--root", type=str, default="outputs/train",
                    help="folder the Browse dialogs and auto-pick start from")
    ap.add_argument("--run-a", type=str, default=None)
    ap.add_argument("--run-b", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    run_a = Path(args.run_a) if args.run_a else None
    run_b = Path(args.run_b) if args.run_b else None
    if run_a is None and run_b is None:
        recent = _auto_pick_recent(root)
        if len(recent) >= 1:
            run_a = recent[0]
        if len(recent) >= 2:
            run_b = recent[1]

    app = InspectorApp(root_dir=root, run_a=run_a, run_b=run_b)
    app.mainloop()


if __name__ == "__main__":
    main()
