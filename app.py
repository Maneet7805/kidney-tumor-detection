"""Tkinter prototype (brief 7.2): browse CT slices, segment the tumor, view overlays and metrics.

Overlay colours: PREDICTED = red contour, GROUND TRUTH = green contour.
Run:  python app.py
"""
import glob
import json
import os
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import preprocess
import segment

METRICS_FILE = os.path.join("results", "metrics.json")


def load_slice(path):
    """Read one .npz slice (data contract in CLAUDE.md) into a plain dict."""
    d = np.load(path)
    return {"hu": d["hu"], "kidney": d["kidney"].astype(bool), "tumor": d["tumor"].astype(bool)}


def read_metrics():
    """Return classification metrics from results/metrics.json, or None if not available."""
    try:
        with open(METRICS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class App:
    def __init__(self, root):
        self.root = root
        root.title("KiTS23 Tumor Segmentation Prototype")
        self.files = []      # list of .npz paths
        self.index = 0
        self.dices = {}      # file path -> Dice, used for the mean

        # ---- left control panel ----
        side = ttk.Frame(root, padding=8)
        side.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Button(side, text="Load image...", command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(side, text="Load folder...", command=self.load_folder).pack(fill=tk.X, pady=2)
        nav = ttk.Frame(side)
        nav.pack(fill=tk.X, pady=6)
        ttk.Button(nav, text="< Prev", command=lambda: self.step(-1)).pack(side=tk.LEFT, expand=True)
        ttk.Button(nav, text="Next >", command=lambda: self.step(1)).pack(side=tk.LEFT, expand=True)
        self.name_label = ttk.Label(side, text="No slice loaded", wraplength=180)
        self.name_label.pack(pady=2)

        ttk.Label(side, text="Segmentation method:").pack(anchor=tk.W, pady=(10, 0))
        self.method = tk.StringVar(value="otsu")
        box = ttk.Combobox(side, textvariable=self.method, state="readonly",
                           values=["otsu", "region_growing"])
        box.pack(fill=tk.X)
        box.bind("<<ComboboxSelected>>", lambda e: self.show())

        self.metrics_label = ttk.Label(side, justify=tk.LEFT, font=("Consolas", 10))
        self.metrics_label.pack(anchor=tk.W, pady=15)

        # ---- right: matplotlib figure with 4 panels ----
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.show()

    # ---------- loading ----------
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("NumPy slice", "*.npz")], initialdir="data/slices")
        if path:
            self.set_files([path])

    def load_folder(self):
        folder = filedialog.askdirectory(initialdir="data/slices")
        if folder:
            self.set_files(sorted(glob.glob(os.path.join(folder, "*.npz"))))

    def set_files(self, files):
        self.files, self.index, self.dices = files, 0, {}
        self.show()

    def step(self, delta):
        if self.files:
            self.index = (self.index + delta) % len(self.files)
            self.show()

    # ---------- drawing ----------
    def show(self):
        self.fig.clear()
        if not self.files:
            self.fig.text(0.5, 0.5, "Load an image or a folder of .npz slices", ha="center")
            self.update_metrics(None)
            self.canvas.draw()
            return

        path = self.files[self.index]
        s = load_slice(path)
        enhanced = preprocess.enhance(s["hu"])
        pred = segment.segment_tumor(enhanced, s["kidney"], self.method.get()).astype(bool)
        gt = s["tumor"]
        dice = segment.dice(pred, gt)
        self.dices[path] = dice
        self.name_label.config(text=f"{os.path.basename(path)}\n({self.index + 1}/{len(self.files)})")

        # (title, predicted mask, ground-truth mask) for each overlay panel; None = not drawn
        panels = [("Tumor only", pred, gt),
                  ("Kidney only (ground truth)", None, s["kidney"]),
                  ("Kidney + tumor", pred, s["kidney"])]
        ax = self.fig.add_subplot(2, 2, 1)
        ax.imshow(pred, cmap="gray")
        ax.set_title(f"Segmented tumor mask ({self.method.get()})")
        ax.axis("off")
        for i, (title, red, green) in enumerate(panels, start=2):
            ax = self.fig.add_subplot(2, 2, i)
            ax.imshow(enhanced, cmap="gray")
            self.contour(ax, green, "lime")
            if red is not None:
                self.contour(ax, red, "red")
            if title.startswith("Kidney +"):
                self.contour(ax, gt, "lime")   # ground-truth tumor too
            ax.set_title(title + "  (red=pred, green=GT)")
            ax.axis("off")
        self.fig.tight_layout()
        self.canvas.draw()
        self.update_metrics(dice)

    @staticmethod
    def contour(ax, mask, colour):
        if mask.any():  # contour() warns on an empty mask
            ax.contour(mask.astype(float), levels=[0.5], colors=colour, linewidths=1.2)

    def update_metrics(self, dice):
        lines = []
        if dice is not None:
            lines.append(f"Dice (slice): {dice:.3f}")
            lines.append(f"Dice (mean):  {np.mean(list(self.dices.values())):.3f}")
            lines.append(f"  over {len(self.dices)} viewed slice(s)")
        m = read_metrics()
        lines.append("")
        if m is None:
            lines.append("Classifier metrics:\n  run run_pipeline.py\n  to create metrics.json")
        else:
            lines.append("Classifier (test set):")
            for key in ("accuracy", "f1", "recall", "specificity"):
                if key in m:
                    lines.append(f"  {key:<12}{m[key]:.3f}")
        self.metrics_label.config(text="\n".join(lines))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
