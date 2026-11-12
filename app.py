"""Tkinter prototype (brief 7.2). One tab per phase of the assignment workflow:

  Raw CT slice -> Phase 1 Enhancement -> Phase 2 Segmentation -> Phase 3 Recognition -> Diagnostic class

Overlay colours: PREDICTED = red contour, GROUND TRUTH = green contour.
Run:  python app.py   (run run_pipeline.py once first: it creates results/metrics.json and the feature table)
"""
import glob
import json
import os
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from skimage.measure import label

import classify
import preprocess
import segment

METRICS_FILE = os.path.join("results", "metrics.json")
FEATURE_FILE = os.path.join("results", "glcm_feature_table.csv")
TABS = ["Overview", "Phase 1: Enhancement", "Phase 2: Segmentation", "Phase 3: Recognition"]


def load_slice(path):
    """Read one .npz slice (data contract in CLAUDE.md) into a plain dict."""
    d = np.load(path)
    return {"hu": d["hu"], "kidney": d["kidney"].astype(bool), "tumor": d["tumor"].astype(bool)}


def load_optional(loader, path):
    """Metrics / model files only exist after run_pipeline.py; return None if missing."""
    try:
        return loader(path)
    except (OSError, ValueError):
        return None


def train_classifier(path):
    """Rebuild the Random Forest from the saved feature table (same data and seed as run_pipeline.py).
    Retraining takes about a second and avoids loading a pickle file, which could run arbitrary code."""
    table = pd.read_csv(path)
    X = table[classify.FEATURE_NAMES].to_numpy()
    return classify.train_and_evaluate(X, table["tumor"].to_numpy(), table["case"].to_numpy())["model"]


def read_json(path):
    with open(path) as f:
        return json.load(f)


def contour(ax, mask, colour):
    if mask.any():  # contour() warns on an empty mask
        ax.contour(mask.astype(float), levels=[0.5], colors=colour, linewidths=1.2)


class App:
    def __init__(self, root):
        self.root = root
        root.title("KiTS23 Tumor Detection Prototype")
        self.files, self.index, self.dices = [], 0, {}
        self.metrics = load_optional(read_json, METRICS_FILE)
        self.model = load_optional(train_classifier, FEATURE_FILE)

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
        box = ttk.Combobox(side, textvariable=self.method, state="readonly", values=["otsu", "region_growing"])
        box.pack(fill=tk.X)
        box.bind("<<ComboboxSelected>>", lambda e: self.show())
        self.info_label = ttk.Label(side, justify=tk.LEFT, font=("Consolas", 10))
        self.info_label.pack(anchor=tk.W, pady=15)

        # ---- right: one tab (matplotlib figure) per phase ----
        notebook = ttk.Notebook(root)
        notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvases = {}
        for name in TABS:
            fig = Figure(figsize=(11, 8))
            canvas = FigureCanvasTkAgg(fig, master=notebook)
            notebook.add(canvas.get_tk_widget(), text=name)
            self.canvases[name] = canvas
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

    # ---------- run the pipeline on the current slice and redraw every tab ----------
    def show(self):
        for canvas in self.canvases.values():
            canvas.figure.clear()
        if not self.files:
            for canvas in self.canvases.values():
                canvas.figure.text(0.5, 0.5, "Load an image or a folder of .npz slices", ha="center")
                canvas.draw()
            self.info_label.config(text="")
            return

        path = self.files[self.index]
        s = load_slice(path)
        base = preprocess.window_normalize(s["hu"])
        enhanced = preprocess.enhance(s["hu"])
        preds = {m: segment.segment_tumor(enhanced, s["kidney"], m).astype(bool) for m in ("otsu", "region_growing")}
        pred = preds[self.method.get()]
        dice = segment.dice(pred, s["tumor"])
        self.dices[path] = dice

        ys, xs = np.where(s["kidney"])  # crop every picture to the kidneys so details are visible
        box = (slice(max(ys.min() - 30, 0), ys.max() + 30), slice(max(xs.min() - 30, 0), xs.max() + 30))
        ctx = dict(s=s, base=base, enhanced=enhanced, preds=preds, pred=pred, box=box)

        self.draw_overview(self.canvases[TABS[0]].figure, ctx)
        self.draw_phase1(self.canvases[TABS[1]].figure, ctx)
        self.draw_phase2(self.canvases[TABS[2]].figure, ctx)
        diagnosis = self.draw_phase3(self.canvases[TABS[3]].figure, ctx)
        for name, canvas in self.canvases.items():
            if name != TABS[3]:  # the Phase 3 tab has a manually placed inset, so no tight_layout there
                canvas.figure.tight_layout()
            canvas.draw()
        self.name_label.config(text=f"{os.path.basename(path)}\n({self.index + 1}/{len(self.files)})")
        self.update_info(dice, diagnosis)

    # ---------- Overview: the 4 views required by brief 7.2 ----------
    def draw_overview(self, fig, c):
        s, enhanced, pred = c["s"], c["enhanced"], c["pred"]
        ax = fig.add_subplot(2, 2, 1)
        ax.imshow(pred, cmap="gray")
        ax.set_title(f"Segmented tumor mask ({self.method.get()})")
        views = [("Tumor only", pred, s["tumor"]), ("Kidney only (ground truth)", None, s["kidney"]),
                 ("Kidney + tumor", pred, s["kidney"])]
        for i, (title, red, green) in enumerate(views, start=2):
            ax = fig.add_subplot(2, 2, i)
            ax.imshow(enhanced, cmap="gray")
            contour(ax, green, "lime")
            if red is not None:
                contour(ax, red, "red")
            if title == "Kidney + tumor":
                contour(ax, s["tumor"], "lime")
            ax.set_title(title + "  (red=pred, green=GT)")
        for ax in fig.axes:
            ax.axis("off")

    # ---------- Phase 1: original -> enhanced, with every compared algorithm ----------
    def draw_phase1(self, fig, c):
        s, base, box = c["s"], c["base"], c["box"]
        roi = s["kidney"]
        ring = np.zeros_like(roi)
        ring[box] = True
        ring &= ~roi

        def score(img):  # PSNR vs the windowed image, CNR between kidney and surrounding tissue
            return f"PSNR {preprocess.psnr(base, img):.1f} dB | CNR {preprocess.cnr(img, roi, ring):.2f}"

        raw = np.clip(s["hu"], -1000, 1000)
        panels = [
            ("ORIGINAL raw CT (full HU range)", raw, ""),
            ("Step 1: window [-160,240] HU -> 0-255", base, f"CNR {preprocess.cnr(base, roi, ring):.2f}"),
            ("Step 2a: anisotropic diffusion (CHOSEN)", preprocess.denoise(base, "anisotropic"), None),
            ("Step 2b: Gaussian", preprocess.denoise(base, "gaussian"), None),
            ("Step 2c: median", preprocess.denoise(base, "median"), None),
            ("Step 3a: CLAHE (compared, not in final chain)", preprocess.enhance_contrast(base, "clahe"), None),
            ("Step 3b: histogram equalization", preprocess.enhance_contrast(base, "he"), None),
            ("Step 4a: log-magnitude 2D DFT", preprocess.log_spectrum(base), "full image"),
            ("Step 4b: Butterworth low-pass (CHOSEN)", preprocess.butterworth_lowpass(base), None),
            ("PHASE 1 OUTPUT (chain: 1 -> 2a -> 4b)", c["enhanced"], None),
        ]
        for i, (title, img, note) in enumerate(panels, start=1):
            ax = fig.add_subplot(3, 4, i)
            ax.imshow(img if note in ("full image", "") else img[box], cmap="gray")
            note = score(img) if note is None else ("" if note == "full image" else note)
            ax.set_title(f"{title}\n{note}".strip(), fontsize=8)
            ax.axis("off")

    # ---------- Phase 2: both segmentation strategies and their overlap scores ----------
    def draw_phase2(self, fig, c):
        s, enhanced, box = c["s"], c["enhanced"], c["box"]
        raw = segment.segment_otsu_raw(enhanced, s["kidney"])
        ax = fig.add_subplot(2, 3, 1)
        ax.imshow(enhanced[box], cmap="gray")
        contour_box = s["kidney"][box]
        contour(ax, contour_box, "yellow")
        ax.set_title("Input: Phase 1 output\n(yellow = kidney search region)", fontsize=9)
        ax = fig.add_subplot(2, 3, 2)
        ax.imshow(raw[box], cmap="gray")
        ax.set_title("Multi-Otsu: darkest class\n(before morphology)", fontsize=9)
        ax = fig.add_subplot(2, 3, 3)
        ax.imshow(s["tumor"][box], cmap="gray")
        ax.set_title("Ground truth tumor mask", fontsize=9)
        for k, (name, key) in enumerate([("Strategy A: Multi-Otsu + morphology", "otsu"),
                                         ("Strategy B: seeded region growing", "region_growing")]):
            pred = c["preds"][key]
            ax = fig.add_subplot(2, 3, 4 + k)
            ax.imshow(enhanced[box], cmap="gray")
            contour(ax, s["tumor"][box], "lime")
            contour(ax, pred[box], "red")
            ax.set_title(f"{name}\nDice {segment.dice(pred, s['tumor']):.3f} | IoU {segment.iou(pred, s['tumor']):.3f}"
                         "\n(red=pred, green=GT)", fontsize=9)
        for ax in fig.axes:
            ax.axis("off")

    # ---------- Phase 3: features -> classifier -> diagnostic class ----------
    def draw_phase3(self, fig, c):
        """Returns the slice-level diagnosis string (also shown in the side panel)."""
        s, enhanced, pred, box = c["s"], c["enhanced"], c["pred"], c["box"]
        parts = label(pred)
        regions = [parts == i for i in range(1, parts.max() + 1) if (parts == i).sum() >= segment.MIN_TUMOR_PIXELS]

        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(enhanced[box], cmap="gray")
        contour(ax, s["tumor"][box], "lime")
        lines, diagnosis = [], "No candidate region found"
        for n, region in enumerate(regions, start=1):
            contour(ax, region[box], "red")
            ys, xs = np.where(region[box])
            ax.text(xs.mean(), ys.min() - 3, f"R{n}", color="red", ha="center", fontsize=10)
            features = classify.extract_features(enhanced, s["hu"], region)
            truth = "TUMOR/CYST" if s["tumor"][region].mean() >= 0.5 else "HEALTHY"
            if self.model is None:
                lines.append(f"R{n}: run run_pipeline.py first (no classifier yet)")
                continue
            prob = self.model.predict_proba(features[None, :])[0, 1]
            label_text = "TUMOR/CYST" if prob >= 0.5 else "HEALTHY"
            lines.append(f"R{n}: classifier says {label_text} (p={prob:.2f})  | ground truth: {truth}")
            if prob >= 0.5:
                diagnosis = "TUMOR/CYST detected"
            elif diagnosis == "No candidate region found":
                diagnosis = "Healthy (no tumor found)"
        ax.set_title("Candidate regions (red) vs GT tumor (green)", fontsize=9)
        ax.set_anchor("N")
        ax.axis("off")

        # middle: the feature values of the first region (the classifier's input vector)
        ax = fig.add_subplot(1, 3, 2)
        ax.axis("off")
        if regions:
            f = classify.extract_features(enhanced, s["hu"], regions[0])
            text = "Feature vector of R1\n(input to Random Forest)\n\n" + "\n".join(
                f"{n:<21}{v:>10.3f}" for n, v in zip(classify.FEATURE_NAMES, f))
        else:
            text = "No region to describe"
        ax.text(0, 1, text, va="top", family="monospace", fontsize=8)

        # right: DIAGNOSTIC CLASS and the overall test-set confusion matrix
        ax = fig.add_subplot(1, 3, 3)
        ax.axis("off")
        ax.text(0, 1, "DIAGNOSTIC CLASS:\n" + diagnosis + "\n\n" + "\n".join(lines), va="top", fontsize=9, wrap=True)
        if self.metrics:
            cm = np.array(self.metrics["confusion_matrix"])
            inset = fig.add_axes([0.72, 0.1, 0.22, 0.3])
            inset.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                inset.text(j, i, v, ha="center", va="center", fontsize=12)
            inset.set_xticks([0, 1], ["healthy", "tumor"], fontsize=8)
            inset.set_yticks([0, 1], ["healthy", "tumor"], fontsize=8)
            inset.set_xlabel("predicted", fontsize=8)
            inset.set_ylabel("actual", fontsize=8)
            inset.set_title("Confusion matrix (test set)", fontsize=8)
        return diagnosis

    def update_info(self, dice, diagnosis):
        lines = [f"Diagnosis:\n  {diagnosis}", "", f"Dice (slice): {dice:.3f}",
                 f"Dice (mean):  {np.mean(list(self.dices.values())):.3f}", f"  over {len(self.dices)} viewed slice(s)", ""]
        m = self.metrics
        if m is None:
            lines.append("Classifier metrics:\n  run run_pipeline.py\n  to create metrics.json")
        else:
            lines.append("Classifier (test set):")
            lines += [f"  {k:<12}{m[k]:.3f}" for k in ("accuracy", "f1", "recall", "specificity") if k in m]
        self.info_label.config(text="\n".join(lines))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
