"""End-to-end run on all extracted slices. Writes tables, figures and metrics.json to results/."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

import classify
import preprocess
import segment

SLICE_DIR, OUT = "data/slices", "results"
METHODS = ["otsu", "region_growing"]


def phase1_table(slices):
    """Compare denoising and contrast methods with PSNR (vs windowed image) and CNR."""
    rows = []
    for d in slices:
        base = preprocess.window_normalize(d["hu"])
        roi = d["kidney"]
        ring = ndi.binary_dilation(roi, iterations=15) & ~ndi.binary_dilation(roi, iterations=4)
        variants = {f"denoise_{m}": preprocess.denoise(base, m) for m in ("anisotropic", "gaussian", "median")}
        variants["contrast_clahe"] = preprocess.enhance_contrast(base, "clahe")
        variants["contrast_he"] = preprocess.enhance_contrast(base, "he")
        variants["butterworth"] = preprocess.butterworth_lowpass(base)
        variants["full_chain"] = preprocess.enhance(d["hu"])
        variants["original_window"] = base
        for name, img in variants.items():
            rows.append({"method": name, "psnr": preprocess.psnr(base, img), "cnr": preprocess.cnr(img, roi, ring)})
    table = pd.DataFrame(rows).replace(np.inf, np.nan).groupby("method").mean().round(3)
    table.to_csv(f"{OUT}/phase1_psnr_cnr.csv")
    return table


def phase2_table(slices):
    """Dice / IoU of both strategies against the ground-truth tumor mask."""
    rows = []
    for d in slices:
        enhanced = preprocess.enhance(d["hu"])
        for m in METHODS:
            pred = segment.segment_tumor(enhanced, d["kidney"], m)
            rows.append({"case": int(d["case"]), "z": int(d["z"]), "method": m,
                         "dice": segment.dice(pred, d["tumor"]), "iou": segment.iou(pred, d["tumor"])})
    table = pd.DataFrame(rows)
    table.to_csv(f"{OUT}/phase2_per_slice.csv", index=False)
    return table


def save_figures(slices, method):
    """Phase 1 comparison and overlay figures (red = predicted, green = ground truth)."""
    d = max(slices, key=lambda s: s["tumor"].sum())
    base = preprocess.window_normalize(d["hu"])
    panels = {"windowed": base, "anisotropic": preprocess.denoise(base, "anisotropic"),
              "gaussian": preprocess.denoise(base, "gaussian"), "median": preprocess.denoise(base, "median"),
              "CLAHE": preprocess.enhance_contrast(base, "clahe"), "HE": preprocess.enhance_contrast(base, "he"),
              "log spectrum": preprocess.log_spectrum(base), "full chain": preprocess.enhance(d["hu"])}
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, (title, img) in zip(axes.ravel(), panels.items()):
        ax.imshow(img, cmap="gray"); ax.set_title(title); ax.axis("off")
    fig.savefig(f"{OUT}/phase1_comparison.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, d in zip(axes.ravel(), sorted(slices, key=lambda s: -s["tumor"].sum())[:8]):
        enhanced = preprocess.enhance(d["hu"])
        pred = segment.segment_tumor(enhanced, d["kidney"], method)
        ys, xs = np.where(d["kidney"])
        ax.imshow(enhanced, cmap="gray")
        ax.contour(d["tumor"], [0.5], colors="lime"); ax.contour(pred, [0.5], colors="red")
        ax.set_xlim(xs.min() - 25, xs.max() + 25); ax.set_ylim(ys.max() + 25, ys.min() - 25)
        ax.set_title(f"case {int(d['case'])} z{int(d['z'])}  Dice {segment.dice(pred, d['tumor']):.2f}"); ax.axis("off")
    fig.savefig(f"{OUT}/overlays_{method}.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def save_confusion_matrix(cm):
    """Confusion matrix figure for the held-out test regions."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(np.array(cm)):
        ax.text(j, i, v, ha="center", va="center", fontsize=16)
    ax.set_xticks([0, 1], ["healthy", "tumor"]); ax.set_yticks([0, 1], ["healthy", "tumor"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    fig.savefig(f"{OUT}/confusion_matrix.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    slices = [dict(np.load(p)) for p in sorted(glob.glob(f"{SLICE_DIR}/*.npz"))]
    print(f"{len(slices)} slices from {len({int(s['case']) for s in slices})} cases")

    print(phase1_table(slices))
    seg = phase2_table(slices)
    summary = seg.groupby("method")[["dice", "iou"]].agg(["mean", "std"]).round(3)
    print(summary)
    best = seg.groupby("method")["dice"].mean().idxmax()

    X, y, cases = classify.build_dataset(SLICE_DIR, best)
    result = classify.train_and_evaluate(X, y, cases)
    result.pop("model")
    pd.DataFrame(X, columns=classify.FEATURE_NAMES).assign(case=cases, tumor=y).to_csv(f"{OUT}/glcm_feature_table.csv", index=False)
    pd.DataFrame(X, columns=classify.FEATURE_NAMES).assign(label=np.where(y == 1, "tumor", "healthy"))         .groupby("label").mean().round(3).T.to_csv(f"{OUT}/feature_means_by_class.csv")
    save_confusion_matrix(result["confusion_matrix"])
    held_out = seg[(seg.method == best) & seg.case.isin(result["test_cases"])]
    result["test_mean_dice"] = float(held_out.dice.mean())
    result.update(best_segmentation=best, mean_dice=float(seg[seg.method == best].dice.mean()),
                  mean_iou=float(seg[seg.method == best].iou.mean()))
    print(json.dumps(result, indent=2, default=float))
    json.dump(result, open(f"{OUT}/metrics.json", "w"), indent=2, default=float)
    save_figures(slices, best)


if __name__ == "__main__":
    main()
