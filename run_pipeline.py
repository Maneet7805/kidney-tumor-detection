"""End-to-end run on all extracted slices. Writes tables, figures and metrics.json to results/."""
import glob
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import closing, disk, ellipse, opening

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
        variants["homomorphic"] = preprocess.homomorphic_filter(base)
        variants["full_chain"] = preprocess.enhance(d["hu"])
        variants["original_window"] = base
        for name, img in variants.items():
            rows.append({"method": name, "psnr": preprocess.psnr(base, img), "cnr": preprocess.cnr(img, roi, ring)})
    table = pd.DataFrame(rows).replace(np.inf, np.nan).groupby("method").mean().round(3)
    table.to_csv(f"{OUT}/phase1_psnr_cnr.csv")
    return table


def chain_check(slices):
    """Does adding the homomorphic filter to the end of the final chain help? Training cases only."""
    test_cases = json.load(open(f"{OUT}/metrics.json"))["test_cases"]
    train = [d for d in slices if int(d["case"]) not in test_cases]
    chains = {"window-anisotropic-butterworth (final)": preprocess.enhance,
              "... + homomorphic": lambda hu: preprocess.homomorphic_filter(preprocess.enhance(hu))}
    rows = []
    for name, chain in chains.items():
        out = [chain(d["hu"]) for d in train]
        row = {"chain": name, "n_train_slices": len(train)}
        for m in METHODS:
            row[f"dice_{m}"] = np.mean([segment.dice(segment.segment_tumor(e, d["kidney"], m), d["tumor"]) for e, d in zip(out, train)])
        row["kidney_cnr"] = np.mean([preprocess.cnr(e, d["kidney"], ndi.binary_dilation(d["kidney"], iterations=15) & ~ndi.binary_dilation(d["kidney"], iterations=4))
                                     for e, d in zip(out, train)])
        rows.append(row)
    table = pd.DataFrame(rows).round(4)
    table.to_csv(f"{OUT}/phase1_chain_check.csv", index=False)
    print(table.to_string(index=False))


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


def save_histograms(slices):
    """Step 2 evidence: grey-level and cumulative histograms of windowed / CLAHE / HE for one slice."""
    base = preprocess.window_normalize(max(slices, key=lambda s: s["tumor"].sum())["hu"])
    panels = {"Windowed (no enhancement)": base, "CLAHE (local, clip limit 3)": preprocess.enhance_contrast(base, "clahe"),
              "Histogram equalization (global)": preprocess.enhance_contrast(base, "he")}
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for col, (title, img) in enumerate(panels.items()):
        counts = np.bincount(img.ravel(), minlength=256)
        axes[0, col].imshow(img, cmap="gray"); axes[0, col].set_title(title); axes[0, col].axis("off")
        axes[1, col].bar(range(256), counts, width=1.0)
        axes[1, col].set_yscale("log")  # the black background pixels would otherwise flatten everything else
        axes[1, col].set_ylabel("grey-level histogram\n(pixel count, log scale)")
        axes[2, col].plot(np.cumsum(counts) / counts.sum())  # HE aims for a straight diagonal here
        axes[2, col].set_ylabel("cumulative histogram\n(fraction of pixels)")
        for row in (1, 2):
            axes[row, col].set_xlabel("grey level (0-255)"); axes[row, col].set_xlim(-1, 256); axes[row, col].grid(alpha=0.3)
    fig.suptitle("Step 2: contrast enhancement on the windowed soft-tissue band (the spike at grey level 0 is the black air background)")
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    fig.savefig(f"{OUT}/phase1_histograms.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def save_edge_figure(slices):
    """Step 2 evidence: does the filter keep the kidney border sharp? Crop + 1-D profile across the border.
    Gaussian and median strengths are matched to the anisotropic filter's noise level in flat kidney tissue,
    so the only difference left is what each filter does to the edge."""
    d = max(slices, key=lambda s: s["kidney"].sum())
    base = preprocess.window_normalize(d["hu"])
    grad_y, grad_x = np.gradient(cv2.GaussianBlur(base.astype(np.float32), (0, 0), 1))
    border = d["kidney"] & ~ndi.binary_erosion(d["kidney"], iterations=2)
    y, x = np.unravel_index(np.argmax(np.where(border, np.hypot(grad_y, grad_x), 0)), base.shape)  # sharpest border point
    ny, nx = grad_y[y, x], grad_x[y, x]
    ny, nx = ny / np.hypot(ny, nx), nx / np.hypot(ny, nx)  # unit vector across the border

    flat = ndi.binary_erosion(d["kidney"] & ~ndi.binary_dilation(d["tumor"], iterations=3), iterations=8)
    noise = lambda im: im[flat].std()
    aniso = preprocess.denoise(base, "anisotropic")
    sigma = min([0.5, 1, 1.5, 2, 3, 4], key=lambda s: abs(noise(cv2.GaussianBlur(base, (0, 0), s)) - noise(aniso)))
    kernel = min([3, 5, 7, 9, 11, 13], key=lambda k: abs(noise(cv2.medianBlur(base, k)) - noise(aniso)))
    panels = {"windowed": base, "anisotropic": aniso, f"Gaussian (sigma {sigma})": cv2.GaussianBlur(base, (0, 0), sigma),
              f"median ({kernel}x{kernel})": cv2.medianBlur(base, kernel)}

    t = np.arange(-15, 16)
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.1])
    prof_ax = fig.add_subplot(grid[1, :])
    for col, (title, img) in enumerate(panels.items()):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.plot([x - 15 * nx, x + 15 * nx], [y - 15 * ny, y + 15 * ny], "r-", lw=1)
        ax.set_xlim(x - 25, x + 25); ax.set_ylim(y + 25, y - 25); ax.axis("off")
        ax.set_title(f"{title}\nflat-tissue noise (std) {noise(img):.1f}", fontsize=10)
        prof_ax.plot(t, ndi.map_coordinates(img.astype(float), [y + t * ny, x + t * nx], order=1), label=title)
    prof_ax.set_xlabel("position along the red line (pixels, 0 = kidney border)"); prof_ax.set_ylabel("grey level")
    prof_ax.set_title("Intensity profile across the kidney border: a steep step = edge kept, a gentle ramp = edge blurred")
    prof_ax.legend(); prof_ax.grid(alpha=0.3)
    fig.suptitle(f"Step 2: edge preservation (case {int(d['case'])}, slice {int(d['z'])}), filters matched to equal noise reduction")
    fig.savefig(f"{OUT}/phase1_edges.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def save_confusion_matrix(cm):
    """Confusion matrix figure for the held-out test regions."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(np.array(cm)):
        ax.text(j, i, v, ha="center", va="center", fontsize=16)
    ax.set_xticks([0, 1], ["healthy", "tumor"]); ax.set_yticks([0, 1], ["healthy", "tumor"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    fig.savefig(f"{OUT}/confusion_matrix.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def parameter_sweep(slices):
    """Phase 1 goal = make the TUMOR stand out from healthy kidney tissue with its border intact.
    Each setting is applied alone to the windowed image and scored by:
      tumor_cnr  |mean tumor - mean healthy kidney| / std of healthy kidney  (can we see the tumor?)
      edge       Sobel gradient on the tumor border relative to the unfiltered window (border kept?)
      dice       Multi-Otsu segmentation on that image vs ground truth (does it help the end result?)"""
    grid = {
        "anisotropic (kappa)": (lambda b, k: preprocess.anisotropic_diffusion(b, kappa=k), [5, 10, 20, 40, 80]),
        "gaussian (sigma)": (lambda b, k: cv2.GaussianBlur(b, (0, 0), k), [0.5, 1, 1.5, 2.5, 4]),
        "median (kernel)": (lambda b, k: cv2.medianBlur(b, k), [3, 5, 7, 9]),
        "butterworth (cutoff)": (lambda b, k: preprocess.butterworth_lowpass(b, cutoff=k), [20, 40, 60, 100, 150]),
        # homomorphic: one parameter varied at a time around the defaults gL 0.5, gH 1.5, D0 6, c 1
        "homomorphic (gamma_low)": (lambda b, k: preprocess.homomorphic_filter(b, gamma_low=k), [0.3, 0.5, 0.7, 1.0]),
        "homomorphic (gamma_high)": (lambda b, k: preprocess.homomorphic_filter(b, gamma_high=k), [1.0, 1.5, 2.0, 3.0]),
        "homomorphic (cutoff)": (lambda b, k: preprocess.homomorphic_filter(b, cutoff=k), [2, 6, 15, 30, 60]),
        "homomorphic (c)": (lambda b, k: preprocess.homomorphic_filter(b, c=k), [0.5, 1, 2, 5]),
        "clahe (clip limit)": (lambda b, k: cv2.createCLAHE(clipLimit=k, tileGridSize=(8, 8)).apply(b), [1, 2, 3, 5, 8]),
        "he (no setting)": (lambda b, k: cv2.equalizeHist(b), [0]),
        "none (windowed only)": (lambda b, k: b, [0]),
    }
    test_cases = json.load(open(f"{OUT}/metrics.json"))["test_cases"]  # held-out patients never influence a setting
    usable = [d for d in slices if int(d["case"]) not in test_cases and d["tumor"].sum() >= 100 and (d["kidney"] & ~ndi.binary_dilation(d["tumor"], iterations=3)).sum() >= 100]
    rows = []
    for name, (apply, values) in grid.items():
        for v in values:
            m = {"tumor_cnr": [], "edge": [], "dice": []}
            for d in usable:
                base = preprocess.window_normalize(d["hu"])
                out = apply(base, v)
                tumor = d["tumor"]
                healthy = d["kidney"] & ~ndi.binary_dilation(tumor, iterations=3)
                border = ndi.binary_dilation(tumor, iterations=2) & ~ndi.binary_erosion(tumor, iterations=2)
                grad = lambda im: np.hypot(cv2.Sobel(im, cv2.CV_32F, 1, 0), cv2.Sobel(im, cv2.CV_32F, 0, 1))[border].mean()
                m["tumor_cnr"].append(preprocess.cnr(out, tumor, healthy))
                m["edge"].append(grad(out) / grad(base))
                m["dice"].append(segment.dice(segment.segment_otsu(out, d["kidney"]), tumor))
            rows.append({"filter": name, "value": v, **{k: np.mean(x) for k, x in m.items()}})
    table = pd.DataFrame(rows).round(3)
    table.to_csv(f"{OUT}/parameter_sweep.csv", index=False)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    plain = table[table["filter"] == "none (windowed only)"].iloc[0]
    for col, (name, g) in zip(range(5), [x for x in table.groupby("filter", sort=False)][:5]):
        for row, metric in enumerate(("dice", "tumor_cnr")):
            ax = axes[row, col]
            ax.plot(g.value, g[metric], "o-")
            ax.axhline(plain[metric], color="gray", ls="--", label="no filter")
            ax.set_title(f"{name}\n{metric}", fontsize=9); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(f"{OUT}/parameter_sweep.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"{len(usable)} slices with a measurable tumor")
    print(table.to_string(index=False))


def phase2_sweep(slices):
    """Justify the Phase 2 settings (3rd item of each grid entry = value kept in segment.py): vary ONE parameter at a time (others at default) and score the
    mean Dice/IoU on TRAINING cases only, so the held-out test cases never influence the choice."""
    test_cases = json.load(open(f"{OUT}/metrics.json"))["test_cases"]
    train = [d for d in slices if int(d["case"]) not in test_cases]
    enhanced = [preprocess.enhance(d["hu"]) for d in train]  # enhance once, reused by every setting
    grid = {
        "Otsu classes": ("classes", [2, 3, 4], 2),
        "dark preference": ("dark_pref", [0.1, 0.2, 0.3, 0.5], 0.2),
        "growing tolerance": ("tolerance", [8, 12, 18, 25, 35], 35),
        "opening radius": ("open_r", [0, 1, 2, 3], 3),
        "closing ellipse": ("close_size", [(2, 1), (4, 3), (6, 4), (8, 6)], (2, 1)),
        "min tumor pixels": ("min_pixels", [10, 30, 60], 30),  # 10 is only 0.002 better (one slice): not worth changing
    }
    rows = []
    for group, (arg, values, _) in grid.items():
        for v in values:
            for method in METHODS:
                if arg == "tolerance" and method == "otsu":
                    continue  # tolerance only exists in region growing
                preds = [segment.segment_tumor(e, d["kidney"], method, **{arg: v}) for e, d in zip(enhanced, train)]
                rows.append({"group": group, "value": str(v), "method": method,
                             "n_train_slices": len(train),
                             "dice": np.mean([segment.dice(p, d["tumor"]) for p, d in zip(preds, train)]),
                             "iou": np.mean([segment.iou(p, d["tumor"]) for p, d in zip(preds, train)])})
    table = pd.DataFrame(rows).round(4)
    table.to_csv(f"{OUT}/phase2_sweep.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (group, g) in zip(axes.ravel(), table.groupby("group", sort=False)):
        for method, line in g.groupby("method"):
            ax.plot(line.value, line.dice, "o-", label=method)
        chosen = g[(g.method == "region_growing") & (g.value == str(grid[group][2]))].iloc[0]
        ax.plot(chosen.value, chosen.dice, "r*", ms=16, label="chosen")
        ax.set_title(group); ax.set_ylabel("train mean Dice"); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(f"{OUT}/phase2_sweep.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"{len(train)} training slices (test cases {test_cases} excluded)")
    print(table.to_string(index=False))


def morphology_demo(slices):
    """Erosion/dilation/opening/closing with custom SEs on a real mask and on two touching discs."""
    d = slices[0]
    raw = segment.segment_otsu_raw(preprocess.enhance(d["hu"]), d["kidney"])
    yy, xx = np.ogrid[:80, :120]
    touching = ((yy - 40) ** 2 + (xx - 40) ** 2 < 24 ** 2) | ((yy - 40) ** 2 + (xx - 76) ** 2 < 24 ** 2)
    ops = {"original": None, "erosion disk(5)": lambda m: ndi.binary_erosion(m, disk(5)),
           "dilation disk(3)": lambda m: ndi.binary_dilation(m, disk(3)),
           "opening disk(2)": lambda m: opening(m, disk(2)), "closing ellipse(4,3)": lambda m: closing(m, ellipse(4, 3))}
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for row, mask in zip(axes, (raw, touching)):
        ys, xs = np.where(mask)
        for ax, (name, op) in zip(row, ops.items()):
            ax.imshow(mask if op is None else op(mask), cmap="gray"); ax.set_title(name); ax.axis("off")
            ax.set_xlim(xs.min() - 10, xs.max() + 10); ax.set_ylim(ys.max() + 10, ys.min() - 10)
    fig.savefig(f"{OUT}/morphology_demo.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def robustness_test(slices, method):
    """Mean Dice when the input CT is perturbed (noise, brightness shift, contrast change)."""
    rng = np.random.default_rng(0)
    cases = {"original": lambda h: h, "gaussian noise 20 HU": lambda h: h + rng.normal(0, 20, h.shape),
             "brightness +30 HU": lambda h: h + 30, "contrast x0.8": lambda h: (h - 40) * 0.8 + 40}
    rows = [{"perturbation": name, "mean_dice": np.mean([segment.dice(
        segment.segment_tumor(preprocess.enhance(f(d["hu"])), d["kidney"], method), d["tumor"]) for d in slices])}
        for name, f in cases.items()]
    pd.DataFrame(rows).round(3).to_csv(f"{OUT}/robustness.csv", index=False)
    print(pd.DataFrame(rows).round(3))


def export_test_images(slices, test_cases):
    """Held-out test slices as plain PNGs (windowed CT + ground-truth tumor mask)."""
    os.makedirs("test_images", exist_ok=True)
    for d in slices:
        if int(d["case"]) in test_cases:
            name = f"test_images/case{int(d['case'])}_z{int(d['z'])}"
            cv2.imwrite(name + "_ct.png", preprocess.window_normalize(d["hu"]))
            cv2.imwrite(name + "_tumor_gt.png", d["tumor"].astype(np.uint8) * 255)


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
    save_histograms(slices)
    save_edge_figure(slices)
    chain_check(slices)
    parameter_sweep(slices)
    phase2_sweep(slices)  # needs metrics.json for the held-out case list
    morphology_demo(slices)
    robustness_test(slices, best)
    export_test_images(slices, result["test_cases"])


if __name__ == "__main__":
    main()
