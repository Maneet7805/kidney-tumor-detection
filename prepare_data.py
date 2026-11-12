"""Prepare KiTS23 data: download cases 100-111 and save the 6 best tumor slices per case.

Output: data/slices/case_XXXXX_zNNN.npz with keys hu, kidney, tumor, case, z
        results/data_montage.png (one example slice per case)
Run: python prepare_data.py
"""
import subprocess
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent
TMP = ROOT / "data" / "_kits23_tmp"          # labels repo + downloaded imaging (temporary)
OUT = ROOT / "data" / "slices"
FIRST_CASE, N_CASES = 100, 12                # cases 100..111
SLICES_PER_CASE, MIN_GAP = 6, 5              # 6 slices, at least 5 apart in z
IMG_URL = "https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/case_{:05d}.nii.gz"


def get_labels():
    """Shallow-clone the KiTS23 repo (it holds segmentation.nii.gz for every case)."""
    if not TMP.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/neheller/kits23", str(TMP)], check=True)


def get_imaging(case):
    """Download imaging.nii.gz for one case next to its labels; return (ct_path, seg_path)."""
    folder = TMP / "dataset" / f"case_{case:05d}"
    ct = folder / "imaging.nii.gz"
    if not ct.exists():
        urllib.request.urlretrieve(IMG_URL.format(case), ct)
    return ct, folder / "segmentation.nii.gz"


def load_upright(path):
    """Load a NIfTI as RAS array (x, y, z), then make each axial slice [row=anterior->posterior, col] upright."""
    arr = nib.as_closest_canonical(nib.load(path)).get_fdata(dtype=np.float32)
    return np.flip(arr.transpose(1, 0, 2), axis=0)     # (H, W, Z), anterior at the top


def pick_slices(tumor_area):
    """Greedy: z indices with the largest tumor area, each >= gap apart from those already chosen.
    Small tumors span <26 slices, so if 6 slices do not fit with gap 5 the gap is relaxed (3, 2, 1)
    instead of dropping the case (otherwise 10 of the first 20 cases would be skipped)."""
    for gap in (MIN_GAP, 3, 2, 1):
        chosen = []
        for z in np.argsort(tumor_area)[::-1]:
            if tumor_area[z] == 0 or len(chosen) == SLICES_PER_CASE:
                break
            if all(abs(z - c) >= gap for c in chosen):
                chosen.append(int(z))
        if len(chosen) == SLICES_PER_CASE:
            break
    return chosen


def process_case(case):
    """Save the chosen slices of one case; return their z list (empty if the case has too few tumor slices)."""
    ct_path, seg_path = get_imaging(case)
    hu, seg = load_upright(ct_path), load_upright(seg_path).astype(np.uint8)  # nibabel applies slope/intercept -> HU
    assert hu.shape == seg.shape and hu.shape[:2] == (512, 512), f"unexpected shape {hu.shape}"
    zs = pick_slices((seg >= 2).sum(axis=(0, 1)))      # tumor = labels 2 (tumor) + 3 (cyst)
    if len(zs) < SLICES_PER_CASE:
        return []
    for z in zs:
        np.savez_compressed(OUT / f"case_{case:05d}_z{z:03d}.npz", hu=hu[:, :, z],
                            kidney=seg[:, :, z] > 0, tumor=seg[:, :, z] >= 2, case=case, z=z)
    return zs


def montage(files):
    """One slice per case, tumor in red and kidney in green contour, saved to results/data_montage.png."""
    first = {}
    for f in files:
        first.setdefault(int(f.name[5:10]), f)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, (case, f) in zip(axes.flat, sorted(first.items())):
        d = np.load(f)
        ax.imshow(np.clip(d["hu"], -160, 240), cmap="gray")
        ax.contour(d["kidney"], [0.5], colors="lime", linewidths=0.6)
        ax.contour(d["tumor"], [0.5], colors="red", linewidths=0.8)
        ax.set_title(f"case {case}  z={int(d['z'])}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    (ROOT / "results").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "data_montage.png", dpi=110)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    get_labels()
    done, case = [], FIRST_CASE
    while len(done) < N_CASES:
        try:
            ok = bool(process_case(case))
        except Exception as e:                       # failed download etc: skip, take the next case number
            print(f"case {case} skipped: {e}")
            ok = False
        if ok:
            done.append(case)
        else:
            print(f"case {case} skipped (no usable tumor slices or error); trying the next number")
        case += 1
    files = sorted(OUT.glob("case_*.npz"))
    px = np.array([np.load(f)["tumor"].sum() for f in files])
    print(f"cases {done}: {len(files)} slices, tumor px min/median/max = {px.min()}/{int(np.median(px))}/{px.max()}")
    montage(files)


if __name__ == "__main__":
    main()
