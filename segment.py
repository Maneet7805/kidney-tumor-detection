"""Phase 2: morphology + two classical tumor segmentation strategies + overlap metrics."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, ellipse, opening, closing, remove_small_objects
from skimage.segmentation import flood

MIN_TUMOR_PIXELS = 30
# Defaults below were chosen by run_pipeline.phase2_sweep (training cases only, results/phase2_sweep.csv)
DARK_PREFERENCE = 0.2


def refine_mask(mask, open_r=3, close_size=(2, 1), min_pixels=MIN_TUMOR_PIXELS):
    """Custom structuring elements: disk opening removes specks, ellipse closing joins gaps.
    The arguments exist so the Phase 2 sweep can vary them; the defaults are the chosen values."""
    mask = opening(mask, disk(open_r))
    mask = closing(mask, ellipse(*close_size))
    mask = ndi.binary_fill_holes(mask)
    return keep_main_blobs(remove_small_objects(mask, max_size=min_pixels - 1))


def keep_main_blobs(mask, ratio=0.25):
    """Drop blobs smaller than 25% of the biggest one (sinus fat, noise in the healthy kidney)."""
    parts = label(mask)
    if parts.max() == 0:
        return mask
    sizes = np.bincount(parts.ravel())[1:]
    return np.isin(parts, np.where(sizes >= ratio * sizes.max())[0] + 1)


def _otsu_candidate(img, region, classes=2, dark_pref=DARK_PREFERENCE, **morph):
    """Strategy A core: 2-class (best in the sweep; 3-4 split the tumor from its own rim) Multi-Otsu inside the kidney. Tumors are usually darker than
    healthy parenchyma but in some cases brighter, so build both the darkest- and brightest-class
    masks and keep the dark one unless it is far smaller (a thin cortical rim is removed by the opening,
    a real tumor survives)."""
    cuts = threshold_multiotsu(img[region], classes=classes)
    low, high = cuts[0], cuts[-1]  # darkest and brightest class borders (equal when classes=2)
    dark, bright = refine_mask(region & (img < low), **morph), refine_mask(region & (img >= high), **morph)
    # Prefer the dark class (typical); fall back to bright only when dark is much smaller (ratio picked by phase2_sweep)
    return dark if dark.sum() >= dark_pref * bright.sum() else bright


def segment_otsu_raw(img, kidney_mask):
    """Multi-Otsu dark-class mask BEFORE morphology (used to demonstrate the structuring elements)."""
    low = threshold_multiotsu(img[kidney_mask], classes=2)[0]
    return kidney_mask & (img < low)


def segment_otsu(img, kidney_mask, **params):
    """Strategy A: Multi-Otsu thresholding restricted to the kidney, then morphology."""
    if kidney_mask.sum() < MIN_TUMOR_PIXELS:
        return np.zeros_like(kidney_mask)
    return _otsu_candidate(img, kidney_mask, **params)


def segment_region_growing(img, kidney_mask, tolerance=35, classes=2, dark_pref=DARK_PREFERENCE, **morph):
    """Strategy B: seeded region growing. The seed is the centroid of the biggest dark blob
    (from Multi-Otsu, either tissue class); neighbours join while within `tolerance` grey levels of the seed."""
    empty = np.zeros_like(kidney_mask)
    if kidney_mask.sum() < MIN_TUMOR_PIXELS:
        return empty
    blobs = label(_otsu_candidate(img, kidney_mask, classes, dark_pref, **morph))
    if blobs.max() == 0:
        return empty
    biggest = max(regionprops(blobs), key=lambda r: r.area)
    seed = tuple(int(c) for c in biggest.centroid)
    if not blobs[seed]:  # centroid of a curved blob may fall outside it: use a blob pixel
        seed = tuple(biggest.coords[len(biggest.coords) // 2])
    smooth = cv2.medianBlur(img, 5)  # stops single noisy pixels from blocking the growth
    grown = flood(smooth, seed, tolerance=tolerance) & kidney_mask
    return refine_mask(grown, **morph)


def segment_tumor(enhanced, kidney_mask, method="otsu", **params):
    """Entry point used by the pipeline and the app. `params` override defaults (used by the sweep)."""
    if method == "otsu":
        return segment_otsu(enhanced, kidney_mask, **params)
    if method == "region_growing":
        return segment_region_growing(enhanced, kidney_mask, **params)
    raise ValueError(f"unknown segmentation method: {method}")


def dice(pred, gt):
    """Dice similarity coefficient: 2|A n B| / (|A| + |B|). Two empty masks count as perfect."""
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else 2 * np.logical_and(pred, gt).sum() / total


def iou(pred, gt):
    """Jaccard index: |A n B| / |A u B|."""
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else np.logical_and(pred, gt).sum() / union


if __name__ == "__main__":
    # Synthetic kidney (bright disk) containing a dark tumor disk, with noise.
    rng = np.random.default_rng(0)
    yy, xx = np.ogrid[:128, :128]
    kidney = (yy - 64) ** 2 + (xx - 64) ** 2 < 40 ** 2
    tumor = (yy - 60) ** 2 + (xx - 70) ** 2 < 20 ** 2  # must exceed DARK_PREFERENCE of the rest of the kidney
    img = np.clip(np.where(kidney, 170, 40) - np.where(tumor, 70, 0) + rng.normal(0, 5, (128, 128)), 0, 255).astype(np.uint8)
    for m in ("otsu", "region_growing"):
        score = dice(segment_tumor(img, kidney, m), tumor)
        assert score > 0.9, (m, score)
    print("segment self-check OK")
