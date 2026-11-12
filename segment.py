"""Phase 2: morphology + two classical tumor segmentation strategies + overlap metrics."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, ellipse, opening, closing, remove_small_objects
from skimage.segmentation import flood

MIN_TUMOR_PIXELS = 30


def refine_mask(mask):
    """Custom structuring elements: disk opening removes specks, ellipse closing joins gaps."""
    mask = opening(mask, disk(2))
    mask = closing(mask, ellipse(4, 3))
    mask = ndi.binary_fill_holes(mask)
    return remove_small_objects(mask, MIN_TUMOR_PIXELS)


def _otsu_dark_class(img, region):
    """Strategy A core: 3-class Multi-Otsu inside the kidney, keep the darkest class
    (tumors and cysts enhance less than healthy renal parenchyma)."""
    thresholds = threshold_multiotsu(img[region], classes=3)
    return region & (img < thresholds[0])


def segment_otsu(img, kidney_mask):
    """Strategy A: Multi-Otsu thresholding restricted to the kidney, then morphology."""
    if kidney_mask.sum() < MIN_TUMOR_PIXELS:
        return np.zeros_like(kidney_mask)
    return refine_mask(_otsu_dark_class(img, kidney_mask))


def segment_region_growing(img, kidney_mask, tolerance=18):
    """Strategy B: seeded region growing. The seed is the centroid of the biggest dark blob
    (from Multi-Otsu); neighbours join while within `tolerance` grey levels of the seed."""
    empty = np.zeros_like(kidney_mask)
    if kidney_mask.sum() < MIN_TUMOR_PIXELS:
        return empty
    blobs = label(refine_mask(_otsu_dark_class(img, kidney_mask)))
    if blobs.max() == 0:
        return empty
    biggest = max(regionprops(blobs), key=lambda r: r.area)
    seed = tuple(int(c) for c in biggest.centroid)
    if not blobs[seed]:  # centroid of a curved blob may fall outside it: use a blob pixel
        seed = tuple(biggest.coords[len(biggest.coords) // 2])
    smooth = cv2.medianBlur(img, 5)  # stops single noisy pixels from blocking the growth
    grown = flood(smooth, seed, tolerance=tolerance) & kidney_mask
    return refine_mask(grown)


def segment_tumor(enhanced, kidney_mask, method="otsu"):
    """Entry point used by the pipeline and the app."""
    if method == "otsu":
        return segment_otsu(enhanced, kidney_mask)
    if method == "region_growing":
        return segment_region_growing(enhanced, kidney_mask)
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
    tumor = (yy - 60) ** 2 + (xx - 70) ** 2 < 12 ** 2
    img = np.clip(np.where(kidney, 170, 40) - np.where(tumor, 70, 0) + rng.normal(0, 5, (128, 128)), 0, 255).astype(np.uint8)
    for m in ("otsu", "region_growing"):
        score = dice(segment_tumor(img, kidney, m), tumor)
        assert score > 0.9, (m, score)
    print("segment self-check OK")
