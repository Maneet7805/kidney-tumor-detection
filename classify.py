"""Phase 3: shape / GLCM / first-order features and a Random Forest classifier."""
import glob

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import preprocess
import segment

ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
LEVELS = 32  # grey levels used for the GLCM
FEATURE_NAMES = ["area", "perimeter_area_ratio", "circularity", "compactness", "eccentricity", "solidity",
                 "glcm_energy", "glcm_contrast", "glcm_homogeneity", "glcm_correlation", "glcm_entropy",
                 "hu_mean", "hu_variance", "hu_skewness", "hu_kurtosis"]


def shape_features(mask):
    """Area, perimeter/area, circularity, compactness, eccentricity, solidity."""
    r = regionprops(mask.astype(np.uint8))[0]
    circularity = 4 * np.pi * r.area / (r.perimeter ** 2 + 1e-8)
    return [r.area, r.perimeter / r.area, circularity, r.perimeter ** 2 / (4 * np.pi * r.area),
            r.eccentricity, r.solidity]


def glcm_features(enhanced, mask):
    """GLCM energy, contrast, homogeneity, correlation, entropy, averaged over 4 angles."""
    ys, xs = np.where(mask)
    crop = enhanced[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    quantized = (crop.astype(np.uint16) * LEVELS // 256).astype(np.uint8)
    glcm = graycomatrix(quantized, [1], ANGLES, levels=LEVELS, symmetric=True, normed=True)
    props = [graycoprops(glcm, p).mean() for p in ("energy", "contrast", "homogeneity", "correlation")]
    entropy = -np.mean(np.sum(glcm * np.log2(glcm + 1e-12), axis=(0, 1)))
    return props + [entropy]


def intensity_features(hu, mask):
    """First-order HU statistics inside the mask: mean, variance, skewness, kurtosis."""
    v = hu[mask]
    return [v.mean(), v.var(), skew(v), kurtosis(v)]


def extract_features(enhanced, hu, mask):
    """Feature vector of one region of interest."""
    return np.array(shape_features(mask) + glcm_features(enhanced, mask) + intensity_features(hu, mask))


def _regions(mask, min_pixels=segment.MIN_TUMOR_PIXELS):
    """Connected components of a mask that are big enough to describe."""
    parts = label(mask)
    return [parts == i for i in range(1, parts.max() + 1) if (parts == i).sum() >= min_pixels]


def build_dataset(slice_dir, method):
    """Candidate regions per slice -> (X, y, case ids).
    Tumor candidates: components of the predicted mask. Healthy candidates: kidney tissue
    away from the true tumor. Label 1 (tumor) if >= 50% of the region overlaps the ground truth."""
    X, y, cases = [], [], []
    for path in sorted(glob.glob(f"{slice_dir}/*.npz")):
        d = np.load(path)
        enhanced = preprocess.enhance(d["hu"])
        pred = segment.segment_tumor(enhanced, d["kidney"], method)
        away_from_tumor = d["kidney"] & ~ndi.binary_dilation(d["tumor"], iterations=5) & ~pred
        healthy = ndi.binary_opening(away_from_tumor, iterations=3)
        for region in _regions(pred) + _regions(healthy)[:3]:
            X.append(extract_features(enhanced, d["hu"], region))
            y.append(int(d["tumor"][region].mean() >= 0.5))
            cases.append(int(d["case"]))
    return np.array(X), np.array(y), np.array(cases)


def train_and_evaluate(X, y, groups, seed=0):
    """70/30 split by patient, 5-fold CV on the 70%, final scores on the unseen 30%."""
    split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train, test = next(split.split(X, y, groups))
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=seed))
    cv = cross_val_score(model, X[train], y[train], groups=groups[train],
                         cv=GroupKFold(n_splits=min(5, len(set(groups[train])))))
    model.fit(X[train], y[train])
    tn, fp, fn, tp = confusion_matrix(y[test], model.predict(X[test]), labels=[0, 1]).ravel()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / len(test),
        "recall": recall,  # sensitivity
        "specificity": tn / max(tn + fp, 1),
        "f1": 2 * precision * recall / max(precision + recall, 1e-8),
        "cv_accuracy_mean": cv.mean(), "cv_accuracy_std": cv.std(),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n_train": len(train), "n_test": len(test), "test_cases": sorted({int(g) for g in groups[test]}),
        "model": model,
    }


if __name__ == "__main__":
    # Self-check: separable synthetic features must be classified well.
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 60)
    X = rng.normal(0, 1, (120, 15)) + y[:, None] * 2
    groups = np.tile(np.arange(12), 10)
    assert train_and_evaluate(X, y, groups)["accuracy"] > 0.9
    # Feature vector length on a toy disk
    yy, xx = np.ogrid[:64, :64]
    disk_mask = (yy - 32) ** 2 + (xx - 32) ** 2 < 15 ** 2
    img = rng.integers(0, 255, (64, 64)).astype(np.uint8)
    assert extract_features(img, img.astype(float), disk_mask).shape == (15,)
    print("classify self-check OK")
