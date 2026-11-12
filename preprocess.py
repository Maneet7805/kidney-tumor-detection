"""Phase 1: CT windowing, denoising, contrast enhancement, frequency filtering, quality metrics."""
import cv2
import numpy as np


def window_normalize(hu, lo=-160, hi=240):
    """Clip HU to the soft-tissue window and rescale linearly to 0-255."""
    # 400 HU spread over 256 grey levels, so 1 grey level is about 1.57 HU
    clipped = np.clip(hu, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


def anisotropic_diffusion(img, iterations=10, kappa=20, gamma=0.2):
    """Perona-Malik diffusion: smooths flat regions but stops at strong edges.

    kappa is in grey levels (20 is about 31 HU): jumps bigger than this (kidney/tumor edges)
    are kept, smaller ones (noise) are smoothed. iterations = number of smoothing rounds.
    gamma must be <= 0.25 for the 4-neighbour scheme to stay stable
    (Perona & Malik, 1990, IEEE TPAMI 12(7)).
    """
    out = img.astype(np.float32)
    for _ in range(iterations):
        # Differences to the 4 neighbours (north, south, east, west).
        # np.roll wraps around the border, which is harmless as kidneys are far from the edge.
        diffs = [np.roll(out, s, axis=a) - out for a in (0, 1) for s in (1, -1)]
        # Conduction coefficient is small where the gradient (edge) is large
        out += gamma * sum(d * np.exp(-(d / kappa) ** 2) for d in diffs)
    return np.clip(out, 0, 255).astype(np.uint8)


def denoise(img, method="anisotropic"):
    """Noise suppression. Methods: anisotropic (edge preserving), gaussian, median."""
    if method == "anisotropic":
        return anisotropic_diffusion(img)
    if method == "gaussian":
        return cv2.GaussianBlur(img, (5, 5), 1.5)
    if method == "median":
        return cv2.medianBlur(img, 5)
    raise ValueError(f"unknown denoise method: {method}")


def enhance_contrast(img, method="clahe"):
    """Contrast enhancement. Methods: clahe (local, clip-limited) or he (global)."""
    if method == "clahe":
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)
    if method == "he":
        return cv2.equalizeHist(img)
    raise ValueError(f"unknown contrast method: {method}")


def log_spectrum(img):
    """Log-magnitude of the centred 2D DFT (used for the report figures)."""
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img))))


def butterworth_lowpass(img, cutoff=60, order=2):
    """Frequency-domain Butterworth low-pass filter: H = 1 / (1 + (D/D0)^(2n))."""
    # cutoff (D0) is a distance in frequency pixels from the centre: lower = blurrier.
    # order (n) sets how sharp the cutoff is: higher = closer to an ideal filter.
    rows, cols = img.shape
    v, u = np.ogrid[:rows, :cols]
    dist = np.sqrt((u - cols / 2) ** 2 + (v - rows / 2) ** 2)
    h = 1 / (1 + (dist / cutoff) ** (2 * order))
    spectrum = np.fft.fftshift(np.fft.fft2(img))
    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * h)))
    return np.clip(filtered, 0, 255).astype(np.uint8)


def enhance(hu):
    """Final Phase 1 chain: window -> anisotropic diffusion -> Butterworth.

    CLAHE is left out: on 45 training slices it amplifies noise and lowers Dice
    (Otsu 0.347 with CLAHE vs 0.390 without; CLAHE alone is the lowest in results/parameter_sweep.csv).
    """
    img = window_normalize(hu)
    img = denoise(img, "anisotropic")
    return butterworth_lowpass(img)


def psnr(reference, processed):
    """Peak signal-to-noise ratio in dB (higher = closer to the reference)."""
    mse = np.mean((reference.astype(np.float64) - processed.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def cnr(img, roi_mask, background_mask):
    """Contrast-to-noise ratio between a tissue ROI and its surrounding background."""
    roi, bg = img[roi_mask].astype(np.float64), img[background_mask].astype(np.float64)
    return abs(roi.mean() - bg.mean()) / (bg.std() + 1e-8)


if __name__ == "__main__":
    # Self-check on synthetic data: a bright square on noise should keep its contrast.
    rng = np.random.default_rng(0)
    fake_hu = rng.normal(40, 30, (128, 128)).astype(np.float32)
    fake_hu[40:90, 40:90] += 60
    out = enhance(fake_hu)
    assert out.shape == fake_hu.shape and out.dtype == np.uint8
    roi = np.zeros((128, 128), bool)
    roi[50:80, 50:80] = True
    bg = np.ones((128, 128), bool)
    bg[30:100, 30:100] = False  # background = well outside the bright square
    base = window_normalize(fake_hu)
    # Denoising + low-pass must raise CNR
    smooth = butterworth_lowpass(denoise(base))
    assert cnr(smooth, roi, bg) > cnr(base, roi, bg)
    print("preprocess self-check OK")
