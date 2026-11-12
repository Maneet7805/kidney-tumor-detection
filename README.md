# Kidney Tumor Detection

**Classical computer-vision pipeline for kidney tumor detection in abdominal CT (KiTS23).**
No deep learning, no pre-trained models: windowing, filtering, morphology, thresholding, region growing, texture features and a Random Forest.

Coursework for *CT036-3-3 Image Processing, Computer Vision and Pattern Recognition* (APU). This repository is the **tumor** part of a group project; the kidney part is done by other group members.

![Predicted (red) vs ground-truth (green) tumor contours](results/overlays_region_growing.png)
*Red = predicted tumor, green = radiologist ground truth.*

## Pipeline

```
Raw CT slice (HU)
  -> Phase 1  Enhancement   window [-160, 240] HU -> anisotropic diffusion -> Butterworth low-pass
  -> Phase 2  Segmentation  morphology (custom SEs) + Multi-Otsu  |  seeded region growing
  -> Phase 3  Recognition   shape + GLCM + HU statistics -> Random Forest -> healthy vs tumor/cyst
```

| Phase | Methods compared | Evaluation |
|---|---|---|
| 1. Enhancement | Anisotropic diffusion vs Gaussian vs Median; CLAHE vs Histogram Equalization (both compared, neither in the final chain); Butterworth low-pass on the 2D DFT | PSNR, CNR |
| 2. Segmentation | Strategy A: Multi-Otsu. Strategy B: seeded region growing | Dice, IoU vs ground truth |
| 3. Classification | 15 features (6 shape, 5 GLCM over 4 angles, 4 first-order HU) -> Random Forest | 70/30 split by patient, 5-fold CV, confusion matrix, accuracy, recall, specificity, F1 |

## Results (12 patients, 72 slices)

**Segmentation (Dice / IoU against ground truth, all slices)**

| Method | Dice | IoU |
|---|---|---|
| Multi-Otsu (A) | 0.512 | 0.408 |
| Seeded region growing (B) | **0.573** | 0.486 |

Dice on the 4 held-out test patients (region growing): 0.608 (0.290 before the Phase 2 sweep; training-slice Dice 0.385 -> 0.555). Region growing barely changes under added noise, a brightness shift or a contrast change (`results/robustness.csv`).

**Classification (held-out patients, 94 regions)**

| Accuracy | Recall | Specificity | F1 | 5-fold CV accuracy |
|---|---|---|---|---|
| 0.946 | 1.000 | 0.938 | 0.818 | 0.898 +/- 0.066 |

**Enhancement quality**

| Step | PSNR (dB) | CNR |
|---|---|---|
| Original window | - | 2.19 |
| Anisotropic diffusion | **30.7** | 2.65 |
| Gaussian | 26.6 | 2.70 |
| Median | 25.7 | 2.74 |
| CLAHE | 20.6 | 1.76 |
| Histogram Equalization | 19.3 | 1.61 |
| Butterworth low-pass | 25.5 | 2.76 |
| **Final chain** (window, diffusion, Butterworth) | 25.1 | 2.88 |

Anisotropic diffusion keeps the most structure (highest PSNR) while still raising CNR. CLAHE is not in the final chain: it amplifies noise (lowest PSNR/CNR of the compared steps) and, on the 45 training slices, lowers Dice (Otsu 0.347 with CLAHE vs 0.390 without; region growing 0.244 vs 0.410).

### Phase 2 parameter sweep
`phase2_sweep` in `run_pipeline.py` varies one setting at a time and scores mean Dice on the training cases only (held-out cases 104, 107, 111, 112 excluded): `results/phase2_sweep.csv` and `results/phase2_sweep.png` (red star = chosen value, region growing).

| Setting | Chosen | Why (training Dice, region growing) |
|---|---|---|
| Multi-Otsu classes | 2 | 0.555 vs 0.471 (3) and 0.389 (4): more classes split a tumor from its own rim |
| `DARK_PREFERENCE` | 0.2 | peak of 0.548 / 0.555 / 0.533 / 0.477 for 0.1 / 0.2 / 0.3 / 0.5 |
| region growing `tolerance` | 35 | Dice rises steadily 0.319 -> 0.555 from 8 to 35 (edge of grid) |
| opening disk radius | 3 | 0.404 / 0.517 / 0.532 / 0.555 for 0 to 3: removes more speckle (edge of grid) |
| closing ellipse | (2, 1) | smallest element is best (0.555 vs 0.495 for (8, 6)): large ones merge neighbours |
| `MIN_TUMOR_PIXELS` | 30 (unchanged) | 10 / 30 / 60 differ by under 0.002, so no reason to change |

### Honest limitations
- Segmentation is hard for classical methods here: tumors whose density matches or exceeds healthy kidney (cases 100, 103, 110) and very small tumors fail.
- The tumor search runs inside the ground-truth kidney mask (the kidney is assumed to be localized by the kidney half of the project).
- The Phase 2 settings were chosen on training cases only (below), but each was swept one at a time and `tolerance`, opening radius and closing size peak at the edge of the grid, so slightly better values may exist.
- The held-out set has only 9 tumor regions, so recall 1.00 (4 false positives among 65 healthy regions) is a small-sample estimate.

## Quick start

```bash
pip install -r requirements.txt
python prepare_data.py      # downloads KiTS23 cases 100-112 (several GB) and extracts 2D slices
python run_pipeline.py      # runs all phases, writes tables and figures to results/
python app.py               # Tkinter prototype
```

The 72 extracted slices are already in `data/slices`, so you can skip the download and run the pipeline directly. `test_images/` has the held-out slices as plain PNG.

## Prototype app

`python app.py` opens a window to load one slice or a folder, pick the segmentation method, and view the mask, tumor overlay, kidney overlay and a combined view (predicted red, ground truth green), with Dice per slice and the classifier metrics. It has one tab per phase: Overview, Phase 1 Enhancement (original -> enhanced with every compared algorithm), Phase 2 Segmentation (both strategies with Dice/IoU), Phase 3 Recognition (features, classifier, diagnostic class). Run `run_pipeline.py` first so the app can load the classifier (rebuilt from the saved feature table).

## Repository layout

| File | Purpose |
|---|---|
| `preprocess.py` | Phase 1: windowing, denoising, CLAHE/HE, Butterworth, PSNR, CNR |
| `segment.py` | Phase 2: morphology, Multi-Otsu, region growing, Dice, IoU |
| `classify.py` | Phase 3: feature extraction, Random Forest, evaluation |
| `run_pipeline.py` | End-to-end run, tables and figures |
| `prepare_data.py` | KiTS23 download and slice extraction |
| `app.py` | Prototype GUI |
| `test_pipeline.py` | Automated tests (`python -m pytest -q`), also run by GitHub Actions on every PR |
| `results/` | Figures and CSV/JSON metrics, incl. `parameter_sweep.csv/png` (Phase 1 setting justification), `phase2_sweep.csv/png` (Phase 2 setting justification) |
| `data/slices/` | Extracted CT slices (`.npz`) |
| `test_images/` | Held-out test slices as PNG |

Each module has a small self-check: `python preprocess.py`, `python segment.py`, `python classify.py`.

## Data and credits

Data: KiTS23 kidney tumor segmentation challenge, https://kits-challenge.org/kits23/ (Heller et al.). Please follow the dataset's license and citation requirements.
