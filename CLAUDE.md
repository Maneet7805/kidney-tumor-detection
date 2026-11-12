# CLAUDE.md — CT036-3-IPPR group assignment (KiTS23, TUMOR member)

Brief: `C:\Users\Maneet\Downloads\CT036-3-3-IPPR - Assignment Question (September 2026 intake) (1).pdf`. Due 13 Dec 2026.
Automated abdominal CT pipeline: Phase 1 enhancement -> Phase 2 segmentation -> Phase 3 features + classifier.

## Scope (this repo = Maneet, TUMOR only)
- Group dataset: KiTS23. 2 members do kidney, 2 do tumor. This repo is one tumor member's work.
- Hard rules from the brief: classical methods only. NO deep learning, NO pre-trained models, NO TensorFlow, NO Haar, NO template matching.
- No two members share an algorithm in any phase, and each member uses different cases/patients.
- Test set >= 10 images. 70/30 train/test split, with 5-fold CV on the 70%. Split BY CASE (no slice of one patient in both sets).
- Fully commented, modular Python. Lecturer prefers clean, readable, simple code: few files, short functions, no clever tricks, no over-engineering, but nothing under-built.

## Our algorithm choices (assumptions made while user was away; change here + in code if groupmates clash)
- Cases: KiTS23 case_00100 .. case_00111 (assumed disjoint from groupmates).
- Phase 1: HU window (soft tissue, 40 +/- 200 -> [-160,240]) -> normalize [0,255]; Anisotropic diffusion (Perona-Malik) compared vs Gaussian and Median; CLAHE compared vs global Histogram Equalization (compared only: CLAHE amplifies noise and lowers Dice, so it is NOT in the final chain); Butterworth lowpass and homomorphic filter (frequency domain; homomorphic is compared only, it lowers region-growing Dice, see `results/phase1_chain_check.csv`); PSNR + CNR. Final chain = window -> anisotropic -> Butterworth.
- Phase 2: custom structuring elements (opening/closing); Strategy A = Multi-Otsu inside kidney ROI; Strategy B = seeded region growing from centroid of the darkest/least-enhancing Otsu class; Dice + IoU vs ground truth.
  Tumor search region = ground-truth kidney mask dilated slightly (tumor members assume the kidney is already localized by the kidney members).
- Phase 3: shape + GLCM (4 angles) + first-order HU features; classifier = Random Forest (sklearn); Healthy kidney tissue vs Tumor/Cyst; confusion matrix, accuracy, sensitivity, specificity, F1.

## Files (keep it this small)
| File | Job |
|---|---|
| `prepare_data.py` | Download KiTS23 cases 100-111, extract 2D axial slices to `data/slices/*.npz` |
| `preprocess.py` | Phase 1: `window_normalize`, `denoise`, `enhance_contrast`, `butterworth_lowpass`, `homomorphic_filter`, `psnr`, `cnr` |
| `segment.py` | Phase 2: `segment_otsu`, `segment_region_growing`, `refine_mask`, `dice`, `iou` |
| `classify.py` | Phase 3: `extract_features`, `build_dataset`, `train_and_evaluate` |
| `run_pipeline.py` | End to end on all slices, writes figures + metrics to `results/` |
| `app.py` | Tkinter prototype (see brief 7.2) |

## Data contract (`data/slices/case_XXXXX_zNNN.npz`)
`hu` float32 (H,W) raw Hounsfield; `kidney` bool (all kidney tissue incl. tumor/cyst); `tumor` bool (tumor + cyst labels); `case` int; `z` int.
`data/` is git-ignored (large). `results/` is committed (small figures + csv).

## Function signatures (shared contract between files)
- `preprocess.window_normalize(hu, lo=-160, hi=240) -> uint8`
- `preprocess.denoise(img_u8, method="anisotropic"|"gaussian"|"median") -> uint8`
- `preprocess.enhance_contrast(img_u8, method="clahe"|"he") -> uint8`
- `preprocess.butterworth_lowpass(img_u8, cutoff=60, order=2) -> uint8`
- `preprocess.homomorphic_filter(img_u8, gamma_low=0.5, gamma_high=1.5, cutoff=6, c=1.0) -> uint8` (illumination correction, not in `enhance`)
- `preprocess.enhance(hu) -> uint8` (final Phase 1 chain: window -> anisotropic -> Butterworth, no CLAHE)
- `segment.segment_tumor(enhanced_u8, kidney_mask, method="otsu"|"region_growing") -> bool mask`
- `segment.dice(pred, gt) -> float`, `segment.iou(pred, gt) -> float`
- `classify.extract_features(enhanced_u8, hu, mask) -> 1D np.ndarray`
- `classify.train_and_evaluate(X, y, groups) -> dict(metrics..., confusion_matrix, model)`

## Conventions
- Python 3.13, numpy / scipy / scikit-image / OpenCV / scikit-learn / matplotlib / nibabel only.
- Overlays: PREDICTED = red contour, GROUND TRUTH = green contour (brief 7.2).
- `pip install -r requirements.txt`; everything reproducible from `python prepare_data.py` then `python run_pipeline.py`.
- Commit small and often; only the main session pushes. Other sessions: do not commit, do not touch files outside your assignment.
- Report (~5000 words, APA refs), slides and video are done by the human; keep `results/` ready to paste in.

## Status (2026-11-13)
- Done: data (12 cases x 6 slices, cases 100-112 minus 106), Phases 1-3, `run_pipeline.py`, `app.py`, results in `results/`.
- Latest run (chain without CLAHE): region growing Dice 0.573 / IoU 0.486 (best; held-out cases 0.608), Otsu Dice 0.512 / IoU 0.408; classifier (region-growing masks) acc 0.905, recall 0.882, specificity 0.912, F1 0.811 (held-out cases, only 9 tumor regions).
- Known weakness (use in report critical analysis): tumors with HU similar to or brighter than parenchyma (cases 100, 103, 110) and tiny tumors fail; Phase 2 settings come from `phase2_sweep` (training cases only, `results/phase2_sweep.csv/png`): Otsu classes 2, `DARK_PREFERENCE` 0.2, tolerance 35, opening disk 3, closing ellipse (2,1), `MIN_TUMOR_PIXELS` 30 (unchanged, 10-60 tie). Tolerance/opening/closing sit at the grid edge. Before the sweep: train Dice 0.385, held-out 0.290.
- Also in `results/`: morphology_demo.png, robustness.csv (Dice stays ~0.57 under noise/brightness/contrast shifts), confusion_matrix.png, GLCM feature tables; `test_images/` = PNGs of held-out test slices (cases 104,107,111,112). `app.py` smoke-tested on real slices.
- TODO (human): report (~5000 words), slides, 10-15 min video, contribution matrix, make repo private.
