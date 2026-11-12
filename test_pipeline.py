"""Automated checks. Run:  python -m pytest -q
Covers the brief's hard rules (no deep learning, split by patient, >= 10 test images) and each phase."""
import glob
import json
import subprocess
import sys

import numpy as np
import pytest

import classify
import preprocess
import segment

SLICES = sorted(glob.glob("data/slices/*.npz"))


@pytest.mark.parametrize("module", ["preprocess.py", "segment.py", "classify.py"])
def test_module_self_checks(module):
    result = subprocess.run([sys.executable, module], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_banned_libraries():
    """Brief: no deep learning, no TensorFlow, no Haar cascades, no template matching."""
    banned = ("tensorflow", "torch", "keras", "CascadeClassifier", "matchTemplate", "pretrained")
    for path in glob.glob("*.py"):
        if path == "test_pipeline.py":
            continue
        text = open(path, encoding="utf-8").read()
        assert not [b for b in banned if b in text], f"{path} uses a banned technique"


def test_window_normalize_range():
    out = preprocess.window_normalize(np.array([-1000.0, -160.0, 40.0, 240.0, 3000.0]))
    assert out.dtype == np.uint8 and out[0] == 0 and out[1] == 0 and out[-1] == 255 and out[-2] == 255


def test_dice_and_iou():
    a = np.zeros((10, 10), bool)
    a[:5] = True
    b = np.zeros((10, 10), bool)
    b[5:] = True
    assert segment.dice(a, a) == 1 and segment.iou(a, a) == 1
    assert segment.dice(a, b) == 0 and segment.iou(a, b) == 0
    assert segment.dice(a & b, a & b) == 1  # two empty masks count as a perfect match


def test_data_contract():
    assert len(SLICES) >= 60
    d = np.load(SLICES[0])
    assert d["hu"].shape == (512, 512) and d["kidney"].dtype == bool and d["tumor"].dtype == bool
    assert not (d["tumor"] & ~d["kidney"]).any(), "tumor must lie inside the kidney mask"


def test_enhance_output_shape():
    d = np.load(SLICES[0])
    out = preprocess.enhance(d["hu"])
    assert out.shape == d["hu"].shape and out.dtype == np.uint8


def test_segmentation_finds_a_large_tumor():
    """Case 102 has a big, clear tumor: both strategies must overlap it clearly."""
    d = np.load([p for p in SLICES if "case_00102" in p][0])
    enhanced = preprocess.enhance(d["hu"])
    for method in ("otsu", "region_growing"):
        pred = segment.segment_tumor(enhanced, d["kidney"], method)
        assert segment.dice(pred, d["tumor"]) > 0.4, method


def test_empty_kidney_gives_empty_mask():
    empty = np.zeros((64, 64), bool)
    assert not segment.segment_tumor(np.zeros((64, 64), np.uint8), empty, "otsu").any()


def test_features_have_expected_length():
    d = np.load(SLICES[0])
    enhanced = preprocess.enhance(d["hu"])
    f = classify.extract_features(enhanced, d["hu"], d["kidney"])
    assert f.shape == (len(classify.FEATURE_NAMES),) and np.isfinite(f).all()


def test_split_is_by_patient_and_has_enough_test_images():
    """Test patients must not appear in training, and the held-out set must cover >= 10 slices."""
    m = json.load(open("results/metrics.json"))
    held_out = [p for p in SLICES if int(np.load(p)["case"]) in m["test_cases"]]
    assert len(held_out) >= 10
    assert len(m["test_cases"]) >= 2


def test_metrics_file_is_valid():
    m = json.load(open("results/metrics.json"))
    for key in ("accuracy", "recall", "specificity", "f1"):
        assert 0 <= m[key] <= 1
    tn, fp, fn, tp = np.array(m["confusion_matrix"]).ravel()
    assert tn + fp + fn + tp == m["n_test"]
    assert abs(m["accuracy"] - (tp + tn) / m["n_test"]) < 1e-9


def test_sweep_table_exists_and_covers_all_filters():
    import pandas as pd
    table = pd.read_csv("results/parameter_sweep.csv")
    assert {"anisotropic (kappa)", "gaussian (sigma)", "median (kernel)", "butterworth (cutoff)",
            "clahe (clip limit)", "he (no setting)", "none (windowed only)"} <= set(table["filter"])
    assert table[["tumor_cnr", "edge", "dice"]].notna().all().all()


def test_app_builds_and_steps_through_slices():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    import app
    a = app.App(root)
    a.set_files(SLICES[:3])
    for _ in range(3):
        a.step(1)
    assert len(a.dices) == 3
    root.destroy()


def test_phase2_sweep_covers_groups_and_uses_training_cases_only():
    import pandas as pd
    table = pd.read_csv("results/phase2_sweep.csv")
    assert table["group"].nunique() == 6  # classes, dark preference, tolerance, opening, closing, min pixels
    assert table[["dice", "iou"]].notna().all().all()
    test_cases = json.load(open("results/metrics.json"))["test_cases"]
    train = [p for p in glob.glob("data/slices/*.npz") if int(np.load(p)["case"]) not in test_cases]
    assert set(table["n_train_slices"]) == {len(train)}  # the sweep scored exactly the non-test slices
