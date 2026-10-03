import json
import math

import numpy as np
import pytest

from metrics import confusion_matrix, format_report, save_report, segmentation_report

NAMES = {1: "bare_soil", 2: "vegetation"}


def test_perfect_prediction():
    y = np.array([[1, 1, 2], [2, 2, 1]])
    report = segmentation_report(y, y, NAMES)
    assert report["mean_iou"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["pixel_accuracy"] == 1.0
    assert report["coverage"] == 1.0


def test_known_values():
    # bare_soil: TP=2, FP=1, FN=1 → IoU=2/4, F1=4/6
    # vegetation: TP=3, FP=1, FN=1 → IoU=3/5, F1=6/8
    y_true = np.array([1, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([1, 1, 2, 2, 2, 2, 1])
    report = segmentation_report(y_true, y_pred, NAMES)

    soil, veg = report["per_class"]["bare_soil"], report["per_class"]["vegetation"]
    assert soil["iou"] == pytest.approx(0.5)
    assert soil["f1"] == pytest.approx(2 / 3)
    assert soil["precision"] == pytest.approx(2 / 3)
    assert soil["recall"] == pytest.approx(2 / 3)
    assert soil["support"] == 3
    assert veg["iou"] == pytest.approx(0.6)
    assert veg["f1"] == pytest.approx(0.75)
    assert report["mean_iou"] == pytest.approx(0.55)
    assert report["pixel_accuracy"] == pytest.approx(5 / 7)


def test_f1_equals_dice_and_relates_to_iou():
    rng = np.random.default_rng(0)
    y_true = rng.integers(1, 3, size=1000)
    y_pred = rng.integers(1, 3, size=1000)
    for m in segmentation_report(y_true, y_pred, NAMES)["per_class"].values():
        # F1 = 2·IoU / (1 + IoU)
        assert m["f1"] == pytest.approx(2 * m["iou"] / (1 + m["iou"]))


def test_unlabeled_pixels_are_ignored():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([2, 1, 1, 2])
    report = segmentation_report(y_true, y_pred, NAMES)
    assert report["pixels"] == 2
    assert report["mean_iou"] == 1.0


def test_uncovered_prediction_counts_as_miss():
    # -1 — пиксель не попал ни в один полигон: это FN, но не FP другим классам
    y_true = np.array([1, 1, 2, 2])
    y_pred = np.array([1, -1, 2, 2])
    report = segmentation_report(y_true, y_pred, NAMES)
    assert report["per_class"]["bare_soil"]["recall"] == pytest.approx(0.5)
    assert report["per_class"]["bare_soil"]["precision"] == 1.0
    assert report["per_class"]["vegetation"]["iou"] == 1.0
    assert report["coverage"] == pytest.approx(0.75)


def test_class_absent_everywhere_gives_nan_and_is_excluded_from_mean():
    y = np.array([2, 2, 2])
    report = segmentation_report(y, y, NAMES)
    assert math.isnan(report["per_class"]["bare_soil"]["iou"])
    assert report["mean_iou"] == 1.0


def test_confusion_matrix_layout():
    cm = confusion_matrix([1, 1, 2, 2], [1, 2, 2, -1], num_classes=3)
    # строки — истина (0,1,2), столбцы — предсказание (0,1,2,«не размечено»)
    assert cm[1].tolist() == [0, 1, 1, 0]
    assert cm[2].tolist() == [0, 0, 1, 1]


def test_shape_mismatch():
    with pytest.raises(ValueError):
        confusion_matrix([1, 2], [1], num_classes=3)


def test_format_and_save(tmp_path):
    report = segmentation_report([1, 2], [1, 2], NAMES)
    assert "mIoU = 1.000" in format_report(report)
    path = tmp_path / "m.json"
    save_report(report, path)
    assert json.loads(path.read_text(encoding="utf-8"))["mean_iou"] == 1.0
