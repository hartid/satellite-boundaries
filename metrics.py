"""Метрики качества сегментации: IoU, F1, precision, recall, pixel accuracy.

Все функции работают с целочисленными масками классов (numpy-массивы любой формы).
Пиксели со значением ``ignore_index`` в эталоне (нет разметки) не учитываются.
"""

import json

import numpy as np

IGNORE_INDEX = 0


def confusion_matrix(y_true, y_pred, num_classes, ignore_index=IGNORE_INDEX):
    """Матрица ошибок [истинный класс, предсказанный класс] размера num_classes × num_classes.

    Предсказание вне диапазона [0, num_classes) (например, -1 — «не размечено моделью»)
    считается промахом: пиксель попадает в отдельный столбец и отбрасывается,
    но остаётся в support своего истинного класса (FN).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Размеры не совпадают: {y_true.shape} и {y_pred.shape}")

    keep = (y_true != ignore_index) & (y_true >= 0) & (y_true < num_classes)
    y_true = y_true[keep].astype(np.int64)
    y_pred = y_pred[keep].astype(np.int64)

    # Отдельный «мусорный» столбец num_classes для предсказаний вне диапазона
    y_pred = np.where((y_pred >= 0) & (y_pred < num_classes), y_pred, num_classes)
    cm = np.bincount(
        y_true * (num_classes + 1) + y_pred, minlength=num_classes * (num_classes + 1)
    ).reshape(num_classes, num_classes + 1)
    return cm


def _safe_div(num, den):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def segmentation_report(y_true, y_pred, class_names, ignore_index=IGNORE_INDEX):
    """Считает метрики по классам и сводные.

    class_names: dict {id класса: название}. id ignore_index в отчёт не входит.

    Для каждого класса c:
        IoU       = TP / (TP + FP + FN)
        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)
        F1        = 2·TP / (2·TP + FP + FN)   (= коэффициент Dice)
    Сводные: mIoU и macro-F1 — среднее по классам, pixel_accuracy — доля верных пикселей.
    """
    num_classes = max(class_names) + 1
    cm = confusion_matrix(y_true, y_pred, num_classes, ignore_index)

    support = cm.sum(axis=1)                 # пикселей класса в эталоне
    tp = np.diag(cm[:, :num_classes])
    fp = cm[:, :num_classes].sum(axis=0) - tp
    fn = support - tp

    iou = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)

    ids = [c for c in sorted(class_names) if c != ignore_index]
    per_class = {
        class_names[c]: {
            "iou": float(iou[c]),
            "f1": float(f1[c]),
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "support": int(support[c]),
        }
        for c in ids
    }
    present = [c for c in ids if support[c] > 0]
    total = int(support[ids].sum())

    return {
        "per_class": per_class,
        "mean_iou": float(np.nanmean(iou[present])) if present else float("nan"),
        "macro_f1": float(np.nanmean(f1[present])) if present else float("nan"),
        "pixel_accuracy": float(tp[ids].sum() / total) if total else float("nan"),
        "coverage": float(cm[ids, :num_classes].sum() / total) if total else float("nan"),
        "pixels": total,
        "confusion_matrix": {
            "labels": [class_names[c] for c in ids] + ["<не размечено>"],
            "matrix": cm[np.ix_(ids, ids + [num_classes])].tolist(),
        },
    }


def format_report(report, title="Метрики качества"):
    """Человекочитаемая таблица для вывода в консоль."""
    lines = [f"=== {title} ===", f"{'класс':<14}{'IoU':>8}{'F1':>8}{'Prec':>8}{'Recall':>8}{'пикселей':>12}"]
    for name, m in report["per_class"].items():
        lines.append(
            f"{name:<14}{m['iou']:>8.3f}{m['f1']:>8.3f}{m['precision']:>8.3f}"
            f"{m['recall']:>8.3f}{m['support']:>12,}".replace(",", " ")
        )
    lines.append(
        f"mIoU = {report['mean_iou']:.3f} | macro-F1 = {report['macro_f1']:.3f} | "
        f"pixel accuracy = {report['pixel_accuracy']:.3f} | покрытие = {report['coverage']:.1%}"
    )
    return "\n".join(lines)


def save_report(report, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
