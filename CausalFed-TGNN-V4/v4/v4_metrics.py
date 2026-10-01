"""Dependency-light open-set metrics used by V4 audit scripts."""

from __future__ import annotations

import numpy as np


def _auc(y: np.ndarray, score: np.ndarray) -> float | None:
    order = np.argsort(-score, kind="mergesort")
    y = y[order]
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    if not positives or not negatives:
        return None
    ranks = np.arange(1, len(y) + 1, dtype=float)
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    order = np.argsort(-score, kind="mergesort")
    y = y[order].astype(int)
    if not y.any():
        return None
    cumulative = np.cumsum(y)
    precision = cumulative / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def binary_metrics(labels, probabilities, threshold=0.5) -> dict[str, float | int | None]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = p >= threshold
    tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {"accuracy": float((pred == y).mean()), "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12), "auroc": _auc(y, p), "aupr": average_precision(y, p), "support": int(len(y))}


def fpr_at_tpr(labels, scores, target_tpr=0.95) -> float | None:
    y = np.asarray(labels, dtype=int); s = np.asarray(scores, dtype=float)
    positives = max(int(y.sum()), 1); negatives = max(int((y == 0).sum()), 1)
    order = np.argsort(-s, kind="mergesort"); y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    valid = np.flatnonzero(tp / positives >= target_tpr)
    return float(fp[valid[0]] / negatives) if len(valid) else None


def open_set_metrics(unknown, scores, threshold) -> dict[str, float | int | None]:
    y = np.asarray(unknown, dtype=int); s = np.asarray(scores, dtype=float); flags = s >= threshold
    return {"unknown_accuracy": float((flags == y).mean()), "unknown_f1": binary_metrics(y, flags.astype(float), 0.5)["f1"], "unknown_auroc": _auc(y, s), "unknown_aupr": average_precision(y, s), "unknown_fpr95": fpr_at_tpr(y, s), "threshold": float(threshold), "unknown_support": int(y.sum())}
