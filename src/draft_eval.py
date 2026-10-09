"""Metrics shared by the draft model scripts."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def ece_quantile(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-count bins (equal-width bins are uninformative when predictions sit near 0.5)."""
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in bins))


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p) if np.std(p) > 0 else 0.5,
        "accuracy": float(((p > 0.5) == y).mean()),
        "ece_q10": ece_quantile(y, p),
        "pred_std": float(np.std(p)),
    }


def per_row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_bootstrap_ci(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    seed: int,
    n_bootstrap: int = 2000,
) -> tuple[float, float, float]:
    """Mean of (loss_a - loss_b) with a 95% bootstrap CI. Positive means model b is better."""
    rng = np.random.default_rng(seed)
    diff = loss_a - loss_b
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_bootstrap)])
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
