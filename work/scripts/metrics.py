"""Shared evaluation helpers for the capstone: precision@K, ROC-AUC, AP.

All helpers take observed 0/1 labels and a ranking score. Base-rate-aware
callers print the majority-class rate next to every headline number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_k(y: pd.Series, scores: pd.Series, k: int) -> float:
    y = y.astype(float)
    s = scores.fillna(0).astype(float)
    order = s.sort_values(ascending=False).index[:k]
    return float(y.loc[order].mean())


def auc_ap(y: pd.Series, scores: pd.Series) -> tuple[float, float]:
    y = y.astype(float)
    s = scores.fillna(0).astype(float)
    if y.nunique() < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def precision_curve(y: pd.Series, scores: pd.Series, ks: list[int]) -> dict[str, float]:
    return {f"precision@{k}": precision_at_k(y, scores, k) for k in ks}