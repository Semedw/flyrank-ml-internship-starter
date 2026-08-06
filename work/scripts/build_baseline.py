"""Phase 2 — transparent rule baseline on the same frame and split as the model.

The baseline is a hand-written, explainable review-priority score built ONLY
from feature-window metadata (nothing from the label window). The model must
beat it on the same test rows and the same metrics. Reason codes explain why a
page was ranked. No label-window columns (_lw) are ever read by this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import auc_ap, precision_at_k

FEATURES_PATH = Path(__file__).resolve().parents[1] / "work" / "outputs" / "capstone" / "features.parquet"
OUT_QUEUE = Path(__file__).resolve().parents[1] / "work" / "outputs" / "capstone" / "baseline_queue.csv"
OUT_METRICS = Path(__file__).resolve().parents[1] / "work" / "outputs" / "capstone" / "baseline_metrics.json"

TEST_DECISION = "2026-05-31"  # sealed test decision; label window = June 2026

# documented policy thresholds (choose-and-defend choices)
VISIBLE_IMP = 500
POS_CEILING = 20
CTR_LOW = 1.0        # CTR% below which a visible page is a ctr_fix_candidate
STALE_DAYS = 180
AGING_DAYS = 90
THIN_WORDS = 1200

REASON_COLUMNS = ["aging_visible", "stale_visible", "ctr_fix_candidate", "thin_visible", "page_one_decay_risk"]


def expected_ctr(pos: pd.Series) -> pd.Series:
    """Expected CTR% by position (documented heuristic): 30 * pos^-0.6."""
    return 100 * 0.30 * np.power(np.maximum(pos.fillna(10), 0.5), -0.6)


def days_stale(df: pd.DataFrame) -> pd.Series:
    """Days since the content last changed; falls back to age (feature-window time)."""
    if "days_since_content_updated" in df.columns:
        return df["days_since_content_updated"].fillna(0)
    if "days_since_content_created" in df.columns:
        return df["days_since_content_created"].fillna(0)
    return pd.Series(0, index=df.index)


def sub_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Four normalized [0,1] signals that decide review priority."""
    imp = df["imp_fw"].replace([np.inf, -np.inf], np.nan).fillna(0)
    vis = (np.log1p(imp) / np.log1p(5000)).clip(0, 1)
    fresh = (days_stale(df) / STALE_DAYS).clip(0, 1)
    ctr_gap = ((expected_ctr(df["pos_fw"]) - df["ctr_fw_pct"].fillna(0)).clip(lower=0) / 10.0).clip(0, 1)
    sessions = df.get("ga4_sessions_fw", pd.Series(0.0, index=df.index)).fillna(0)
    se_share = sessions / imp.clip(lower=1)
    eng = (1 - (se_share / 0.02).clip(0, 1))
    return pd.DataFrame({"vis": vis, "fresh": fresh, "ctr_gap": ctr_gap, "eng": eng}, index=df.index)


def score_pages(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    sub = sub_scores(df)
    d["score"] = 100 * (0.30 * sub["fresh"] + 0.30 * sub["ctr_gap"] + 0.25 * sub["vis"] + 0.15 * sub["eng"])
    return d


def reason_codes(df: pd.DataFrame) -> pd.DataFrame:
    stale = days_stale(df)
    codes = pd.DataFrame(index=df.index)
    codes["aging_visible"] = (stale >= AGING_DAYS) & (df["imp_fw"] >= VISIBLE_IMP)
    codes["stale_visible"] = (stale >= STALE_DAYS) & (df["imp_fw"] >= VISIBLE_IMP)
    codes["ctr_fix_candidate"] = (
        (df["imp_fw"] >= VISIBLE_IMP)
        & (df["pos_fw"] >= 1)
        & (df["pos_fw"] <= POS_CEILING)
        & (df["ctr_fw_pct"] < CTR_LOW)
    )
    thin_column = df["word_count"] if "word_count" in df.columns else pd.Series(0, index=df.index)
    codes["thin_visible"] = (thin_column > 0) & (thin_column < THIN_WORDS) & (df["imp_fw"] >= VISIBLE_IMP)
    codes["page_one_decay_risk"] = (df["pos_fw"] >= 1) & (df["pos_fw"] <= 10) & (stale >= STALE_DAYS)
    codes["monitor"] = ~codes[REASON_COLUMNS].any(axis=1)
    return codes


def compute_metrics(y: pd.Series, score: pd.Series) -> dict:
    out = {"rows": int(len(y)), "base_rate": float(y.mean()), "n_pos": int(y.sum())}
    a, p = auc_ap(y, score)
    out["ROC_AUC"] = round(a, 4)
    out["average_precision"] = round(p, 4)
    for k in (5, 20, 50):
        out[f"precision@{k}"] = round(precision_at_k(y, score, k), 4)
    return out


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    scored = score_pages(df)
    scored = pd.concat([scored, reason_codes(scored)], axis=1)

    test = scored[(scored["decision_date"] == TEST_DECISION) & scored["decline_30d"].notna()].copy()
    m = compute_metrics(test["decline_30d"], test["score"])

    Path(OUT_QUEUE.parent).mkdir(parents=True, exist_ok=True)
    out_cols = ["client_hash_id", "content_hash_id", "decision_date", "score"] + REASON_COLUMNS + ["monitor"]
    scored.sort_values("score", ascending=False).to_csv(OUT_QUEUE, columns=out_cols, index=False)
    OUT_METRICS.write_text(json.dumps(m, indent=2) + "\n")

    print(f"baseline on test ({TEST_DECISION}):")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()