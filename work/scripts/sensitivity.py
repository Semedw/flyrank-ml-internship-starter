"""Phase 3b — label-threshold sensitivity analysis (light, cached frame only).

Re-derives the decline label with alternative thresholds (clicks and impressions
drop ratios) on the SEALED test decision (D=2026-05-31) and re-fits a light
RandomForest on the TRAIN decisions with the same re-derived label, reporting
how test precision@50 and ROC-AUC move. Purpose: show the headline numbers are
not an artifact of the single 70%-of-prior threshold. No new data is read;
June is only ever the label window of the sealed test decision.

Writes work/outputs/capstone/sensitivity.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = Path(__file__).resolve().parents[1] / "work" / "outputs" / "capstone" / "sensitivity.json"

TEST_DECISION = "2026-05-31"
TRAIN_DECISIONS = ["2026-02-28", "2026-03-31", "2026-04-30"]
RANDOM_STATE = 42
N_ESTIMATORS = 150
TOP_K = 50


def _features(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in df.columns if c.endswith(("_fw", "_lw", "_ratio", "_share", "_flag"))] \
        + [c for c in df.columns if c in (
            "word_count", "char_count", "keyword_token_count", "url_char_count",
            "search_volume", "competition", "cpc", "backlinks", "category_count",
            "content_type", "main_intent", "competition_level", "provider_used", "model_used",
        )]
    feats += [c for c in df.columns if c.startswith("days_since_")]
    return df[feats]


def _numeric(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype.kind in "ifb"]


def _fit_eval(df: pd.DataFrame, drop_ratio: float, floor: int) -> dict:
    fw = df[df.decision_date < TEST_DECISION].copy()
    te = df[df.decision_date == TEST_DECISION].copy()

    def derive(d):
        e = (d["clk_fw"] >= 30) | (d["imp_fw"] >= 300)
        prior_clk = np.maximum(d["clk_fw"], 30)
        prior_imp = np.maximum(d["imp_fw"], 300)
        y = (d["clk_lw"] < drop_ratio * prior_clk) | (d["imp_lw"] < drop_ratio * prior_imp)
        y = np.where(e, y, np.nan).astype(float)
        return y

    fw["y"] = derive(fw)
    te["y"] = derive(te)

    keep = ~fw["y"].isna()
    Xtr, ytr = _features(fw[keep]), fw.loc[keep, "y"].astype(int)
    Xte, yte = _features(te), te["y"].astype(int)
    yte = yte[~Xte.isna().any(axis=1)]  # sealed test: no imputation on features
    Xte = Xte.loc[yte.index]

    num = _numeric(Xtr)
    cat = [c for c in Xtr.columns if c not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
    model = Pipeline([("pre", pre), ("clf", clf)])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(iter(gss.split(Xtr, ytr, groups=fw.loc[keep, "client_hash_id"])))
    model.fit(Xtr.iloc[tr_idx], ytr.iloc[tr_idx])
    pva = model.predict_proba(Xtr.iloc[va_idx])[:, 1]
    va_auc = roc_auc_score(ytr.iloc[va_idx], pva)

    model.fit(Xtr, ytr)
    pte = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, pte)
    p50 = (yte.iloc[np.argsort(-pte)[:TOP_K]]).mean()

    return {
        "drop_ratio": drop_ratio,
        "floor": floor,
        "label_rate_train": float(ytr.mean()),
        "label_rate_test": float(yte.mean()),
        "val_auc": round(float(va_auc), 4),
        "test_auc": round(float(auc), 4),
        f"test_precision_at_{TOP_K}": round(float(p50), 4),
    }


def main() -> None:
    frame = pd.read_parquet(Path(__file__).resolve().parents[1] / "work" / "outputs" / "capstone" / "frame.parquet")
    results = [_fit_eval(frame, r, f) for r, f in [(0.50, 30), (0.70, 30), (0.85, 30), (0.70, 15)]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print("sensitivity grid:")
    for r in results:
        print(" ", r)


if __name__ == "__main__":
    main()
