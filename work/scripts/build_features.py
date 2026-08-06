"""Phase 1b — label definition, feature engineering, leakage asserts.

Reads the cached frame (work/outputs/capstone/frame.parquet), defines the
observed future-outcome label, engineers the feature matrix, and runs the
window-adjacency + no-leakage asserts the paper's methodology section
reports. Everything here must be re-runnable from the cached frame.

Label (one sentence, from the contract):
  decline_30d = 1 if the page's clicks in the 30 days after the decision moment
  fell below 70% of its prior-30-day clicks (clicks floor 30), or — for
  low-click pages — its impressions fell below 70% of prior-30-day impressions
  (impressions floor 300). The label is an OBSERVED future outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_PATH = Path(__file__).resolve().parents[1] / "work" / "outputs" / "capstone" / "frame.parquet"
OUT_PATH = Path(__file__).resolve().parents[1] / "work" / "outputs" / "capstone" / "features.parquet"
LABEL_META_PATH = Path(__file__).resolve().parents[1] / "work" / "outputs" / "capstone" / "label_meta.json"

CLICK_FLOOR = 30        # enough click evidence to trust a click-based ratio
IMPRESSION_FLOOR = 300  # fallback evidence floor for impression-based ratio
DROP_RATIO = 0.70       # label = 1 if next-window volume < 70% of feature-window volume

# numeric dim-metadata columns (may be missing for some rows; has_ flags exist)
DIM_NUMERIC = ["word_count", "char_count", "keyword_token_count", "url_char_count",
               "search_volume", "competition", "cpc", "backlinks", "category_count"]
DIM_CATEGORICAL = ["content_type", "main_intent", "competition_level", "provider_used", "model_used"]

# columns that must never appear as features (label window / identity)
FORBIDDEN_FEATURES = ("_lw", "client_hash_id", "content_hash_id", "decision_date")


def _load() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        raise SystemExit(f"frame cache missing — run build_frame.py first ({CACHE_PATH})")
    return pd.read_parquet(CACHE_PATH)


def _window_asserts(frame: pd.DataFrame) -> dict:
    feats = [c for c in frame.columns if c.endswith("_fw") and not c.startswith("days_since")]
    labs = [c for c in frame.columns if c.endswith("_lw")]
    overlap = sorted(set(feats) & set(labs))
    assert not overlap, f"feature/label column overlap: {overlap}"
    assert len(labs) >= 3, f"expected label-window columns, got {labs}"
    return {"feature_cols": feats, "label_cols": labs}


def define_label(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    clk_ev = df["clk_fw"] >= CLICK_FLOOR
    imp_ev = df["imp_fw"] >= IMPRESSION_FLOOR
    evidence = clk_ev | imp_ev

    click_decline = clk_ev & (df["clk_lw"] < DROP_RATIO * df["clk_fw"])
    imp_decline = (~clk_ev) & imp_ev & (df["imp_lw"] < DROP_RATIO * df["imp_fw"])

    df["evidence"] = evidence.astype(int)
    df["decline_30d"] = np.where(evidence, (click_decline | imp_decline).astype(int), np.nan).astype(float)
    return df


def engineer_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = frame.copy().replace([np.inf, -np.inf], np.nan)

    # --- capture + engagement (feature window only) -------------------------
    df["ctr_fw_pct"] = np.where(df["imp_fw"] > 0, 100.0 * df["clk_fw"] / df["imp_fw"], np.nan)
    df["days_imp_ratio"] = df["days_imp_fw"] / 30.0
    df["days_clk_ratio"] = df["days_clk_fw"] / 30.0
    df["days_pos_ratio"] = df["days_pos_fw"] / 30.0

    # --- log transforms ------------------------------------------------------
    for c in ["imp_fw", "clk_fw", "ga4_sessions_fw", "sessions_ai_fw", "scroll_events_fw"]:
        if c in df.columns:
            df[f"log1p_{c}"] = np.log1p(df[c].fillna(0))
    for c in ["search_volume", "backlinks", "cpc", "competition", "category_count", "keyword_token_count"]:
        if c in df.columns:
            df[f"log1p_{c}"] = np.log1p(df[c].fillna(0))

    # --- GA4 availability (zero-filled before ga4 start; flag it, don't hide it)
    if "days_ga4_fw" in df.columns:
        df["has_ga4"] = (df["days_ga4_fw"] > 0).astype(int)
        df["ga4_share"] = df["days_ga4_fw"] / df["days_imp_fw"].clip(lower=1)
    if "ga4_sessions_fw" in df.columns:
        df["sess_per_imp"] = np.where(df["imp_fw"] > 0, df["ga4_sessions_fw"] / df["imp_fw"], 0.0)

    # --- has_ flags for metadata (missing is informative, don't blind-fill) --
    for c in DIM_NUMERIC:
        if c in df.columns:
            df[f"has_{c}"] = df[c].notna().astype(int)

    # --- feature bookkeeping --------------------------------------------------
    drop_cols = [c for c in df.columns if c.endswith("_lw") or c in FORBIDDEN_FEATURES]
    features = [c for c in df.columns if c not in drop_cols]
    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_features if c not in ("evidence", "decline_30d")]
    return df, numeric_features


def main() -> None:
    frame = _load()
    checks = _window_asserts(frame)
    df = define_label(frame)
    df, numeric_features = engineer_features(df)

    categoricals = [c for c in DIM_CATEGORICAL if c in df.columns]

    Path(OUT_PATH.parent).mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    meta = {
        "rows": int(len(df)),
        "labelled": int(df["evidence"].sum()),
        "base_rate": float(df["decline_30d"].mean()),
        "click_floor": CLICK_FLOOR,
        "impression_floor": IMPRESSION_FLOOR,
        "drop_ratio": DROP_RATIO,
        "decline_by_month": {str(k): round(v, 4) for k, v in df.groupby("decision_date")["decline_30d"].mean().items()},
        "labelled_by_month": {str(k): int(v) for k, v in df.groupby("decision_date")["evidence"].sum().items()},
        "numeric_features": numeric_features,
        "categorical_features": categoricals,
        "window_asserts": checks,
    }
    LABEL_META_PATH.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"rows: {len(df):,} | labelled: {int(df['evidence'].sum()):,} ({df['evidence'].mean():.1%}) | base rate: {df['decline_30d'].mean():.3f}")
    print(f"decline by decision month:\n{df.groupby('decision_date')['decline_30d'].agg(['count','mean']).round(3).to_string()}")
    print(f"\nnumeric features ({len(numeric_features)}): {', '.join(numeric_features)}")
    print(f"categorical features ({len(categoricals)}): {', '.join(categoricals)}")


if __name__ == "__main__":
    main()