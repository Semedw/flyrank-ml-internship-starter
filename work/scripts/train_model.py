"""Phase 3 — honest model comparison on the sealed test split.

Design (documented in the paper's methodology):
- Train decisions: 2026-02-28, 2026-03-31, 2026-04-30 (labels Mar/Apr/May).
- Test decision:  2026-05-31 ONCE, label window = June 2026 (sealed).
- No hyperparameter tuning: default sklearn settings, random_state=42 — the
  sealed test is evaluated exactly once, blind.
- Robustness check: GroupShuffleSplit on client_hash_id inside the training
  months (does the pattern survive unseen clients?).
- Leakage asserts: no _lw columns, no hash columns, no future rows in train.

Low-evidence rows (label NaN) are excluded from supervised learning and kept
in the ranked queue as monitor items.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from metrics import auc_ap, precision_at_k

FEATURES_PATH = Path(__file__).resolve().parents[1] / "work" / "outputs" / "capstone" / "features.parquet"
META_PATH = Path(__file__).resolve().parents[1] / "work" / "outputs" / "capstone" / "label_meta.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "work" / "outputs" / "capstone"
TEST_DECISION = "2026-05-31"
RANDOM_STATE = 42

LABEL = "decline_30d"


def load() -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_parquet(FEATURES_PATH)
    meta = json.loads(META_PATH.read_text())
    avail_num = [c for c in meta["numeric_features"] if c in df.columns]
    avail_cat = [c for c in meta["categorical_features"] if c in df.columns]

    feature_cols = avail_num + avail_cat
    assert not any(c.endswith("_lw") for c in feature_cols), f"label-window leak in features: {feature_cols}"
    assert not any("hash" in c for c in feature_cols), f"identity leak in features: {feature_cols}"
    return df, avail_num, avail_cat


def make_preprocessor(avail_num: list[str], avail_cat: list[str]) -> ColumnTransformer:
    transformers = [("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), avail_num)]
    if avail_cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), avail_cat))
    return ColumnTransformer(transformers, remainder="drop")


def make_models (avail_num, avail_cat) -> dict[str, Pipeline]:
    pre = make_preprocessor(avail_num, avail_cat)

    def pipe(estimator):
        return Pipeline([("pre", pre), ("model", estimator)])

    return {
        "logistic_regression": pipe(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "decision_tree": pipe(DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=RANDOM_STATE)),
        "random_forest": pipe(
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
        ),
    }


def make_x(df: pd.DataFrame, avail_num: list[str], avail_cat: list[str]) -> pd.DataFrame:
    if not avail_cat:
        return df[avail_num]
    x = df[avail_num].copy()
    for c in avail_cat:
        x[c] = df[c].astype(str)
    return x


def row_metrics(y: pd.Series, proba: np.ndarray) -> dict:
    s = pd.Series(proba, index=y.index)
    auc, ap = auc_ap(y, s)
    return {
        "ROC_AUC": round(auc, 4),
        "average_precision": round(ap, 4),
        "precision@5": round(precision_at_k(y, s, 5), 4),
        "precision@20": round(precision_at_k(y, s, 20), 4),
        "precision@50": round(precision_at_k(y, s, 50), 4),
    }


def evaluate(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, idx: pd.Index) -> pd.DataFrame:
    rows = []
    for name, pipe in models.items():
        proba = pipe.predict_proba(X.loc[idx])[:, 1]
        m = row_metrics(y.loc[idx], proba)
        rows.append({"model": name, **m})
    return pd.DataFrame(rows)


def main() -> None:
    df, avail_num, avail_cat = load()
    df = df.dropna(subset=[LABEL]).copy()
    df[LABEL] = df[LABEL].astype(int)

    train = df[df["decision_date"] != TEST_DECISION]
    test = df[df["decision_date"] == TEST_DECISION]
    assert (train["decision_date"].astype(str) < TEST_DECISION).all(), "future rows in train!"

    X = make_x(df, avail_num, avail_cat)
    print(f"train rows: {len(train):,} | test rows: {len(test):,} | test base rate: {test[LABEL].mean():.3f}")
    print(f"numeric features ({len(avail_num)}): {', '.join(avail_num)}")

    models = make_models(avail_num, avail_cat)
    for name, pipe in models.items():
        pipe.fit(X.loc[train.index], train[LABEL])

    results = evaluate(models, X, df[LABEL], test.index)
    print("\nsealed-test results (evaluated once, blind):")
    print(results.to_string(index=False))

    # robustness: client-holdout within training months only (never touches test rows)
    rf = models["random_forest"]
    train_idx_all = df.index[df["decision_date"] != TEST_DECISION].to_numpy()
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_STATE)
    gss_rows = []
    for fold, (tr_pos, va_pos) in enumerate(
        gss.split(X.loc[train_idx_all], df.loc[train_idx_all, LABEL], df.loc[train_idx_all, "client_hash_id"])
    ):
        tr = train_idx_all[tr_pos]
        va = train_idx_all[va_pos]
        sub_run = {"random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
        )}
        sub_run["random_forest"].fit(X.loc[tr], df.loc[tr, LABEL])
        r = evaluate(sub_run, X, df[LABEL], va).iloc[0].to_dict()
        r["fold"] = fold
        gss_rows.append(r)
    gss_df = pd.DataFrame(gss_rows)
    print("\nclient-holdout robustness (random forest, train months):")
    print(gss_df[["fold", "ROC_AUC", "average_precision", "precision@20", "precision@50"]].round(4).to_string(index=False))

    # feature importances
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=rf.named_steps["pre"].get_feature_names_out())
    print("\ntop 12 features by importance:")
    print(imp.sort_values(ascending=False).head(12).round(4).to_string())

    # ranked test predictions
    test_frame = test.copy()
    test_frame["model_prob"] = rf.predict_proba(X.loc[test.index])[:, 1]
    test_frame.sort_values("model_prob", ascending=False).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    results.to_json(OUT_DIR / "model_results.json", orient="records", indent=2)
    gss_df.round(4).to_json(OUT_DIR / "client_holdout_robustness.json", orient="records", indent=2)
    imp.sort_values(ascending=False).head(20).round(4).to_json(OUT_DIR / "feature_importances.json")
    print("\nwrote model_results.json, client_holdout_robustness.json, feature_importances.json, test_predictions.csv")


if __name__ == "__main__":
    main()