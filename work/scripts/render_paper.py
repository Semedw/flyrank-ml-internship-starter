"""Phase 6 — render the deployed research paper from the committed receipts.

Reads the metrics JSONs + charts from work/outputs/capstone and work/figures,
fills the placeholders in work/paper/template.html, and writes docs/index.html
(GitHub Pages root). The numbers on the page are therefore always the numbers
in the receipts — one source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "work/outputs/capstone"
TEMPLATE = ROOT / "work/paper/template.html"
OUT = ROOT / "docs/index.html"


def collect() -> dict:
    models = pd.read_json(RESULTS / "model_results.json").set_index("model")
    baseline = json.loads((RESULTS / "baseline_metrics.json").read_text())
    robustness = pd.read_json(RESULTS / "client_holdout_robustness.json")

    df = pd.read_parquet(RESULTS / "features.parquet", columns=["decision_date", "decline_30d", "evidence", "client_hash_id"])
    test = df[df["decision_date"] == "2026-05-31"]
    train = df[df["decision_date"] != "2026-05-31"]
    by_month = df.groupby("decision_date")["decline_30d"].mean()

    b = baseline
    rf = models.loc["random_forest"]
    lr = models.loc["logistic_regression"]
    dt = models.loc["decision_tree"]
    n_test = int(test["decline_30d"].notna().sum())
    n_train = int(train["decline_30d"].notna().sum())

    lift20 = rf["precision@20"] / b["precision@20"] if b["precision@20"] else float("nan")
    lift50 = rf["precision@50"] / b["precision@50"] if b["precision@50"] else float("nan")
    auc_gain = rf["ROC_AUC"] - b["ROC_AUC"]

    return {
        "N_TRAIN": f"{n_train:,}",
        "N_TEST": f"{n_test:,}",
        "BASE_RATE": f"{b['base_rate']:.2%}",
        "BASE_RATE_PCT": f"{b['base_rate']*100:.0f}%",
        "BL_AUC": f"{b['ROC_AUC']:.3f}",
        "BL_P20": f"{b['precision@20']:.3f}",
        "BL_P50": f"{b['precision@50']:.3f}",
        "RF_AUC": f"{rf['ROC_AUC']:.3f}",
        "RF_P20": f"{rf['precision@20']:.3f}",
        "RF_P50": f"{rf['precision@50']:.3f}",
        "LR_AUC": f"{lr['ROC_AUC']:.3f}",
        "DT_AUC": f"{dt['ROC_AUC']:.3f}",
        "LIFT_P20": f"{lift20:.1f}x",
        "LIFT_P50": f"{lift50:.1f}x",
        "AUC_GAIN": f"{auc_gain:+.3f}",
        "GSS_MED_AUC": f"{robustness['ROC_AUC'].median():.3f}",
        "GSS_MIN_AUC": f"{robustness['ROC_AUC'].min():.3f}",
        "GSS_MAX_AUC": f"{robustness['ROC_AUC'].max():.3f}",
        "FEB_RATE": f"{by_month.get('2026-02-28', float('nan')):.2%}",
        "MAR_RATE": f"{by_month.get('2026-03-31', float('nan')):.2%}",
        "APR_RATE": f"{by_month.get('2026-04-30', float('nan')):.2%}",
        "MAY_RATE": f"{by_month.get('2026-05-31', float('nan')):.2%}",
        "N_CLIENTS": str(df["client_hash_id"].nunique()),
        "N_EVIDENCE": f"{int(df['evidence'].sum()):,}",
        "RF_P50_PAGES": str(round(rf["precision@50"] * 50)),
        "BL_P50_PAGES": str(round(b["precision@50"] * 50)),
    }


def main() -> None:
    if not (RESULTS / "model_results.json").exists():
        raise SystemExit("run the pipeline first: build_frame -> build_features -> build_baseline -> train_model -> make_charts")
    vals = collect()
    html = TEMPLATE.read_text()
    missing = [k for k in vals if f"{{{{{k}}}}}" not in html]
    if missing:
        raise SystemExit(f"template is missing placeholders: {missing}")
    for k, v in vals.items():
        html = html.replace(f"{{{{{k}}}}}", v)
    unused = [line.strip()[:60] for line in html.splitlines() if "{{" in line]
    if unused:
        raise SystemExit(f"template has unused placeholders:\n" + "\n".join(unused))
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # copy the charts the page embeds (relative paths)
    figures = ROOT / "work/figures"
    out_figs = OUT.parent / "figures"
    out_figs.mkdir(parents=True, exist_ok=True)
    for f in figures.glob("*.svg"):
        (out_figs / f.name).write_bytes(f.read_bytes())

    OUT.write_text(html)
    print(f"wrote {OUT} with {len(vals)} values filled; figures -> {out_figs}")


if __name__ == "__main__":
    main()