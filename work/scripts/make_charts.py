"""Phase 4 — export the paper's figures from the committed JSON receipts.

Reads work/outputs/capstone/*.json + features.parquet and writes charts to
work/figures/ as SVG (and PNG for the paper). One message per chart, takeaway
under the chart, axis labels a stranger understands.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "work" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = Path(__file__).resolve().parents[1] / "work" / "outputs" / "capstone"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "svg.fonttype": "none",
    }
)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def load_results() -> dict:
    models = pd.read_json(RESULTS / "model_results.json").set_index("model")
    baseline = json.loads((RESULTS / "baseline_metrics.json").read_text())
    return {"models": models, "baseline": baseline}


def chart_precision_at_k(results: dict) -> None:
    ks = [5, 20, 50]
    rf = results["models"].loc["random_forest"]
    b = results["baseline"]
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(x, [rf[f"precision@{k}"] for k in ks], "o-", label="Random forest", color="#1f77b4", lw=2)
    ax.plot(x, [b[f"precision@{k}"] for k in ks], "s--", label="Rule baseline", color="#d62728", lw=2)
    ax.axhline(b["base_rate"], ls=":", color="gray", lw=1.2)
    ax.text(2.02, b["base_rate"], f"base rate {b['base_rate']:.2f}", color="gray", va="center", fontsize=9)
    ax.set_xticks(x, [f"@{k}" for k in ks])
    ax.set_xlabel("Top-K of the review queue")
    ax.set_ylabel("Precision (share of top-K that declined)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("Precision@K on the sealed June test month: model vs baseline")
    save(fig, "precision_at_k")


def chart_model_auc(results: dict) -> None:
    models = results["models"]
    names = models.index.tolist()
    aucs = models["ROC_AUC"].tolist()
    b = results["baseline"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    y = np.arange(len(names))
    ax.barh(y, aucs, height=0.55, color="#4c78a8")
    ax.axvline(b["ROC_AUC"], ls="--", color="#d62728", lw=1.5)
    ax.text(b["ROC_AUC"], len(names) - 0.4, f"baseline {b['ROC_AUC']:.3f}", color="#d62728", va="center", fontsize=9)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("ROC-AUC on the sealed June test month")
    ax.set_title("Model families vs the transparent rule")
    save(fig, "model_auc")


def chart_feature_importance() -> None:
    imp = pd.read_json(RESULTS / "feature_importances.json", typ="series").sort_values(ascending=False).head(10)
    imp.index = [i.replace("num__", "") for i in imp.index]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(imp.index[::-1], imp.values[::-1], color="#4c78a8")
    ax.set_xlabel("Feature importance (random forest)")
    ax.set_title("What moves the decline ranking")
    save(fig, "feature_importance")


def chart_base_rate_by_month() -> None:
    df = pd.read_parquet(RESULTS / "features.parquet", columns=["decision_date", "decline_30d"])
    g = df.groupby("decision_date")["decline_30d"].agg(["count", "mean"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(range(len(g)), g["mean"], "o-", color="#2ca02c")
    for i, (_, r) in enumerate(g.iterrows()):
        ax.annotate(f"{r['mean']:.2f}\n(n={int(r['count']):,})", (i, r["mean"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(range(len(g)), [str(d)[:10] for d in g["decision_date"]])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Decline rate (label share)")
    ax.set_title("Observed decline rate by decision month (base rate)")
    save(fig, "base_rate_by_month")


def chart_reason_codes() -> None:
    q = pd.read_csv(RESULTS / "baseline_queue.csv")
    cols = [c for c in ["aging_visible", "stale_visible", "ctr_fix_candidate", "thin_visible", "page_one_decay_risk", "monitor"] if c in q.columns]
    counts = q[cols].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(counts.index, counts.values, color="#8c564b")
    ax.set_ylabel("Pages flagged (all decisions)")
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Why pages enter the review queue (reason codes)")
    save(fig, "reason_codes")


def main() -> None:
    results = load_results()
    chart_precision_at_k(results)
    chart_model_auc(results)
    chart_feature_importance()
    chart_base_rate_by_month()
    chart_reason_codes()


if __name__ == "__main__":
    main()