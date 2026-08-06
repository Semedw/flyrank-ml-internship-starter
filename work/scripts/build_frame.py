"""Phase 1 — build the capstone analysis frame in ONE aggregated pass.

One row per (content item x decision moment). Features come from the 30 days
before the decision moment, the label from the 30 days after it. The windows
never overlap by construction (feature window [D-30, D), label window [D, D+30)).

June 2026 (month=2026-06) is the SEALED test-label month: it is read only as the
label window of the D=2026-05-31 decision, and only for that decision's query.
The `fact_content_daily_performance_sample` table (also June) is never read.

Known release artifact: the final month (2026-06) contains 6,390 exact-duplicate
daily rows (verified identical on all columns). They are removed with DISTINCT
before aggregation, so label sums are not double-counted.

The frame is cached to work/outputs/capstone/frame.parquet (gitignored).
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd

from hf_token import resolve_hf_token

REL = "hf://datasets/FlyRank/internship-warehouse"
DIM_CONTENT = f"read_parquet('{REL}/dim_content.parquet')"
DIM_CLIENTS = f"read_parquet('{REL}/dim_clients.parquet')"

DECISION_DATES = ["2026-02-28", "2026-03-31", "2026-04-30", "2026-05-31"]

MIN_FW_IMPRESSIONS = 50

CACHE_DIR = str(Path(__file__).resolve().parents[1] / "work" / "outputs" / "capstone")
CACHE_PATH = f"{CACHE_DIR}/frame.parquet"
CONTRACT_PATH = f"{CACHE_DIR}/contract.json"

REQUIRED_FACT = ["report_date", "client_hash_id", "content_hash_id", "gsc_impressions", "gsc_clicks", "gsc_avg_position", "ga4_data_available"]
OPTIONAL_FACT = ["ga4_sessions", "sessions_ai", "scroll_events"]  # GA4-derived; gated per-row by the flag
REQUIRED_DIM = ["content_hash_id"]
REQUIRED_CLIENTS = ["client_hash_id", "gsc_data_start"]

# safe content-metadata features (verified against the release schema)
DIM_NUMERIC = [
    "word_count", "char_count", "keyword_token_count", "url_char_count",
    "search_volume", "competition", "cpc", "backlinks", "category_count",
]
DIM_CATEGORICAL = ["content_type", "main_intent", "competition_level", "provider_used", "model_used"]
DIM_DATES = ["content_created_date", "content_updated_date", "last_optimized_date"]


def probe(con: duckdb.DuckDBPyConnection) -> None:
    for name, src in [
        ("fact_daily months 2026-01..06", f"read_parquet('{REL}/fact_content_daily_performance/month=2026-*/data_0.parquet')"),
        ("dim_content", DIM_CONTENT),
        ("dim_clients", DIM_CLIENTS),
    ]:
        n = con.sql(f"SELECT COUNT(*) FROM {src}").fetchone()[0]
        cols = [_[0] for _ in con.sql(f"DESCRIBE SELECT * FROM {src}").fetchall()]
        print(f"== {name}: {n:,} rows")
        print("   cols:", ", ".join(cols))
    print()


def _months_for(decision: str) -> list[str]:
    """Partitions a decision touches: first feature day .. last label day."""
    d = pd.Timestamp(decision)
    parts = {(d - timedelta(days=29)).strftime("%Y-%m"), d.strftime("%Y-%m"), (d + timedelta(days=29)).strftime("%Y-%m")}
    return sorted(parts)


def _fact_source(months: list[str]) -> str:
    paths = ", ".join(f"'{REL}/fact_content_daily_performance/month={m}/data_0.parquet'" for m in months)
    return f"read_parquet([{paths}])"


def _feature_sql(decision: str, eng_cols: list[str], flag_cols: list[str]) -> str:
    d = f"DATE '{decision}'"
    win = f"report_date >= {d} - INTERVAL 30 DAY AND report_date < {d}"
    gated = f" AND f.{flag_cols[0]} IS TRUE" if flag_cols else ""
    rows = [
        f"SUM(CASE WHEN {win} THEN f.gsc_impressions ELSE 0 END) AS imp_fw",
        f"SUM(CASE WHEN {win} THEN f.gsc_clicks ELSE 0 END) AS clk_fw",
        f"AVG(CASE WHEN {win} AND f.gsc_avg_position > 0 THEN f.gsc_avg_position END) AS pos_fw",
        f"COUNT(DISTINCT CASE WHEN {win} AND f.gsc_impressions > 0 THEN report_date END) AS days_imp_fw",
        f"COUNT(DISTINCT CASE WHEN {win} AND f.gsc_clicks > 0 THEN report_date END) AS days_clk_fw",
        f"COUNT(CASE WHEN {win} AND f.gsc_avg_position > 0 THEN 1 END) AS days_pos_fw",
    ]
    for c in eng_cols:
        rows.append(f"SUM(CASE WHEN {win}{gated} THEN f.{c} ELSE 0 END) AS {c}_fw")
    if flag_cols:
        rows.append(
            f"COUNT(CASE WHEN {win} AND f.{flag_cols[0]} IS TRUE THEN 1 END) AS days_ga4_fw"
        )
    return ",\n".join(rows)


def _label_sql(decision: str, eng_cols: list[str], flag_cols: list[str]) -> str:
    d = f"DATE '{decision}'"
    win = f"report_date >= {d} AND report_date < {d} + INTERVAL 30 DAY"
    gated = f" AND f.{flag_cols[0]} IS TRUE" if flag_cols else ""
    rows = [
        f"SUM(CASE WHEN {win} THEN f.gsc_impressions ELSE 0 END) AS imp_lw",
        f"SUM(CASE WHEN {win} THEN f.gsc_clicks ELSE 0 END) AS clk_lw",
        f"AVG(CASE WHEN {win} AND f.gsc_avg_position > 0 THEN f.gsc_avg_position END) AS pos_lw",
    ]
    for c in eng_cols:
        rows.append(f"SUM(CASE WHEN {win}{gated} THEN f.{c} ELSE 0 END) AS {c}_lw")
    return ",\n".join(rows)


def _decision_query(con, decision: str, eng_cols: list[str], flag_cols: list[str]) -> pd.DataFrame:
    months = _months_for(decision)
    src = _fact_source(months)
    # June (sealed month) carries exact duplicate rows; remove them before label aggregation
    if "2026-06" in months:
        src = f"(SELECT DISTINCT * FROM {src})"
    feature_sql = _feature_sql(decision, eng_cols, flag_cols)
    label_sql = _label_sql(decision, eng_cols, flag_cols)

    num_sel = ",\n    ".join(f"ANY_VALUE(dc.{c}) AS {c}" for c in DIM_NUMERIC)
    cat_sel = ",\n    " + ",\n    ".join(f"ANY_VALUE(dc.{c}) AS {c}" for c in DIM_CATEGORICAL)
    dim_sel = num_sel + cat_sel + ",\n    " + ",\n    ".join(
        f"DATE_DIFF('day', ANY_VALUE(dc.{c}), DATE '{decision}') AS days_since_{c.split('_date')[0]}" for c in DIM_DATES
    )

    q = f"""
SELECT
    DATE '{decision}' AS decision_date,
    f.client_hash_id,
    f.content_hash_id,
{feature_sql},
{label_sql},
{dim_sel}
FROM {src} f
JOIN {DIM_CLIENTS} cl USING(client_hash_id)
LEFT JOIN {DIM_CONTENT} dc USING(content_hash_id)
WHERE DATE '{decision}' >= CAST(cl.gsc_data_start AS DATE)
  AND (dc.is_deleted IS NULL OR dc.is_deleted IS NOT TRUE)
GROUP BY f.client_hash_id, f.content_hash_id
HAVING SUM(CASE WHEN report_date >= DATE '{decision}' - INTERVAL 30 DAY AND report_date < DATE '{decision}' THEN f.gsc_impressions ELSE 0 END) >= {MIN_FW_IMPRESSIONS}
"""
    return con.sql(q).df()


def build(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    fact_cols = [_[0] for _ in con.sql(f"DESCRIBE SELECT * FROM {_fact_source(['2026-03'])}").fetchall()]
    dim_cols_raw = [_[0] for _ in con.sql(f"DESCRIBE SELECT * FROM {DIM_CONTENT}").fetchall()]
    client_cols = [_[0] for _ in con.sql(f"DESCRIBE SELECT * FROM {DIM_CLIENTS}").fetchall()]

    for req, got, where in [
        (REQUIRED_FACT, fact_cols, "fact_daily"),
        (REQUIRED_DIM + DIM_NUMERIC + DIM_CATEGORICAL + DIM_DATES + ["is_deleted"], dim_cols_raw, "dim_content"),
        (REQUIRED_CLIENTS, client_cols, "dim_clients"),
    ]:
        miss = [c for c in req if c not in got]
        if miss:
            raise SystemExit(f"Missing required columns in {where}: {miss}")

    eng_cols = [c for c in OPTIONAL_FACT if c in fact_cols]
    flag_cols = [c for c in ["ga4_data_available"] if c in fact_cols]

    frames = []
    for decision_date in DECISION_DATES:
        print(f"-> building {decision_date} ...", flush=True)
        frames.append(_decision_query(con, decision_date, eng_cols, flag_cols))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["decision_date", "client_hash_id", "content_hash_id"])
    return df


def write_contract(df: pd.DataFrame) -> None:
    contract = {
        "release": "flyrank_pseudonymized_warehouse_release_v20260703",
        "tables": ["fact_content_daily_performance", "dim_content", "dim_clients"],
        "decision_dates": DECISION_DATES,
        "feature_window": "[D-30, D)",
        "label_window": "[D, D+30)",
        "min_fw_impressions": MIN_FW_IMPRESSIONS,
        "excluded": [
            "trend_direction / trend_pct (label-derived)",
            "fact_content_query_90d (unalignable 90d window)",
            "product decision flags (not shipped)",
            "identity hashes as features",
            "fact_content_daily_performance_sample (June = test month)",
            "rows before client gsc_data_start",
            "deleted content (is_deleted)",
            "GA4 columns outside ga4_data_available rows (zero-filled)",
        ],
        "release_artifact": "2026-06 contains 6,390 exact duplicate daily rows; removed via DISTINCT before aggregation",
        "rows": int(len(df)),
        "clients": int(df["client_hash_id"].nunique()),
        "contents": int(df["content_hash_id"].nunique()),
        "decision_counts": df.groupby("decision_date")["content_hash_id"].count().astype(int).to_dict(),
    }
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(CONTRACT_PATH).write_text(json.dumps(contract, indent=2) + "\n")


def main() -> None:
    token = resolve_hf_token()
    con = duckdb.connect()
    try:
        con.execute("PRAGMA disable_progress_bar")
    except Exception:
        pass
    con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{token}')")
    probe(con)
    df = build(con)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    write_contract(df)
    print(f"\ncached {len(df):,} rows x {df.shape[1]} cols -> {CACHE_PATH}")
    print("columns:", ", ".join(df.columns))
    print(df.groupby("decision_date").size().to_string())


if __name__ == "__main__":
    main()