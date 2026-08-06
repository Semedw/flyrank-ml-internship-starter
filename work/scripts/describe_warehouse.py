"""Phase 0 — verify warehouse access + dump the exact table schemas.

Run once (needs the HF READ token): prints row counts, date spans, grain
probes, and column lists for the tables the capstone contract uses. The
schema print is the source of truth for column names in the frame builder.
"""

import duckdb

from hf_token import resolve_hf_token

REL = "hf://datasets/FlyRank/internship-warehouse"
TABLES = {
    "dim_clients": f"read_parquet('{REL}/dim_clients.parquet')",
    "dim_content": f"read_parquet('{REL}/dim_content.parquet')",
    "fact_daily": f"read_parquet('{REL}/fact_content_daily_performance/**/*.parquet')",
    "fact_daily_sample": f"read_parquet('{REL}/fact_content_daily_performance_sample.parquet')",
    "fact_query_90d": f"read_parquet('{REL}/fact_content_query_90d.parquet')",
}


def main() -> None:
    token = resolve_hf_token()
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{token}')")

    for name, src in TABLES.items():
        n = con.sql(f"SELECT COUNT(*) FROM {src}").fetchone()[0]
        print(f"== {name:18} {n:>12,} rows")
        cols = [c[0] for c in con.sql(f"DESCRIBE SELECT * FROM {src}").fetchall()]
        print("   cols:", ", ".join(cols))
        print()

    print("== fact_daily date span (full table)")
    row = con.sql(
        f"SELECT COUNT(*) AS rows, MIN(report_date) AS min_d, MAX(report_date) AS max_d, "
        f"COUNT(DISTINCT client_hash_id) AS clients, COUNT(DISTINCT content_hash_id) AS contents "
        f"FROM {TABLES['fact_daily']}"
    ).fetchone()
    print("   ", row)

    print("\n== grain probe (daily, grouped by grain, duplicates?) ==")
    q = (
        f"SELECT report_date, client_hash_id, content_hash_id, COUNT(*) c "
        f"FROM {TABLES['fact_daily']} GROUP BY 1,2,3 HAVING c > 1 LIMIT 5"
    )
    print(con.sql(q).df().to_string())


if __name__ == "__main__":
    main()