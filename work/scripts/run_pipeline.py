from __future__ import annotations

import argparse
import sys
from pathlib import Path

import build_baseline  # noqa: F401  (imports metrics + sets sys.path)
import build_features
import build_frame
import make_charts
import render_paper
import sensitivity
import train_model

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-scan", action="store_true", help="skip build_frame if frame.parquet exists")
    args = parser.parse_args()

    frame = ROOT / "work" / "outputs" / "capstone" / "frame.parquet"
    if not args.skip_scan or not frame.exists():
        print("==> Phase 1: scan warehouse and build the frame (heavy, ~10-20 min)", flush=True)
        build_frame.main()
    else:
        print("==> Phase 1: SKIPPED (cached frame found)", flush=True)

    print("==> Phase 2: label + features + leakage asserts", flush=True)
    build_features.main()

    print("==> Phase 3a: rule baseline", flush=True)
    build_baseline.main()

    print("==> Phase 4: sealed-test models (LR/DT/RF)", flush=True)
    train_model.main()

    print("==> Phase 3b: label-threshold sensitivity (cached frame only)", flush=True)
    sensitivity.main()

    print("==> Phase 5: charts", flush=True)
    make_charts.main()

    print("==> Phase 6: render paper -> docs/index.html", flush=True)
    render_paper.main()

    print("DONE. Artifacts in work/outputs/capstone/, work/figures/, docs/")


if __name__ == "__main__":
    main()