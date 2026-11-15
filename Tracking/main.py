"""Master Entry-Point Pipeline (`Tracking/main.py`).

Runs the complete Football Possession Detection pipeline end-to-end, writes the
final competition submission CSV to `evaluation/results/tracking_final.csv`
(so the official evaluator automatically scores it), validates the CSV format
via `evaluation/validate_submission.py`, and prints the official leaderboard via
`evaluation/run_evaluation.py`.

Usage:
    # Standard run (writes evaluation/results/tracking_final.csv + validates + evaluates):
    python Tracking/main.py

    # Also export a copy named after your team for final submission upload:
    python Tracking/main.py --out MyTeamName.csv
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)

from games import GAME  # noqa: E402
from baseline import write_submission  # noqa: E402
from possession import features, tracker  # noqa: E402
from stage1_atd_tracking import (  # noqa: E402
    FULL_LABELS,
    detect_enhanced_stoppages,
    label_enhanced,
)
from stage2_smart_tagging import load_smart_tags  # noqa: E402

DEFAULT_EVAL_CSV = os.path.join(ROOT, "evaluation", "results", "tracking_final.csv")


def run_pipeline(out_csv_path: str = DEFAULT_EVAL_CSV, extra_out: str | None = None) -> str:
    print("=" * 80)
    print("FOOTBALL POSSESSION PIPELINE — MASTER RUN (`Tracking/main.py`)")
    print("=" * 80)

    # 1. Load ATD Tracking Physics & Smart Tagging Metadata
    print("\n[1/4] Loading ATD Tracking Physics & Smart Tagging Metadata...")
    table = features.load(GAME)
    smart_events = load_smart_tags(GAME)
    set_pieces = [e for e in smart_events if (e.get("code") or "").strip().upper() == "SET PIECES"]
    print(
        f"      -> Loaded {len(table['half']):,} tracking frames (25 fps) | "
        f"{len(smart_events)} Smart Tagging events ({len(set_pieces)} set pieces)"
    )

    # 2. Run Stage 1 Enhanced Hybrid Possession Engine
    print("\n[2/4] Running Stage 1 Enhanced Hybrid Possession Engine (`stage1_atd_tracking.py`)...")
    stoppages = detect_enhanced_stoppages(table, GAME)
    codes = label_enhanced(table, stoppages)

    # 3. Write Submission CSV
    os.makedirs(os.path.dirname(os.path.abspath(out_csv_path)), exist_ok=True)
    write_submission(out_csv_path, FULL_LABELS[codes])
    print("\n[3/4] Submission Summary & Output:")
    tracker.describe(codes, table)
    rel_eval_path = os.path.relpath(out_csv_path, ROOT)
    print(f"      -> Saved evaluation CSV: {rel_eval_path}")

    if extra_out:
        extra_abs = os.path.abspath(extra_out)
        if os.path.dirname(extra_abs):
            os.makedirs(os.path.dirname(extra_abs), exist_ok=True)
        shutil.copyfile(out_csv_path, extra_abs)
        print(f"      -> Saved team submission copy: {extra_abs}")

    # 4. Validate & Evaluate
    print("\n[4/4] Validating Submission Format & Running Official Scoreboard...")
    validate_script = os.path.join(ROOT, "evaluation", "validate_submission.py")
    subprocess.run([sys.executable, validate_script, out_csv_path], check=True)

    eval_script = os.path.join(ROOT, "evaluation", "run_evaluation.py")
    subprocess.run([sys.executable, eval_script, "--no-detail"], check=False)

    return out_csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete Football Possession pipeline and generate the submission CSV."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional extra file path to save a copy of the final CSV (e.g. TeamName.csv).",
    )
    args = parser.parse_args()
    run_pipeline(DEFAULT_EVAL_CSV, extra_out=args.out)


if __name__ == "__main__":
    main()
