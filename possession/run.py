"""Make the submission file: match data in, one frame,label CSV out.

    .venv/bin/python -m possession.run                       # writes evaluation/results/ours.csv
    .venv/bin/python -m possession.run --out GunnersFC.csv   # writes GunnersFC.csv

Nothing is random, so the same data always gives exactly the same file.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import write_submission  # noqa: E402
from possession import deadball, features, tracker  # noqa: E402

# Params
DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "ours.csv")


def build(refresh=False):
    """Run all three steps. Returns the answer codes for every frame, and the table."""
    table = features.load(refresh=refresh)     # step 1: measurements
    stoppages = deadball.detect(table)         # step 2: stoppages
    codes = tracker.label(table, stoppages)    # step 3: possession
    return codes, table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to write the CSV")
    parser.add_argument("--refresh", action="store_true",
                        help="read the tracking file again instead of using the saved copy")
    args = parser.parse_args()

    codes, table = build(args.refresh)
    letters = tracker.LABELS[codes]
    path = write_submission(args.out, letters)
    tracker.describe(codes, table)
    print("wrote {}".format(os.path.relpath(path, ROOT)))
