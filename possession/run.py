"""Regenerate the submission: the provided data in, one frame,label CSV out.

    .venv/bin/python -m possession.run                           # -> evaluation/results/ours.csv
    .venv/bin/python -m possession.run --out some/where/team.csv

Deterministic end to end -- no randomness, no training at run time -- so the same data always gives
the same file, byte for byte, which is what the rules require of a submission.
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
    """Label codes for every frame of the tactical video."""
    table = features.load(refresh=refresh)
    return tracker.label(table, deadball.detect(table)), table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to write the CSV")
    parser.add_argument("--refresh", action="store_true",
                        help="rebuild the cached tracking and measurements first")
    args = parser.parse_args()

    codes, table = build(args.refresh)
    path = write_submission(args.out, tracker.LABELS[codes])
    tracker.describe(codes, table)
    print("wrote {}".format(os.path.relpath(path, ROOT)))
