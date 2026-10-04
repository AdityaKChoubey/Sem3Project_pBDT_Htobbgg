"""
inspect_trees.py
================
Inspects decision trees and verifies mass_point split thresholds.

Corresponds to Semester Report:
- Section 5.2: Model Diagnostics and Tree Inspection (Figure 4)
- Section 5.3: More Boosting Rounds & Capacity Scaling (Table 8)

Computes the split distribution across mass hypotheses to reproduce Table 8:
- 10 trees: no thresholds below 25 GeV (mass starvation)
- 40 trees: thresholds appear from 15 to 60 GeV
"""

import os
import sys
import argparse
import pickle
import json
import re
import numpy as np

# Ensure config can be found from parent directory or current dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MASS_POINTS, find_data_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect XGBoost decision tree structures and mass_point splits."
    )
    parser.add_argument(
        "--model",
        default="HtoAATo2b2g_run2.pkl",
        help="Pickled model file path"
    )
    parser.add_argument(
        "--outdir",
        default="plots/trees",
        help="Output directory for text tree dumps"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    model_path = find_data_file(args.model)
    print(f"[Tree Inspection] Loading model from: {model_path}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    if isinstance(model_data, dict) and "xgbModel" in model_data:
        booster = model_data["xgbModel"]
    else:
        booster = model_data

    # Dump trees to text
    dump = booster.get_dump(dump_format="text")
    total_trees = len(dump)
    print(f"[Model Architecture] Total trees in booster: {total_trees}")

    dump_file = os.path.join(args.outdir, "trees_dump.txt")
    with open(dump_file, "w", encoding="utf-8") as f_dump:
        for idx, tree in enumerate(dump):
            f_dump.write(f"=== Tree {idx} ===\n{tree}\n")
    print(f"[Dump Written] Tree text dump saved to {dump_file}")

    # -------------------------------------------------------------------------
    # Parse splits on mass_point (Table 8)
    # -------------------------------------------------------------------------
    mass_splits = []
    tree_mass_info = {}

    for idx, tree in enumerate(dump):
        splits_in_tree = []
        for line in tree.split("\n"):
            line = line.strip()
            if "mass_point" in line:
                # Format: [mass_point<25] or [mass_point<35]
                m = re.search(r"mass_point<([\d\.]+)", line)
                if m:
                    thresh = float(m.group(1))
                    splits_in_tree.append(thresh)
                    mass_splits.append(thresh)
        tree_mass_info[idx] = splits_in_tree

    print("\n" + "=" * 70)
    print("SECTION 5.2 INSPECTION: Decision splits on mass_point per tree")
    print("=" * 70)
    for idx, splits in tree_mass_info.items():
        if splits:
            print(f"Tree {idx:2d}: mass_point splits at thresholds {splits}")
        else:
            print(f"Tree {idx:2d}: NO splits on mass_point")

    # Reproduce Table 8
    print("\n" + "=" * 78)
    print("TABLE 8: Number of splits on mass_point assigned to nearest mass-point value")
    print("=" * 78)
    header = f"{'Model':<12}" + "".join([f"{mp:<6}" for mp in MASS_POINTS])
    print(header)
    print("-" * 78)

    counts = {mp: 0 for mp in MASS_POINTS}
    for thresh in mass_splits:
        # Find nearest mass point in MASS_POINTS
        nearest = min(MASS_POINTS, key=lambda x: abs(x - thresh))
        counts[nearest] += 1

    row = f"{total_trees} Trees".ljust(12)
    for mp in MASS_POINTS:
        c = counts[mp]
        row += f"{c:<6}" if c > 0 else f"{'—':<6}"
    print(row)
    print("=" * 78)


if __name__ == "__main__":
    main()
