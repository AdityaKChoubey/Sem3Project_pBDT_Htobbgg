"""
plot_feature_importance.py
===========================
Extracts and plots XGBoost feature importance by weight (split frequency) and gain.

Corresponds to Semester Report:
- Section 5.2: Model Diagnostics and Tree Inspection
- Figure 3: Feature importance for Case 1 pBDT (10 trees) by weight (left) and gain (right),
  with mass_point highlighted in red. Five variables show zero importance.
- Section 7.2 / Figure 8a: Feature importance by gain for new engineered feature model.
"""

import os
import sys
import argparse
import pickle
import matplotlib.pyplot as plt

# Ensure config can be found from parent directory or current dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import find_data_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot XGBoost feature importance by weight and gain."
    )
    parser.add_argument(
        "--model",
        default="HtoAATo2b2g_run2.pkl",
        help="Pickled XGBoost model file"
    )
    parser.add_argument(
        "--outdir",
        default="plots/feature_importance",
        help="Output directory"
    )
    parser.add_argument(
        "--prefix",
        default="feature_importance",
        help="Prefix for output plot files"
    )
    return parser.parse_args()


def plot_single_metric(ax, score_dict, metric_name, title, all_features=None):
    if all_features:
        for f in all_features:
            if f not in score_dict:
                score_dict[f] = 0.0

    items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, v in items]
    vals = [v for k, v in items]

    colors = ["#D35E60" if n == "mass_point" else "#4A90E2" for n in names]

    ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="black", height=0.7)
    ax.set_xlabel(f"Importance ({metric_name})", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    model_path = find_data_file(args.model)
    print(f"[Feature Importance] Loading model from: {model_path}")

    with open(model_path, "rb") as f:
        m_dict = pickle.load(f)

    if isinstance(m_dict, dict) and "xgbModel" in m_dict:
        booster = m_dict["xgbModel"]
        features = m_dict.get("features", None)
    else:
        booster = m_dict
        features = None

    score_weight = booster.get_score(importance_type="weight")
    score_gain = booster.get_score(importance_type="gain")

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE BY WEIGHT (Split Frequency):")
    print("=" * 60)
    for k, v in sorted(score_weight.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:25s}: {v}")

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE BY GAIN (Average Loss Reduction):")
    print("=" * 60)
    for k, v in sorted(score_gain.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:25s}: {v:.2f}")
    print("=" * 60)

    # Combined Figure 3 plot (Weight left, Gain right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    plot_single_metric(ax1, score_weight, "weight", "(a) By weight", all_features=features)
    plot_single_metric(ax2, score_gain, "gain", "(b) By gain", all_features=features)

    plt.tight_layout()
    out_comb = os.path.join(args.outdir, f"{args.prefix}_weight_gain.png")
    plt.savefig(out_comb, dpi=300)
    plt.savefig(os.path.join(args.outdir, f"{args.prefix}_weight_gain.pdf"))
    plt.close()
    print(f"[Done] Saved combined importance plot to: {out_comb}")


if __name__ == "__main__":
    main()
