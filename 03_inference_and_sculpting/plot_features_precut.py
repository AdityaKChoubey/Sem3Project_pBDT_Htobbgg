"""
plot_features_precut.py
=======================
Plots stacked background and signal distributions for input features before BDT selection.

Corresponds to Semester Report:
- Section 5.1: Baseline Configuration and Performance (Table 7)
"""

import os
import sys
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt

# Ensure config can be found from parent directory or current dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    INPUT_VARS,
    PROCESS_NAMES,
    PLOT_SETTINGS,
    find_data_file
)

PROCESS_COLORS = {
    0: "#8B4513",  # TTG1Jets: brown
    1: "#1f77b4",  # TTto2L2Nu: dark blue
    2: "#aec7e8",  # TTtoLNu2Q: light blue
    3: "#2ca02c",  # WGtoLNuG: green
    4: "#ff7f0e",  # DYto2Mu50: orange
    5: "#d62728",  # DYto2E50: red
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot input feature distributions before BDT cut."
    )
    parser.add_argument(
        "--infile",
        default="plots/histograms/histograms_physwgt.root",
        help="Input ROOT histogram file"
    )
    parser.add_argument(
        "--outdir",
        default="plots/features_precut",
        help="Output directory"
    )
    parser.add_argument(
        "--sig_masses",
        default="20,40,60",
        help="Comma-separated signal mass points to overlay"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    in_file = find_data_file(args.infile)
    print(f"[Plotting Features] Reading histograms from: {in_file}")

    sig_mps = [int(m.strip()) for m in args.sig_masses.split(",")]

    with uproot.open(in_file) as f_in:
        for var in INPUT_VARS:
            # Check if histograms exist for this variable
            bkg_hists = {}
            bins = None
            for pid in [0, 1, 2, 3, 4, 5]:
                k = f"{var}_bkg_pid{pid}_precut"
                if k in f_in:
                    counts, edges = f_in[k].to_numpy()
                    bkg_hists[pid] = counts
                    bins = edges

            if not bkg_hists or bins is None:
                continue

            sig_hists = {}
            for mp in sig_mps:
                k = f"{var}_sig_m{mp}_precut"
                if k in f_in:
                    counts, _ = f_in[k].to_numpy()
                    sig_hists[mp] = counts

            centers = (bins[:-1] + bins[1:]) / 2.0
            width = bins[1] - bins[0]

            fig, ax = plt.subplots(figsize=(8, 6))
            bottom = np.zeros(len(centers))

            for pid in [0, 1, 2, 3, 4, 5]:
                if pid in bkg_hists:
                    counts = bkg_hists[pid]
                    pname = PROCESS_NAMES.get(pid, f"PID {pid}")
                    color = PROCESS_COLORS.get(pid, "gray")
                    ax.bar(centers, counts, width=width, bottom=bottom, color=color, label=pname)
                    bottom += counts

            # Overlay signal
            sig_colors = ["black", "darkviolet", "teal"]
            for i, mp in enumerate(sig_mps):
                if mp in sig_hists:
                    c = sig_colors[i % len(sig_colors)]
                    ax.step(bins, np.append(sig_hists[mp], sig_hists[mp][-1]), where="post",
                            color=c, linewidth=1.8, label=f"m_a = {mp} GeV")

            label = PLOT_SETTINGS.get(var, {}).get("label", var)
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel(f"Events / bin", fontsize=12)
            ax.set_title(f"Input Feature: {var} (Pre-cut)", fontsize=13)
            ax.legend(loc="upper right", frameon=True, fontsize=9, ncol=2)
            ax.grid(True, linestyle=":", alpha=0.5)

            if PLOT_SETTINGS.get(var, {}).get("logy", False):
                ax.set_yscale("log")

            plt.tight_layout()
            out_png = os.path.join(args.outdir, f"{var}_before.png")
            plt.savefig(out_png, dpi=300)
            plt.close()
            print(f"  Saved {var} plot to {out_png}")


if __name__ == "__main__":
    main()
