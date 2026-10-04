"""
plot_mgg_sculpting_no_dy.py
===========================
Plots the diphoton mass sculpting excluding Drell-Yan processes.

Corresponds to Semester Report:
- Section 6.2: Drell-Yan Spikes and Inference Evaluation Fix
  "1. Separating Drell-Yan fluctuations from the continuum: Initial post-cut
   background plots displayed sharp, isolated peaks that resembled resonant structures.
   By comparing distributions with and without Drell-Yan events, these peaks were
   confirmed to be statistical fluctuations from the very small number of surviving
   Drell-Yan events with large weights. Removing Drell-Yan reveals the smoothly
   sculpted shape of the dominant ttbar continuum."
"""

import os
import sys
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt

# Ensure config can be found from parent directory or current dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MASS_POINTS, PROCESS_NAMES, MGG_RANGE, find_data_file

# Color mapping excluding Drell-Yan (DYto2Mu50 and DYto2E50 omitted)
PROCESS_COLORS_NO_DY = {
    0: "#8B4513",  # TTG1Jets: brown
    1: "#1f77b4",  # TTto2L2Nu: dark blue
    2: "#aec7e8",  # TTtoLNu2Q: light blue
    3: "#2ca02c",  # WGtoLNuG: green
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot diphoton mass sculpting without Drell-Yan background."
    )
    parser.add_argument(
        "--infile",
        default="plots/histograms/histograms_physwgt.root",
        help="Input ROOT histogram file"
    )
    parser.add_argument(
        "--outdir",
        default="plots/mgg_no_dy",
        help="Output directory"
    )
    parser.add_argument(
        "--bdt_cut",
        type=float,
        default=0.7,
        help="BDT cut value label"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    in_file = find_data_file(args.infile)
    print(f"[Plotting Sculpting (No DY)] Reading histograms from: {in_file}")

    with uproot.open(in_file) as f_in:
        ref_mp = 40
        bkg_post = {}
        bins = None

        # Exclude PID 4 (DYto2Mu50) and PID 5 (DYto2E50)
        for pid in [0, 1, 2, 3]:
            k = f"mgg_bkg_pid{pid}_m{ref_mp}_postcut"
            if k in f_in:
                counts, edges = f_in[k].to_numpy()
                bkg_post[pid] = counts
                bins = edges

        sig_post = {}
        if f"mgg_sig_m{ref_mp}_postcut" in f_in:
            sig_post[ref_mp], _ = f_in[f"mgg_sig_m{ref_mp}_postcut"].to_numpy()

        fig, ax = plt.subplots(figsize=(8, 6))
        centers = (bins[:-1] + bins[1:]) / 2.0
        width = bins[1] - bins[0]

        bottom = np.zeros(len(centers))
        for pid in [0, 1, 2, 3]:
            if pid in bkg_post:
                counts = bkg_post[pid]
                pname = PROCESS_NAMES.get(pid, f"PID {pid}")
                color = PROCESS_COLORS_NO_DY.get(pid, "gray")
                ax.bar(centers, counts, width=width, bottom=bottom, color=color, label=pname)
                bottom += counts

        # Overlay signal
        if ref_mp in sig_post:
            ax.step(bins, np.append(sig_post[ref_mp], sig_post[ref_mp][-1]), where="post",
                    color="black", linewidth=1.8, label=f"Signal m_a = {ref_mp} GeV")

        ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]", fontsize=12)
        ax.set_ylabel(f"Events / {width:.1f} GeV", fontsize=12)
        ax.set_title(f"Post-cut $m_{{\gamma\gamma}}$ without Drell–Yan ($m_a = {ref_mp}$ GeV hypothesis)", fontsize=13)
        ax.set_xlim(MGG_RANGE[0], MGG_RANGE[1])
        ax.set_ylim(0, max(bottom.max() * 1.35, 1.0))
        ax.legend(loc="upper right", frameon=True, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        out_path = os.path.join(args.outdir, f"mgg_postcut_without_DY_m{ref_mp}.png")
        plt.savefig(out_path, dpi=300)
        plt.savefig(os.path.join(args.outdir, f"mgg_postcut_without_DY_m{ref_mp}.pdf"))
        plt.close()
        print(f"[Done] Saved Drell-Yan isolated plot to: {out_path}")


if __name__ == "__main__":
    main()
