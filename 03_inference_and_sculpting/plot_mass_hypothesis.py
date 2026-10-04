"""
plot_mass_hypothesis.py
=======================
Plots background and signal event distribution across mass hypotheses before and after cuts.

Corresponds to Semester Report:
- Section 4: First Exploratory pBDT and Selection Effects
  "2. Distortion of background mass hypothesis distribution: Prior to the cut,
   background events were uniformly distributed across mass hypotheses by construction
   (1/11 per mass point). After the BDT > 0.7 cut, the background mass hypothesis
   distribution became distinctly non-flat, retaining more events at higher mass
   hypotheses where the classifier boundary was more permissive."
"""

import os
import sys
import argparse
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt

# Ensure config can be found from parent directory or current dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MASS_POINTS, PROCESS_NAMES, find_data_file

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
        description="Plot mass hypothesis distributions before and after BDT cuts."
    )
    parser.add_argument(
        "--bkg",
        default="merged_bkg_withMET.root",
        help="Background ROOT file"
    )
    parser.add_argument(
        "--sig",
        default="merged_signal_withMET.root",
        help="Signal ROOT file"
    )
    parser.add_argument(
        "--tree",
        default="DiphotonTree",
        help="TTree name in ROOT files"
    )
    parser.add_argument(
        "--outdir",
        default="plots/mass_hypothesis",
        help="Output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sig_file = find_data_file(args.sig)
    bkg_file = find_data_file(args.bkg)

    print(f"[Mass Hypothesis] Reading Signal from {sig_file}")
    with uproot.open(sig_file) as f_sig:
        sig_ak = f_sig[args.tree].arrays(["mass_point", "evt_wgt"], library="ak")
        df_sig = ak.to_dataframe(sig_ak)

    print(f"[Mass Hypothesis] Reading Background from {bkg_file}")
    with uproot.open(bkg_file) as f_bkg:
        bkg_ak = f_bkg[args.tree].arrays(["mass_point", "process_id", "evt_wgt"], library="ak")
        df_bkg = ak.to_dataframe(bkg_ak)

    mps = np.array(MASS_POINTS)
    bins = np.linspace(mps[0] - 2.5, mps[-1] + 2.5, len(mps) + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    width = bins[1] - bins[0]

    # Pre-cut distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(centers))

    for pid in [0, 1, 2, 3, 4, 5]:
        sub = df_bkg[df_bkg["process_id"] == pid]
        counts, _ = np.histogram(sub["mass_point"], bins=bins)
        pname = PROCESS_NAMES.get(pid, f"PID {pid}")
        ax.bar(centers, counts, width=width * 0.8, bottom=bottom, color=PROCESS_COLORS.get(pid, "gray"), label=pname)
        bottom += counts

    # Overlay signal counts
    sig_counts, _ = np.histogram(df_sig["mass_point"], bins=bins)
    ax.step(bins, np.append(sig_counts, sig_counts[-1]), where="post", color="black", linewidth=2, label="Signal")

    ax.set_xlabel(r"Assigned Mass Hypothesis $\theta$ [GeV]", fontsize=12)
    ax.set_ylabel("Raw Event Counts", fontsize=12)
    ax.set_title("Mass Hypothesis Distribution (Before BDT Selection)", fontsize=13)
    ax.set_xticks(mps)
    ax.legend(loc="upper left", frameon=True, fontsize=10, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_pre = os.path.join(args.outdir, "mass_hypothesis_precut.png")
    plt.savefig(out_pre, dpi=300)
    plt.savefig(os.path.join(args.outdir, "mass_hypothesis_precut.pdf"))
    plt.close()
    print(f"[Done] Saved pre-cut mass hypothesis plot to {out_pre}")


if __name__ == "__main__":
    main()
