"""
plot_correlations.py
====================
Computes Pearson correlation matrices for signal and background events.

Corresponds to Semester Report:
- Section 6.1: Mechanism of Mass Sculpting
  "The kinematic mechanism behind this sculpting was traced through a correlation
   analysis. Pearson correlation coefficients between the input features and m_gg
   in the background sample were computed:
   - pT(gamma2)/m_gg: r = -0.47
   - delphi_gg:       r = +0.35
   - pT(gamma1)/m_gg: r = -0.31
   - Njets:           r = +0.12"
- Figure 6: Pearson correlation matrices for background (left) and signal (right).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import awkward as ak
import uproot
import matplotlib.pyplot as plt

# Ensure config can be found from parent directory or current dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import INPUT_VARS, find_data_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and plot Pearson correlation matrices."
    )
    parser.add_argument(
        "--sig",
        default="merged_signal_withMET.root",
        help="Signal ROOT file"
    )
    parser.add_argument(
        "--bkg",
        default="merged_bkg_withMET.root",
        help="Background ROOT file"
    )
    parser.add_argument(
        "--tree",
        default="DiphotonTree",
        help="Tree name in ROOT files"
    )
    parser.add_argument(
        "--outdir",
        default="plots/correlations",
        help="Output directory"
    )
    return parser.parse_args()


def plot_corr_matrix(df, vars_to_corr, title, outpath):
    corr = df[vars_to_corr].corr(method="pearson").values
    n = len(vars_to_corr)

    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(vars_to_corr, rotation=90, fontsize=9)
    ax.set_yticklabels(vars_to_corr, fontsize=9)
    ax.set_title(title, fontsize=14, pad=15)

    # Annotate numeric correlation values
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(outpath + ".png", dpi=300)
    plt.savefig(outpath + ".pdf")
    plt.close()
    print(f"[Correlation Matrix] Saved {outpath}.png")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sig_file = find_data_file(args.sig)
    bkg_file = find_data_file(args.bkg)

    vars_needed = list(INPUT_VARS)
    if "CMS_hgg_mass" not in vars_needed:
        vars_needed.append("CMS_hgg_mass")

    print(f"[Correlations] Loading Background from: {bkg_file}")
    with uproot.open(bkg_file) as f_bkg:
        bkg_ak = f_bkg[args.tree].arrays(vars_needed, library="ak")
        df_bkg = ak.to_dataframe(bkg_ak)

    print(f"[Correlations] Loading Signal from: {sig_file}")
    with uproot.open(sig_file) as f_sig:
        sig_ak = f_sig[args.tree].arrays(vars_needed, library="ak")
        df_sig = ak.to_dataframe(sig_ak)

    # 1. Print Background correlations with m_gg (Section 6.1)
    bkg_corr = df_bkg[vars_needed].corr(method="pearson")["CMS_hgg_mass"].sort_values(ascending=False)
    print("\n" + "=" * 60)
    print("SECTION 6.1: Pearson Correlation with m_gg in Background")
    print("=" * 60)
    for var, val in bkg_corr.items():
        if var != "CMS_hgg_mass":
            print(f"  {var:25s}: r = {val:+0.3f}")
    print("=" * 60)

    # 2. Plot matrices (Figure 6)
    out_bkg = os.path.join(args.outdir, "corr_matrix_background")
    out_sig = os.path.join(args.outdir, "corr_matrix_signal")

    plot_corr_matrix(df_bkg, vars_needed, "Pearson Correlation Matrix — Background", out_bkg)
    plot_corr_matrix(df_sig, vars_needed, "Pearson Correlation Matrix — Signal", out_sig)


if __name__ == "__main__":
    main()
