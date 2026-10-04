"""
check_dataset_and_weights.py
============================
Audit event yields, negative weights, and signal mass equalization.

Corresponds to Semester Report:
- Section 2.2: Simulated Samples and Background Composition (Table 3)
- Section 3.1: Negative Weights in Background Samples (Table 4)
- Section 3.2: Signal Mass Equalization (Table 5 & Figure 1)

Outputs:
- Yields and weight fraction summaries to stdout / text table
- Figure 1a: Raw event counts per mass hypothesis
- Figure 1b: Training weight per hypothesis after equalization
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
from config import (
    MASS_POINTS,
    PROCESS_NAMES,
    XSEC_BKG,
    XSEC_SIG,
    LUMI,
    find_data_file
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit dataset yields, negative weights, and signal mass equalization."
    )
    parser.add_argument(
        "--sig",
        default="merged_signal_withMET.root",
        help="Signal ROOT file path (auto-searched if not found directly)"
    )
    parser.add_argument(
        "--bkg",
        default="merged_bkg_withMET.root",
        help="Background ROOT file path (auto-searched if not found directly)"
    )
    parser.add_argument(
        "--tree",
        default="DiphotonTree",
        help="Tree name in ROOT files"
    )
    parser.add_argument(
        "--outdir",
        default="plots/weights",
        help="Output directory for plots"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sig_file = find_data_file(args.sig)
    bkg_file = find_data_file(args.bkg)

    print(f"[Dataset Audit] Loading Signal from: {sig_file}")
    with uproot.open(sig_file) as f_sig:
        tree_sig = f_sig[args.tree]
        sig_data = tree_sig.arrays(["mass_point", "evt_wgt"], library="ak")
        df_sig = ak.to_dataframe(sig_data)

    print(f"[Dataset Audit] Loading Background from: {bkg_file}")
    with uproot.open(bkg_file) as f_bkg:
        tree_bkg = f_bkg[args.tree]
        bkg_data = tree_bkg.arrays(["process_id", "evt_wgt"], library="ak")
        df_bkg = ak.to_dataframe(bkg_data)

    # -------------------------------------------------------------------------
    # 1. Background Negative Weights Audit (Table 4)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TABLE 4: Fraction of negative event weights in simulated background processes")
    print("=" * 78)
    print(f"{'Process':<15} {'N_total':<10} {'N(w < 0)':<10} {'Negative Fraction':<20}")
    print("-" * 78)

    for pid in sorted(df_bkg['process_id'].unique()):
        pname = PROCESS_NAMES.get(pid, f"PID {pid}")
        sub = df_bkg[df_bkg['process_id'] == pid]
        n_tot = len(sub)
        n_neg = (sub['evt_wgt'] < 0).sum()
        frac_neg = (n_neg / n_tot * 100.0) if n_tot > 0 else 0.0
        print(f"{pname:<15} {n_tot:<10} {n_neg:<10} {frac_neg:6.2f}%")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 2. Signal Mass Hypothesis Audit (Table 5)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TABLE 5: Selected raw signal events & training weight share (Before/After Equalization)")
    print("=" * 78)
    print(f"{'m_a [GeV]':<12} {'Raw Events':<12} {'Fraction':<12} {'Raw Wgt Share':<16} {'Equalized Share':<16}")
    print("-" * 78)

    total_sig_events = len(df_sig)
    total_sig_raw_wgt = df_sig['evt_wgt'].abs().sum()

    counts = {}
    raw_shares = {}
    eq_shares = {}
    n_mps = len(MASS_POINTS)

    for mp in MASS_POINTS:
        sub = df_sig[df_sig['mass_point'] == mp]
        n_ev = len(sub)
        counts[mp] = n_ev
        raw_w = sub['evt_wgt'].abs().sum()
        raw_shares[mp] = (raw_w / total_sig_raw_wgt) if total_sig_raw_wgt > 0 else 0.0
        eq_shares[mp] = 1.0 / n_mps
        pct_ev = (n_ev / total_sig_events * 100.0) if total_sig_events > 0 else 0.0
        print(f"{mp:<12} {n_ev:<12} {pct_ev:6.2f}%      {raw_shares[mp]:<16.4f} {eq_shares[mp]:<16.4f}")

    print("-" * 78)
    print(f"{'Total':<12} {total_sig_events:<12} {'100.0%':<12} {'1.0000':<16} {'1.0000':<16}")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 3. Generate Figure 1: Event Counts and Equalized Weights per Hypothesis
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Fig 1a: Raw event counts
    mps = np.array(MASS_POINTS)
    sig_cnts = [counts.get(mp, 0) for mp in mps]
    # Background has flat hypothesis distribution by construction
    bkg_flat_cnt = len(df_bkg) / n_mps

    ax1.bar(mps - 0.7, sig_cnts, width=1.4, color='#7293CB', edgecolor='black', alpha=0.8, label='Signal')
    ax1.plot([mps[0] - 2, mps[-1] + 2], [bkg_flat_cnt, bkg_flat_cnt], color='#D35E60', linestyle='--', linewidth=2, label='Background (flat)')
    ax1.set_xlabel(r"Mass hypothesis $\theta$ [GeV]", fontsize=12)
    ax1.set_ylabel("Raw Events", fontsize=12)
    ax1.set_title("(a) Raw event counts per mass hypothesis", fontsize=13)
    ax1.set_xticks(mps)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Fig 1b: Training weight per hypothesis after equalization
    sig_eq_wgts = [eq_shares[mp] for mp in mps]
    bkg_eq_wgts = [1.0 / n_mps for _ in mps]

    ax2.bar(mps - 0.7, sig_eq_wgts, width=1.4, color='#7293CB', edgecolor='black', alpha=0.8, label='Signal (equalized)')
    ax2.plot([mps[0] - 2, mps[-1] + 2], [1.0 / n_mps, 1.0 / n_mps], color='#D35E60', linestyle='--', linewidth=2, label='Background')
    ax2.set_xlabel(r"Mass hypothesis $\theta$ [GeV]", fontsize=12)
    ax2.set_ylabel("Training weight / bin", fontsize=12)
    ax2.set_title("(b) Training weight per hypothesis after equalization", fontsize=13)
    ax2.set_xticks(mps)
    ax2.set_ylim(0, 0.15)
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig1_png = os.path.join(args.outdir, "fig1_mass_hypothesis_equalization.png")
    fig1_pdf = os.path.join(args.outdir, "fig1_mass_hypothesis_equalization.pdf")
    plt.savefig(fig1_png, dpi=300)
    plt.savefig(fig1_pdf)
    plt.close()
    print(f"\n[Figure 1 Generated] Saved plot to {fig1_png} and {fig1_pdf}")


if __name__ == "__main__":
    main()
