"""
calculate_ams_scan.py
=====================
Computes Approximate Median Significance (AMS) across BDT cut thresholds.

Corresponds to Semester Report:
- Section 8.1: Expected Significance and Equalization Impact (Equation 8)
  AMS = sqrt( 2 * [ (S + B + b_r) * ln(1 + S / (B + b_r)) - S ] )
- Table 11: Maximum AMS and optimal BDT score cut for each mass point
- Figure 9b: Maximum AMS vs. m_a (equalized training)
"""

import os
import sys
import argparse
import pickle
import numpy as np
import awkward as ak
import uproot
import xgboost as xgb
import matplotlib.pyplot as plt

# Ensure config is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    INPUT_VARS,
    MASS_POINTS,
    LUMI,
    get_physics_weights_signal,
    get_physics_weights_background,
    find_data_file
)


def compute_ams(s, b, b_r=0.001):
    """Computes AMS with regularizing offset b_r (Equation 8 in report)."""
    if s <= 0:
        return 0.0
    val = 2.0 * ((s + b + b_r) * np.log(1.0 + s / (b + b_r)) - s)
    return np.sqrt(max(val, 0.0))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate AMS scan vs BDT cut threshold for each mass hypothesis."
    )
    parser.add_argument("--model", default="HtoAATo2b2g_run2.pkl", help="Pickled model file")
    parser.add_argument("--bkg", default="merged_bkg_withMET.root", help="Background ROOT file")
    parser.add_argument("--sig", default="merged_signal_withMET.root", help="Signal ROOT file")
    parser.add_argument("--tree", default="DiphotonTree", help="TTree name")
    parser.add_argument("--outdir", default="plots/ams", help="Output directory")
    parser.add_argument("--n_cuts", type=int, default=1001, help="Number of cut thresholds to scan")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    model_path = find_data_file(args.model)
    sig_path = find_data_file(args.sig)
    bkg_path = find_data_file(args.bkg)

    print(f"[AMS Calculation] Loading model: {model_path}")
    with open(model_path, "rb") as f:
        m_dict = pickle.load(f)
    model = m_dict["xgbModel"] if isinstance(m_dict, dict) and "xgbModel" in m_dict else m_dict
    features = m_dict.get("features", list(INPUT_VARS)) if isinstance(m_dict, dict) else list(INPUT_VARS)

    branches = list(set(features + ["process_id", "evt_wgt", "mass_point"]))

    print(f"[AMS Calculation] Loading datasets...")
    with uproot.open(bkg_path) as fb, uproot.open(sig_path) as fs:
        df_bkg = ak.to_dataframe(fb[args.tree].arrays(branches, library="ak"))
        df_sig = ak.to_dataframe(fs[args.tree].arrays(branches, library="ak"))

    df_sig["phys_wgt"] = get_physics_weights_signal(df_sig, lumi=LUMI)
    df_bkg["phys_wgt"] = get_physics_weights_background(df_bkg, lumi=LUMI)

    thresholds = np.linspace(0.0, 1.0, args.n_cuts)
    results = {}

    print("\n" + "=" * 78)
    print("TABLE 11: Maximum AMS and optimal BDT score cut for each mass point")
    print("=" * 78)
    print(f"{'m_a [GeV]':<12} {'Max AMS':<14} {'Opt. Cut':<14} {'Sig Yield (S)':<16} {'Bkg Yield (B)':<16}")
    print("-" * 78)

    max_ams_list = []

    for mp in MASS_POINTS:
        sub_s = df_sig[df_sig["mass_point"] == mp]
        dmat_s = xgb.DMatrix(sub_s[features].values, feature_names=features)
        pred_s = model.predict(dmat_s)
        w_s = sub_s["phys_wgt"].values

        df_bkg_mp = df_bkg.copy()
        df_bkg_mp["mass_point"] = mp
        dmat_b = xgb.DMatrix(df_bkg_mp[features].values, feature_names=features)
        pred_b = model.predict(dmat_b)
        w_b = df_bkg["phys_wgt"].values

        best_ams = 0.0
        best_cut = 0.0
        best_s = 0.0
        best_b = 0.0

        for cut in thresholds:
            s_pass = w_s[pred_s > cut].sum()
            b_pass = w_b[pred_b > cut].sum()
            ams = compute_ams(s_pass, b_pass)
            if ams > best_ams:
                best_ams = ams
                best_cut = cut
                best_s = s_pass
                best_b = b_pass

        results[mp] = (best_ams, best_cut, best_s, best_b)
        max_ams_list.append(best_ams)
        print(f"{mp:<12} {best_ams:<14.4f} {best_cut:<14.3f} {best_s:<16.2f} {best_b:<16.2f}")

    print("-" * 78)
    combined_ams = np.sqrt(sum(a**2 for a in max_ams_list))
    print(f"{'Combined':<12} {combined_ams:<14.4f}")
    print("=" * 78)

    # Plot Figure 9b: Maximum AMS vs m_a
    fig, ax = plt.subplots(figsize=(8, 6))
    mps = np.array(MASS_POINTS)
    ams_vals = np.array(max_ams_list)

    ax.plot(mps, ams_vals, marker="o", color="black", linewidth=1.8, markersize=4, label="Equalized BDT")
    ax.set_xlabel(r"$m_a$ [GeV]", fontsize=12)
    ax.set_ylabel("Max AMS", fontsize=12)
    ax.set_title("Maximum AMS vs mass point", fontsize=13)
    ax.set_xticks(mps)
    ax.set_ylim(0, max(ams_vals) * 1.25)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_ams = os.path.join(args.outdir, "ams_vs_mass.png")
    plt.savefig(out_ams, dpi=300)
    plt.savefig(os.path.join(args.outdir, "ams_vs_mass.pdf"))
    plt.close()
    print(f"\n[Figure 9b Generated] Saved {out_ams}")


if __name__ == "__main__":
    main()
