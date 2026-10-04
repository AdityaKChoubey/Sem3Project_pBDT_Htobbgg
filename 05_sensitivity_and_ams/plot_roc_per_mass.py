"""
plot_roc_per_mass.py
====================
Generates per-mass hypothesis ROC curves with hypothesis-dependent background evaluation.

Corresponds to Semester Report:
- Section 5.1: Baseline Configuration (Figure 2c: Case 1 ROC curves at m_a = 20, 40, 60 GeV)
- Section 8.1: Expected Significance and Equalization Impact
- Figure 9a: Per-mass ROC curves with hypothesis-dependent background evaluation
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
from sklearn import metrics

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-mass ROC curves with hypothesis-dependent background scoring."
    )
    parser.add_argument("--model", default="HtoAATo2b2g_run2.pkl", help="Pickled model file")
    parser.add_argument("--bkg", default="merged_bkg_withMET.root", help="Background ROOT file")
    parser.add_argument("--sig", default="merged_signal_withMET.root", help="Signal ROOT file")
    parser.add_argument("--tree", default="DiphotonTree", help="TTree name")
    parser.add_argument("--outdir", default="plots/roc", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    model_path = find_data_file(args.model)
    sig_path = find_data_file(args.sig)
    bkg_path = find_data_file(args.bkg)

    print(f"[ROC Evaluation] Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        m_dict = pickle.load(f)
    model = m_dict["xgbModel"] if isinstance(m_dict, dict) and "xgbModel" in m_dict else m_dict
    features = m_dict.get("features", list(INPUT_VARS)) if isinstance(m_dict, dict) else list(INPUT_VARS)

    branches = list(set(features + ["process_id", "evt_wgt", "mass_point"]))

    print(f"[ROC Evaluation] Loading background and signal...")
    with uproot.open(bkg_path) as fb, uproot.open(sig_path) as fs:
        df_bkg = ak.to_dataframe(fb[args.tree].arrays(branches, library="ak"))
        df_sig = ak.to_dataframe(fs[args.tree].arrays(branches, library="ak"))

    df_sig["phys_wgt"] = get_physics_weights_signal(df_sig, lumi=LUMI)
    df_bkg["phys_wgt"] = get_physics_weights_background(df_bkg, lumi=LUMI)

    # Global canvas (Figure 9a)
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("turbo", len(MASS_POINTS))

    auc_per_mass = {}

    for i, mp in enumerate(MASS_POINTS):
        # Signal at true mass hypothesis
        sub_s = df_sig[df_sig["mass_point"] == mp]
        dmat_s = xgb.DMatrix(sub_s[features].values, feature_names=features)
        pred_s = model.predict(dmat_s)
        w_s = sub_s["phys_wgt"].values

        # Background evaluated with mass_point overwritten to mp
        df_bkg_mp = df_bkg.copy()
        df_bkg_mp["mass_point"] = mp
        dmat_b = xgb.DMatrix(df_bkg_mp[features].values, feature_names=features)
        pred_b = model.predict(dmat_b)
        w_b = df_bkg["phys_wgt"].values

        y_true = np.concatenate([np.ones(len(pred_s)), np.zeros(len(pred_b))])
        y_pred = np.concatenate([pred_s, pred_b])
        weights = np.concatenate([w_s, w_b])

        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, sample_weight=weights)
        auc_val = metrics.auc(fpr, tpr)
        auc_per_mass[mp] = auc_val

        ax.plot(tpr, 1 - fpr, color=cmap(i), label=f"$m_a = {mp}$ GeV (AUC = {auc_val:.3f})", linewidth=1.5)

    ax.set_xlabel("Signal efficiency (TPR)", fontsize=12)
    ax.set_ylabel("1 - Background efficiency (1 - FPR)", fontsize=12)
    ax.set_title("ROC curves per mass hypothesis\n(Background re-scored per hypothesis, signal at true $m_a$)", fontsize=12)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_all = os.path.join(args.outdir, "roc_per_mass.png")
    plt.savefig(out_all, dpi=300)
    plt.savefig(os.path.join(args.outdir, "roc_per_mass.pdf"))
    plt.close()
    print(f"[Figure 9a Generated] Saved {out_all}")

    # Case 1 overlay at 20, 40, 60 GeV (Figure 2c)
    fig, ax = plt.subplots(figsize=(7, 7))
    styles = {20: ("teal", "-"), 40: ("darkorange", "--"), 60: ("purple", ":")}
    for mp, (color, ls) in styles.items():
        if mp in auc_per_mass:
            sub_s = df_sig[df_sig["mass_point"] == mp]
            dmat_s = xgb.DMatrix(sub_s[features].values, feature_names=features)
            pred_s = model.predict(dmat_s)

            df_bkg_mp = df_bkg.copy()
            df_bkg_mp["mass_point"] = mp
            dmat_b = xgb.DMatrix(df_bkg_mp[features].values, feature_names=features)
            pred_b = model.predict(dmat_b)

            y_true = np.concatenate([np.ones(len(pred_s)), np.zeros(len(pred_b))])
            y_pred = np.concatenate([pred_s, pred_b])
            weights = np.concatenate([sub_s["phys_wgt"].values, df_bkg["phys_wgt"].values])

            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, sample_weight=weights)
            ax.plot(tpr, 1 - fpr, color=color, linestyle=ls, linewidth=2,
                    label=f"$m_a = {mp}$ GeV (AUC = {auc_per_mass[mp]:.3f})")

    ax.set_xlabel("Signal efficiency (TPR)", fontsize=12)
    ax.set_ylabel("1 - Background efficiency (1 - FPR)", fontsize=12)
    ax.set_title("ROC curves — $m_a = 20, 40, 60$ GeV", fontsize=13)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_overlay = os.path.join(args.outdir, "roc_case1_mass_20_40_60.png")
    plt.savefig(out_overlay, dpi=300)
    plt.savefig(os.path.join(args.outdir, "roc_case1_mass_20_40_60.pdf"))
    plt.close()
    print(f"[Figure 2c Generated] Saved {out_overlay}")


if __name__ == "__main__":
    main()
