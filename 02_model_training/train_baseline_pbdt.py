"""
train_baseline_pbdt.py
======================
Trains the Run 2 baseline Parameterized Boosted Decision Tree (pBDT).

Corresponds to Semester Report:
- Section 5.1: Baseline Configuration and Performance (19 features / Case 1: 18 features)
- Section 5.3: Capacity Scaling (10 boosting rounds vs. 40 boosting rounds)
- Section 3.2: Signal Mass Equalization training weight scheme
- Figures: Figure 2a (ROC curve), Figure 2b (BDT score separation)
- Tables: Table 7 (Input features), Table 10 (BDT configurations summary)
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import awkward as ak
import uproot
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Ensure config can be found from parent directory or current dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    INPUT_VARS,
    MASS_POINTS,
    find_data_file
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Run 2 baseline parameterized BDT (pBDT)."
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
        help="TTree name in ROOT files"
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=10,
        help="Number of boosting rounds / trees (Report uses 10 for baseline, 40 for capacity study)"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=3,
        help="Maximum tree depth (default: 3)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.5,
        help="Learning rate eta (default: 0.5)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Minimum loss reduction gamma (default: 0.1)"
    )
    parser.add_argument(
        "--omit_met_angle",
        action="store_true",
        help="Omit delphi_ggMET_PF (Run 2 Case 1 with 18 features)"
    )
    parser.add_argument(
        "--equalize_weights",
        action="store_true",
        default=True,
        help="Apply 1/11 signal mass equalization scheme (Section 3.2)"
    )
    parser.add_argument(
        "--outdir",
        default="models",
        help="Output directory for model and diagnostic plots"
    )
    parser.add_argument(
        "--model_name",
        default="baseline_pbdt",
        help="Base name for exported model and plots"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for train/test split"
    )
    return parser.parse_args()


def load_dataset(sig_path, bkg_path, tree_name, feature_list, equalize_weights, seed=12345):
    print(f"[DataLoader] Reading Signal from {sig_path}")
    branches_sig = list(set(feature_list + ["evt_wgt", "mass_point"]))
    with uproot.open(sig_path) as f_sig:
        sig_ak = f_sig[tree_name].arrays(branches_sig, library="ak")
        df_sig = ak.to_dataframe(sig_ak)

    print(f"[DataLoader] Reading Background from {bkg_path}")
    branches_bkg = list(set(feature_list + ["evt_wgt", "mass_point"]))
    with uproot.open(bkg_path) as f_bkg:
        bkg_ak = f_bkg[tree_name].arrays(branches_bkg, library="ak")
        df_bkg = ak.to_dataframe(bkg_ak)

    # Subsample background to 1/5 for training balance as done in baseline
    df_bkg = df_bkg.sample(frac=0.20, random_state=seed)

    df_sig["target"] = 1
    df_bkg["target"] = 0

    # -------------------------------------------------------------------------
    # Weight handling (Section 3.1 & 3.2)
    # -------------------------------------------------------------------------
    if equalize_weights:
        # Equalize signal mass hypotheses so each of 11 mass points gets 1/11 total share
        df_sig["evt_wgt"] = 0.0
        mps = np.sort(df_sig["mass_point"].unique())
        n_mps = len(mps)
        for mp in mps:
            mask = df_sig["mass_point"] == mp
            n_ev = mask.sum()
            df_sig.loc[mask, "evt_wgt"] = (1.0 / n_mps) / n_ev
    else:
        # Standard positive weight rectification for signal
        sig_sum = df_sig["evt_wgt"].sum()
        df_sig["evt_wgt"] = df_sig["evt_wgt"].abs()
        df_sig["evt_wgt"] *= (sig_sum / df_sig["evt_wgt"].sum())

    # Background negative weight rectification (Equation 4)
    bkg_sum = df_bkg["evt_wgt"].sum()
    df_bkg["evt_wgt"] = df_bkg["evt_wgt"].abs()
    df_bkg["evt_wgt"] *= (bkg_sum / df_bkg["evt_wgt"].sum())

    # Scale total signal to match total background (50:50 effective training ratio)
    sig_total = df_sig["evt_wgt"].sum()
    bkg_total = df_bkg["evt_wgt"].sum()
    df_sig["evt_wgt"] *= (bkg_total / sig_total)

    # Combine datasets
    columns = ["evt_wgt", "target"] + feature_list
    df_all = pd.concat([df_sig[columns], df_bkg[columns]], ignore_index=True)
    return df_all


def plot_diagnostics(y_train, y_test, pred_train, pred_test, w_train, w_test, outdir, name):
    # 1. ROC Curve (Figure 2a)
    fpr_tr, tpr_tr, _ = metrics.roc_curve(y_train, pred_train, sample_weight=w_train)
    auc_tr = metrics.auc(fpr_tr, tpr_tr)

    fpr_te, tpr_te, _ = metrics.roc_curve(y_test, pred_test, sample_weight=w_test)
    auc_te = metrics.auc(fpr_te, tpr_te)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(tpr_tr, 1 - fpr_tr, label=f"Train: AUC = {auc_tr:.3f}", color="blue", linewidth=1.5)
    ax.plot(tpr_te, 1 - fpr_te, label=f"Test:  AUC = {auc_te:.3f}", color="darkorange", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Signal Efficiency (TPR)", fontsize=12)
    ax.set_ylabel("1 - Background Efficiency (1 - FPR)", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(loc="lower left", fontsize=11, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_roc.png"), dpi=300)
    plt.savefig(os.path.join(outdir, f"{name}_roc.pdf"))
    plt.close()

    # 2. BDT Score Distribution (Figure 2b)
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, 1, 41)
    centers = (bins[:-1] + bins[1:]) / 2.0
    width = bins[1] - bins[0]

    sig_tr = pred_train[y_train == 1]
    bkg_tr = pred_train[y_train == 0]
    w_sig_tr = w_train[y_train == 1]
    w_bkg_tr = w_train[y_train == 0]

    sig_te = pred_test[y_test == 1]
    bkg_te = pred_test[y_test == 0]
    w_sig_te = w_test[y_test == 1] * (w_sig_tr.sum() / w_test[y_test == 1].sum())
    w_bkg_te = w_test[y_test == 0] * (w_bkg_tr.sum() / w_test[y_test == 0].sum())

    h_sig_tr, _ = np.histogram(sig_tr, bins=bins, weights=w_sig_tr)
    h_bkg_tr, _ = np.histogram(bkg_tr, bins=bins, weights=w_bkg_tr)
    h_sig_te, _ = np.histogram(sig_te, bins=bins, weights=w_sig_te)
    h_bkg_te, _ = np.histogram(bkg_te, bins=bins, weights=w_bkg_te)

    ax.bar(centers, h_sig_tr, width=width, color="blue", alpha=0.4, label="Sig (Train)")
    ax.bar(centers, h_bkg_tr, width=width, color="red", alpha=0.4, label="Bkg (Train)")
    ax.errorbar(centers, h_sig_te, yerr=np.sqrt(np.maximum(h_sig_te, 0)), fmt="o", color="blue", label="Sig (Test)")
    ax.errorbar(centers, h_bkg_te, yerr=np.sqrt(np.maximum(h_bkg_te, 0)), fmt="o", color="red", label="Bkg (Test)")

    ax.set_xlabel("BDT Output Score", fontsize=12)
    ax.set_ylabel("Weighted Frequency", fontsize=12)
    ax.set_title("BDT Signal-Background Separation", fontsize=14)
    ax.legend(loc="upper center", ncol=2, fontsize=11, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_separation.png"), dpi=300)
    plt.savefig(os.path.join(outdir, f"{name}_separation.pdf"))
    plt.close()

    print(f"\n[Training Summary] Train AUC: {auc_tr:.4f} | Test AUC: {auc_te:.4f}")
    return auc_tr, auc_te


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sig_file = find_data_file(args.sig)
    bkg_file = find_data_file(args.bkg)

    feature_list = list(INPUT_VARS)
    if args.omit_met_angle and "delphi_ggMET_PF" in feature_list:
        feature_list.remove("delphi_ggMET_PF")
        print("[Config] Case 1 selected: Omitting delphi_ggMET_PF (18 features).")
    else:
        print(f"[Config] Full baseline feature set ({len(feature_list)} features).")

    df = load_dataset(sig_file, bkg_file, args.tree, feature_list, args.equalize_weights, seed=args.seed)

    x = df[feature_list].values
    y = df["target"].values
    w = df["evt_wgt"].values * 1e5

    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
        x, y, w, test_size=0.20, random_state=args.seed, shuffle=True
    )

    params = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": args.lr,
        "max_depth": args.max_depth,
        "gamma": args.gamma,
        "seed": args.seed,
    }

    dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train, feature_names=feature_list)
    dtest = xgb.DMatrix(x_test, label=y_test, weight=w_test, feature_names=feature_list)

    print(f"\n[XGBoost Training] Training {args.trees} trees of depth {args.max_depth}...")
    model = xgb.train(params, dtrain, num_boost_round=args.trees)

    pred_train = model.predict(dtrain)
    pred_test = model.predict(dtest)

    plot_diagnostics(y_train, y_test, pred_train, pred_test, w_train, w_test, args.outdir, args.model_name)

    # Export model package
    export_dict = {
        "xgbModel": model,
        "features": feature_list,
        "params": params,
        "n_trees": args.trees,
        "equalized": args.equalize_weights
    }
    model_pkl = os.path.join(args.outdir, f"{args.model_name}.pkl")
    with open(model_pkl, "wb") as f_out:
        pickle.dump(export_dict, f_out)
    print(f"[Model Saved] Pickled model written to: {model_pkl}")


if __name__ == "__main__":
    main()
