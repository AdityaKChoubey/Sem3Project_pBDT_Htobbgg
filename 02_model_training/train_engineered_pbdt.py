"""
train_engineered_pbdt.py
========================
Trains the Parameterized BDT with newly constructed kinematic features.

Corresponds to Semester Report:
- Section 7.1: Constructed Kinematic Observables (Table 9)
- Section 7.2: Evaluation of New Feature Model (Figure 8, Table 10)

Excludes M_gg and abs_mbb_minus_mgg from classifier training to prevent
diphoton mass sculpting.
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

# Import config and feature extraction library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '04_feature_engineering')))
from config import find_data_file, ENGINEERED_BDT_FEATURES
from kinematic_features import derive_all_features


COMMON_BRANCHES = [
    "pholead_pt", "pholead_eta", "pholead_phi", "pholead_mass",
    "phosublead_pt", "phosublead_eta", "phosublead_phi", "phosublead_mass",
    "first_jet_pt", "first_jet_eta", "first_jet_phi", "first_jet_mass",
    "second_jet_pt", "second_jet_eta", "second_jet_phi", "second_jet_mass",
    "electron_pt", "electron_phi",
    "muon_phi",
    "leppt", "lepeta",
    "PFMET_pt", "PFMET_phi", "PFMET_sumEt",
    "PuppiMET_pt",
    "mass_point",
    "evt_wgt",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train pBDT with newly engineered kinematic observables."
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
        help="Number of boosting rounds (default: 10)"
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
        "--outdir",
        default="models",
        help="Output directory"
    )
    parser.add_argument(
        "--model_name",
        default="engineered_features_pbdt",
        help="Base name for exported model and diagnostics"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for train/test split"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sig_file = find_data_file(args.sig)
    bkg_file = find_data_file(args.bkg)

    print(f"[DataLoader] Reading Signal from {sig_file}")
    with uproot.open(sig_file) as f_sig:
        sig_ak = f_sig[args.tree].arrays(COMMON_BRANCHES, library="ak")
        sig_derived = derive_all_features(sig_ak)
        df_sig = ak.to_dataframe(sig_derived)

    print(f"[DataLoader] Reading Background from {bkg_file}")
    with uproot.open(bkg_file) as f_bkg:
        bkg_ak = f_bkg[args.tree].arrays(COMMON_BRANCHES + ["process_id"], library="ak")
        bkg_derived = derive_all_features(bkg_ak)
        df_bkg = ak.to_dataframe(bkg_derived)

    # Subsample background
    df_bkg = df_bkg.sample(frac=0.20, random_state=args.seed)

    df_sig["target"] = 1
    df_bkg["target"] = 0

    # Equalize signal mass hypotheses
    df_sig["evt_wgt"] = 0.0
    mps = np.sort(df_sig["mass_point"].unique())
    n_mps = len(mps)
    for mp in mps:
        mask = df_sig["mass_point"] == mp
        n_ev = mask.sum()
        df_sig.loc[mask, "evt_wgt"] = (1.0 / n_mps) / n_ev

    # Background negative weight rectification (Equation 4)
    bkg_sum = df_bkg["evt_wgt"].sum()
    df_bkg["evt_wgt"] = df_bkg["evt_wgt"].abs()
    df_bkg["evt_wgt"] *= (bkg_sum / df_bkg["evt_wgt"].sum())

    # Balance 50:50 effective ratio
    sig_total = df_sig["evt_wgt"].sum()
    bkg_total = df_bkg["evt_wgt"].sum()
    df_sig["evt_wgt"] *= (bkg_total / sig_total)

    # Select engineered feature list
    feature_list = [f for f in ENGINEERED_BDT_FEATURES if f in df_sig.columns]
    print(f"\n[Features Selected] Using {len(feature_list)} features for training:")
    for i, feat in enumerate(feature_list):
        print(f"  {i+1:2d}. {feat}")

    columns = ["evt_wgt", "target"] + feature_list
    df_all = pd.concat([df_sig[columns], df_bkg[columns]], ignore_index=True)

    x = df_all[feature_list].values
    y = df_all["target"].values
    w = df_all["evt_wgt"].values * 1e5

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

    fpr_tr, tpr_tr, _ = metrics.roc_curve(y_train, pred_train, sample_weight=w_train)
    auc_tr = metrics.auc(fpr_tr, tpr_tr)

    fpr_te, tpr_te, _ = metrics.roc_curve(y_test, pred_test, sample_weight=w_test)
    auc_te = metrics.auc(fpr_te, tpr_te)

    # Plot ROC curve (Figure 8b)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(tpr_tr, 1 - fpr_tr, label=f"Train: AUC = {auc_tr:.3f}", color="blue", linewidth=1.5)
    ax.plot(tpr_te, 1 - fpr_te, label=f"Test:  AUC = {auc_te:.3f}", color="darkorange", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Signal Efficiency (TPR)", fontsize=12)
    ax.set_ylabel("1 - Background Efficiency (1 - FPR)", fontsize=12)
    ax.set_title("New Feature pBDT: ROC Curve", fontsize=14)
    ax.legend(loc="lower left", fontsize=11, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.model_name}_roc.png"), dpi=300)
    plt.savefig(os.path.join(args.outdir, f"{args.model_name}_roc.pdf"))
    plt.close()

    # Export model package
    export_dict = {
        "xgbModel": model,
        "features": feature_list,
        "params": params,
        "n_trees": args.trees,
        "equalized": True
    }
    model_pkl = os.path.join(args.outdir, f"{args.model_name}.pkl")
    with open(model_pkl, "wb") as f_out:
        pickle.dump(export_dict, f_out)

    print(f"\n[Training Summary] Train AUC: {auc_tr:.4f} | Test AUC: {auc_te:.4f}")
    print(f"[Model Saved] Pickled model written to: {model_pkl}")


if __name__ == "__main__":
    main()
