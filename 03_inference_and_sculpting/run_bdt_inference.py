"""
run_bdt_inference.py
====================
Runs BDT inference and generates pre-cut and post-cut ROOT histograms.

Corresponds to Semester Report:
- Section 6.2: Drell-Yan Spikes and Inference Evaluation Fix
  "In the updated procedure, every background event is evaluated across all
   11 mass hypotheses individually (m_a in {12, ..., 60} GeV). This provides
   the proper, hypothesis-dependent background rejection necessary for computing
   realistic exclusion limits."

Outputs:
- ROOT histogram file containing precut and per-mass postcut histograms for
  diphoton mass m_gg, BDT score, and input features.
"""

import os
import sys
import argparse
import pickle
import numpy as np
import uproot
import awkward as ak
import xgboost as xgb

# Ensure config can be found from parent directory or current dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    INPUT_VARS,
    MASS_BRANCH,
    MASS_POINTS,
    TRAIN_WEIGHT_BRANCH,
    PLOT_WEIGHT_BRANCH,
    N_BINS_FEATURE,
    N_BINS_MGG, MGG_RANGE,
    N_BINS_BDT, BDT_RANGE,
    PLOT_SETTINGS,
    LUMI, XSEC_SIG, XSEC_BKG,
    get_physics_weights_signal,
    get_physics_weights_background,
    find_data_file
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BDT inference with per-hypothesis background evaluation fix."
    )
    parser.add_argument(
        "--model",
        default="HtoAATo2b2g_run2.pkl",
        help="Pickled XGBoost model file"
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
        "--bdt_cut",
        type=float,
        default=0.7,
        help="BDT score threshold for post-cut histograms"
    )
    parser.add_argument(
        "--outdir",
        default="plots/histograms",
        help="Output directory"
    )
    parser.add_argument(
        "--outfile",
        default="histograms_physwgt.root",
        help="Output ROOT histogram filename"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    model_path = find_data_file(args.model)
    sig_path = find_data_file(args.sig)
    bkg_path = find_data_file(args.bkg)

    print(f"[Inference Pipeline] Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        m_dict = pickle.load(f)
    model = m_dict["xgbModel"] if isinstance(m_dict, dict) and "xgbModel" in m_dict else m_dict
    features = m_dict.get("features", list(INPUT_VARS)) if isinstance(m_dict, dict) else list(INPUT_VARS)

    # -------------------------------------------------------------------------
    # 1. Load Background & Signal
    # -------------------------------------------------------------------------
    branches = list(set(features + ["CMS_hgg_mass", "process_id", "evt_wgt", "mass_point"]))

    print(f"[Inference Pipeline] Reading Background from: {bkg_path}")
    with uproot.open(bkg_path) as f_bkg:
        bkg_ak = f_bkg["DiphotonTree"].arrays(branches, library="ak")
        df_bkg = ak.to_dataframe(bkg_ak)

    print(f"[Inference Pipeline] Reading Signal from: {sig_path}")
    with uproot.open(sig_path) as f_sig:
        sig_ak = f_sig["DiphotonTree"].arrays(branches, library="ak")
        df_sig = ak.to_dataframe(sig_ak)

    # Physics weights (Lumi = 109 fb^-1)
    df_sig["phys_wgt"] = get_physics_weights_signal(df_sig, lumi=LUMI)
    df_bkg["phys_wgt"] = get_physics_weights_background(df_bkg, lumi=LUMI)

    # -------------------------------------------------------------------------
    # 2. Evaluate Signal (True Hypothesis)
    # -------------------------------------------------------------------------
    print("\n[Inference Pipeline] Scoring Signal under true mass hypotheses...")
    dmatrix_sig = xgb.DMatrix(df_sig[features].values, feature_names=features)
    df_sig["bdt_score"] = model.predict(dmatrix_sig)

    # -------------------------------------------------------------------------
    # 3. Evaluate Background (Per-Hypothesis Evaluation Fix - Section 6.2)
    # -------------------------------------------------------------------------
    print("[Inference Pipeline] Scoring Background under EVERY mass hypothesis...")
    bkg_scores_per_mp = {}
    for mp in MASS_POINTS:
        df_bkg_mp = df_bkg.copy()
        df_bkg_mp["mass_point"] = mp
        dmat = xgb.DMatrix(df_bkg_mp[features].values, feature_names=features)
        bkg_scores_per_mp[mp] = model.predict(dmat)

    # -------------------------------------------------------------------------
    # 4. Fill and Export ROOT Histograms
    # -------------------------------------------------------------------------
    out_root = os.path.join(args.outdir, args.outfile)
    print(f"\n[Inference Pipeline] Exporting histograms to: {out_root}")

    out_hists = {}

    # Precut & Postcut for m_gg
    bins_mgg = np.linspace(MGG_RANGE[0], MGG_RANGE[1], N_BINS_MGG + 1)
    bins_bdt = np.linspace(BDT_RANGE[0], BDT_RANGE[1], N_BINS_BDT + 1)

    # Precut Signal
    for mp in MASS_POINTS:
        sub_s = df_sig[df_sig["mass_point"] == mp]
        h_pre, _ = np.histogram(sub_s["CMS_hgg_mass"], bins=bins_mgg, weights=sub_s["phys_wgt"])
        out_hists[f"mgg_sig_m{mp}_precut"] = (h_pre, bins_mgg)

        # Postcut Signal
        sub_s_cut = sub_s[sub_s["bdt_score"] > args.bdt_cut]
        h_post, _ = np.histogram(sub_s_cut["CMS_hgg_mass"], bins=bins_mgg, weights=sub_s_cut["phys_wgt"])
        out_hists[f"mgg_sig_m{mp}_postcut"] = (h_post, bins_mgg)

    # Precut Background (per PID)
    for pid in df_bkg["process_id"].unique():
        sub_b = df_bkg[df_bkg["process_id"] == pid]
        h_pre, _ = np.histogram(sub_b["CMS_hgg_mass"], bins=bins_mgg, weights=sub_b["phys_wgt"])
        out_hists[f"mgg_bkg_pid{pid}_precut"] = (h_pre, bins_mgg)

        # Postcut Background (Per hypothesis!)
        for mp in MASS_POINTS:
            mask_cut = bkg_scores_per_mp[mp][df_bkg["process_id"] == pid] > args.bdt_cut
            h_post, _ = np.histogram(sub_b["CMS_hgg_mass"][mask_cut], bins=bins_mgg, weights=sub_b["phys_wgt"][mask_cut])
            out_hists[f"mgg_bkg_pid{pid}_m{mp}_postcut"] = (h_post, bins_mgg)

    # Save features precut histograms
    for var in features:
        if var in PLOT_SETTINGS:
            cfg = PLOT_SETTINGS[var]
            bins_var = np.linspace(cfg["range"][0], cfg["range"][1], N_BINS_FEATURE + 1)

            # Signal precut
            for mp in MASS_POINTS:
                sub_s = df_sig[df_sig["mass_point"] == mp]
                h, _ = np.histogram(sub_s[var], bins=bins_var, weights=sub_s["phys_wgt"])
                out_hists[f"{var}_sig_m{mp}_precut"] = (h, bins_var)

            # Bkg precut
            for pid in df_bkg["process_id"].unique():
                sub_b = df_bkg[df_bkg["process_id"] == pid]
                h, _ = np.histogram(sub_b[var], bins=bins_var, weights=sub_b["phys_wgt"])
                out_hists[f"{var}_bkg_pid{pid}_precut"] = (h, bins_var)

    with uproot.recreate(out_root) as f_out:
        for name, (h, b_edges) in out_hists.items():
            f_out[name] = (h, b_edges)

    print(f"[Done] Exported {len(out_hists)} histograms successfully to {out_root}")


if __name__ == "__main__":
    main()
