"""
plot_engineered_features.py
===========================
Derives and plots stacked background vs signal distributions for engineered features.

Corresponds to Semester Report:
- Section 7.1: Constructed Kinematic Observables (Table 9)
- Figure 7: Selected engineered kinematic observables: stacked background versus signal (x100)
  for m_a = 20, 40, 60 GeV before BDT selection.
  (a) Opening angle alpha_gg [rad]
  (b) Delta-R(bb, gamma-gamma)
  (c) pT asymmetry (bb, gamma-gamma)
  (d) mT(l, MET) [GeV]
"""

import os
import sys
import argparse
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt

# Ensure config and feature library are found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from config import PROCESS_NAMES, LUMI, get_physics_weights_signal, get_physics_weights_background, find_data_file
from kinematic_features import derive_all_features

PROCESS_COLORS = {
    0: "#8B4513",  # TTG1Jets: brown
    1: "#1f77b4",  # TTto2L2Nu: dark blue
    2: "#aec7e8",  # TTtoLNu2Q: light blue
    3: "#2ca02c",  # WGtoLNuG: green
    4: "#ff7f0e",  # DYto2Mu50: orange
    5: "#d62728",  # DYto2E50: red
}

FEATURE_CONFIGS = {
    "alpha_gg": {"range": (0.0, 3.14159), "bins": 50, "label": r"Opening angle $\alpha_{\gamma\gamma}$ [rad]"},
    "dR_bb_gg": {"range": (0.0, 5.0), "bins": 50, "label": r"$\Delta R(bb, \gamma\gamma)$"},
    "pt_asym_bbgg": {"range": (-1.0, 1.0), "bins": 50, "label": r"$p_T$ asymmetry $(bb, \gamma\gamma)$"},
    "mT_l_MET": {"range": (0.0, 200.0), "bins": 50, "label": r"$m_T(\ell, E_T^{\rm miss})$ [GeV]"},
    "dR_l_bnearest": {"range": (0.0, 5.0), "bins": 50, "label": r"$\Delta R(\ell, b_{\rm nearest})$"},
    "met_significance": {"range": (0.0, 20.0), "bins": 50, "label": r"$E_T^{\rm miss} / \sqrt{\sum E_T}$"},
    "pt_bb": {"range": (0.0, 300.0), "bins": 50, "label": r"$p_T(bb)$ [GeV]"},
    "pt_gg": {"range": (0.0, 300.0), "bins": 50, "label": r"$p_T(\gamma\gamma)$ [GeV]"},
    "M_bb": {"range": (0.0, 250.0), "bins": 50, "label": r"$m_{bb}$ [GeV]"},
    "M_bbgg": {"range": (0.0, 400.0), "bins": 50, "label": r"$m_{bb\gamma\gamma}$ [GeV]"},
}

RAW_BRANCHES = [
    "pholead_pt", "pholead_eta", "pholead_phi", "pholead_mass",
    "phosublead_pt", "phosublead_eta", "phosublead_phi", "phosublead_mass",
    "first_jet_pt", "first_jet_eta", "first_jet_phi", "first_jet_mass",
    "second_jet_pt", "second_jet_eta", "second_jet_phi", "second_jet_mass",
    "electron_pt", "electron_eta", "electron_phi",
    "muon_phi",
    "leppt", "lepeta",
    "PFMET_pt", "PFMET_phi", "PFMET_sumEt",
    "PuppiMET_pt",
    "mass_point",
    "evt_wgt",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive and plot stacked distributions for engineered observables."
    )
    parser.add_argument("--bkg", default="merged_bkg_withMET.root", help="Background ROOT file")
    parser.add_argument("--sig", default="merged_signal_withMET.root", help="Signal ROOT file")
    parser.add_argument("--tree", default="DiphotonTree", help="Tree name")
    parser.add_argument("--outdir", default="plots/engineered_features", help="Output directory")
    parser.add_argument("--sig_masses", default="20,40,60", help="Signal mass points to overlay")
    parser.add_argument("--sig_boost", type=float, default=100.0, help="Signal scaling factor (default: 100)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sig_file = find_data_file(args.sig)
    bkg_file = find_data_file(args.bkg)

    print(f"[Feature Plotting] Reading signal from: {sig_file}")
    with uproot.open(sig_file) as f_sig:
        sig_ak = f_sig[args.tree].arrays(RAW_BRANCHES, library="ak")
        sig_ak = derive_all_features(sig_ak)
        df_sig = ak.to_dataframe(sig_ak)

    print(f"[Feature Plotting] Reading background from: {bkg_file}")
    with uproot.open(bkg_file) as f_bkg:
        bkg_ak = f_bkg[args.tree].arrays(RAW_BRANCHES + ["process_id"], library="ak")
        bkg_ak = derive_all_features(bkg_ak)
        df_bkg = ak.to_dataframe(bkg_ak)

    df_sig["phys_wgt"] = get_physics_weights_signal(df_sig, lumi=LUMI)
    df_bkg["phys_wgt"] = get_physics_weights_background(df_bkg, lumi=LUMI)

    sig_mps = [int(m.strip()) for m in args.sig_masses.split(",")]

    print("\n[Feature Plotting] Generating stacked plots...")
    for var, cfg in FEATURE_CONFIGS.items():
        if var not in df_sig.columns:
            continue

        bins = np.linspace(cfg["range"][0], cfg["range"][1], cfg["bins"] + 1)
        centers = (bins[:-1] + bins[1:]) / 2.0
        width = bins[1] - bins[0]

        fig, ax = plt.subplots(figsize=(8, 6))
        bottom = np.zeros(len(centers))

        # Background stack
        for pid in [0, 1, 2, 3, 4, 5]:
            sub = df_bkg[df_bkg["process_id"] == pid]
            counts, _ = np.histogram(sub[var], bins=bins, weights=sub["phys_wgt"])
            pname = PROCESS_NAMES.get(pid, f"PID {pid}")
            color = PROCESS_COLORS.get(pid, "gray")
            ax.bar(centers, counts, width=width, bottom=bottom, color=color, label=pname)
            bottom += counts

        # Overlay signal (scaled by sig_boost)
        sig_styles = [("black", "-"), ("darkviolet", "--"), ("teal", ":")]
        for i, mp in enumerate(sig_mps):
            sub_s = df_sig[df_sig["mass_point"] == mp]
            s_counts, _ = np.histogram(sub_s[var], bins=bins, weights=sub_s["phys_wgt"] * args.sig_boost)
            c, ls = sig_styles[i % len(sig_styles)]
            lbl = f"$m_a = {mp}$ GeV ($\\times {int(args.sig_boost)}$)"
            ax.step(bins, np.append(s_counts, s_counts[-1]), where="post", color=c, linestyle=ls, linewidth=1.8, label=lbl)

        ax.set_xlabel(cfg["label"], fontsize=12)
        ax.set_ylabel(f"Events / {width:.2f}", fontsize=12)
        ax.set_title(f"{cfg['label']} (Before BDT Selection)", fontsize=13)
        ax.set_xlim(cfg["range"][0], cfg["range"][1])
        ax.set_ylim(0, max(bottom.max() * 1.35, 1.0))
        ax.legend(loc="upper right", frameon=True, fontsize=9, ncol=2)
        ax.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        out_png = os.path.join(args.outdir, f"{var}.png")
        plt.savefig(out_png, dpi=300)
        plt.savefig(os.path.join(args.outdir, f"{var}.pdf"))
        plt.close()
        print(f"  Generated {out_png}")

    print("\n[Done] All engineered feature plots successfully generated!")


if __name__ == "__main__":
    main()
