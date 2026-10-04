# Multivariate Analysis Suite for $WH \to \ell\nu aa \to \ell\nu b\bar{b}\gamma\gamma$

**Complementary Codebase for the Mid-Semester Project Report**  
**Author:** Aditya Kumar Choubey  
**Affiliation:** Department of Physics, Indian Institute of Science Education and Research (IISER), Pune  
**Contact:** `aditya.choubey@students.iiserpune.ac.in`  
**Date:** October 2026  

---

## 1. Executive Summary & Physics Context

This repository provides the complete, modular Python analysis suite for the multivariate search for exotic Higgs decays into light pseudoscalars:
$$pp \to W^\pm H + X \to (\ell^\pm \nu) + (aa) \to \ell^\pm \nu b\bar{b}\gamma\gamma + X, \quad (\ell = e, \mu)$$
at the CMS experiment ($\sqrt{s} = 13.6\text{ TeV}$, $\mathcal{L} = 109\text{ fb}^{-1}$) for pseudoscalar masses $m_a \in [12, 60]\text{ GeV}$.

The suite reproduces all quantitative results, tables, and figures in the accompanying project report:
- **Event Weighting & Equalization:** Eliminates the 18-fold training imbalance favoring high-mass signals and rectifies NLO negative weights.
- **Run 2 Baseline pBDT Reproduction & Diagnostics:** Benchmarks 18- and 19-variable models, extracts feature importance by weight and gain, and inspects tree splits across mass hypotheses.
- **Inference Evaluation Correction:** Eliminates improper random assignment by evaluating every background event across all 11 mass hypotheses individually.
- **Diphoton Mass Sculpting & Correlation Study:** Identifies kinematic mechanisms driving $m_{\gamma\gamma}$ distortions ($p_T^{\gamma_2}/m_{\gamma\gamma}$, $\Delta\phi_{\gamma\gamma}$) and isolates Drell–Yan statistical fluctuations.
- **Kinematic Feature Engineering:** Constructs 24 topological, recoil, transverse mass, and angular observables.
- **Statistical Sensitivity & AMS Optimization:** Scans BDT selection thresholds to compute the Approximate Median Significance (AMS) across all mass points.

---

## 2. Directory Layout

```
analysis_code/
├── config.py                          # Central configuration (cross-sections, mass points, variables, plotting)
├── requirements.txt                   # Python package dependencies
├── run_pipeline.py                    # Master CLI pipeline orchestrator
│
├── 01_dataset_and_weights/
│   └── check_dataset_and_weights.py   # Audits yields, negative weights (Table 4) & equalization (Table 5, Fig 1)
│
├── 02_model_training/
│   ├── train_baseline_pbdt.py         # Trains baseline Run 2 pBDT (10/40 trees, equalized/raw) (Fig 2, Tab 7)
│   ├── train_engineered_pbdt.py       # Trains pBDT using engineered features (Sec 7.2, Fig 8b, Tab 10)
│   └── inspect_trees.py               # Inspects decision trees & counts mass_point splits (Fig 4, Tab 8)
│
├── 03_inference_and_sculpting/
│   ├── run_bdt_inference.py           # Per-hypothesis background evaluation & ROOT histogram export (Sec 6.2)
│   ├── plot_mgg_sculpting.py          # Pre-cut and post-cut diphoton mass distributions (Figure 5)
│   ├── plot_mgg_sculpting_no_dy.py    # Post-cut m_gg without Drell-Yan, isolating ttbar continuum (Sec 6.2)
│   ├── plot_correlations.py           # Pearson correlation matrices for signal & background (Figure 6)
│   ├── plot_feature_importance.py     # Feature importance by weight and gain (Figure 3 & Figure 8a)
│   ├── plot_features_precut.py        # Input feature distributions before BDT cut
│   └── plot_mass_hypothesis.py        # Background & signal distribution across mass hypotheses (Sec 4)
│
├── 04_feature_engineering/
│   ├── kinematic_features.py          # Modular library deriving 24 topological/angular observables (Table 9)
│   └── plot_engineered_features.py    # Stacked background vs signal distributions for new features (Figure 7)
│
└── 05_sensitivity_and_ams/
    ├── plot_roc_per_mass.py           # Per-mass ROC curves with hypothesis-dependent background (Fig 2c, 9a)
    └── calculate_ams_scan.py          # 1001-step threshold scan for AMS vs mass hypothesis (Table 11, Fig 9b)
```

---

## 3. Mapping Code to Report Sections, Tables, and Figures

| Report Item | Description | Generating Script | CLI Command |
| :--- | :--- | :--- | :--- |
| **Table 3** | Simulated processes, cross sections, yields | `01_dataset_and_weights/check_dataset_and_weights.py` | `python check_dataset_and_weights.py` |
| **Table 4** | Negative weight fractions per process | `01_dataset_and_weights/check_dataset_and_weights.py` | `python check_dataset_and_weights.py` |
| **Table 5** | Raw events & equalized weight share | `01_dataset_and_weights/check_dataset_and_weights.py` | `python check_dataset_and_weights.py` |
| **Figure 1** | Raw counts & equalized training weights | `01_dataset_and_weights/check_dataset_and_weights.py` | `python check_dataset_and_weights.py` |
| **Table 7** | Baseline pBDT 19 input features | `config.py` (`INPUT_VARS`) | `python config.py` |
| **Figure 2a,b** | Baseline ROC curve & BDT score separation | `02_model_training/train_baseline_pbdt.py` | `python train_baseline_pbdt.py --trees 10` |
| **Figure 2c,d** | Case 1 ROC at 20, 40, 60 GeV & Stacked BDT | `05_sensitivity_and_ams/plot_roc_per_mass.py` | `python plot_roc_per_mass.py` |
| **Figure 3** | Feature importance by weight and gain | `03_inference_and_sculpting/plot_feature_importance.py` | `python plot_feature_importance.py` |
| **Figure 4** | Decision tree 1 and tree 4 dumps | `02_model_training/inspect_trees.py` | `python inspect_trees.py` |
| **Table 8** | Splits on `mass_point` (10 vs 40 trees) | `02_model_training/inspect_trees.py` | `python inspect_trees.py` |
| **Figure 5** | Diphoton mass sculpting before/after cut | `03_inference_and_sculpting/plot_mgg_sculpting.py` | `python plot_mgg_sculpting.py` |
| **Section 6.2** | Sculpting without Drell–Yan spikes | `03_inference_and_sculpting/plot_mgg_sculpting_no_dy.py` | `python plot_mgg_sculpting_no_dy.py` |
| **Figure 6** | Pearson correlation matrices (signal & bkg) | `03_inference_and_sculpting/plot_correlations.py` | `python plot_correlations.py` |
| **Table 9** | 24 newly constructed kinematic observables | `04_feature_engineering/kinematic_features.py` | `python kinematic_features.py` |
| **Figure 7** | Engineered observables stacked distributions | `04_feature_engineering/plot_engineered_features.py`| `python plot_engineered_features.py` |
| **Figure 8a** | Feature importance for engineered model | `03_inference_and_sculpting/plot_feature_importance.py` | `python plot_feature_importance.py --model engineered_features_pbdt.pkl` |
| **Figure 8b** | ROC curve for engineered feature model | `02_model_training/train_engineered_pbdt.py` | `python train_engineered_pbdt.py` |
| **Figure 9a** | Per-mass ROC curves with inference fix | `05_sensitivity_and_ams/plot_roc_per_mass.py` | `python plot_roc_per_mass.py` |
| **Figure 9b** | Maximum AMS vs $m_a$ (equalized training) | `05_sensitivity_and_ams/calculate_ams_scan.py` | `python calculate_ams_scan.py` |
| **Table 11** | Maximum AMS & optimal BDT cut per mass | `05_sensitivity_and_ams/calculate_ams_scan.py` | `python calculate_ams_scan.py` |

---

## 4. Setup & Installation

### Prerequisites
- Python 3.8+
- Recommended packages:
```bash
pip install -r requirements.txt
```
*(Dependencies: `uproot`, `awkward`, `xgboost`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `scipy`)*

### Data Files
The pipeline automatically searches for `merged_signal_withMET.root` and `merged_bkg_withMET.root` in the local working directory, `data/`, or `../RUN2/`. You can also supply custom paths via `--sig` and `--bkg` flags.

---

## 5. Quick Start: Running the Master Pipeline

To run the complete analysis from start to finish:
```bash
python run_pipeline.py --stage all
```

Or execute individual analysis stages:
```bash
# 1. Dataset & event weights audit (Table 3, 4, 5, Fig 1)
python run_pipeline.py --stage audit

# 2. Train baseline Run 2 pBDT and inspect tree structures (Tab 7, 8, Fig 2, 4)
python run_pipeline.py --stage train --trees 10

# 3. Run per-hypothesis background inference (Sec 6.2)
python run_pipeline.py --stage inference --bdt_cut 0.7

# 4. Generate diphoton mass sculpting plots (Fig 3, 5, Sec 6.2)
python run_pipeline.py --stage sculpting

# 5. Compute Pearson correlation matrices (Fig 6)
python run_pipeline.py --stage correlations

# 6. Reconstruct engineered kinematic observables (Tab 9, Fig 7)
python run_pipeline.py --stage features

# 7. Compute per-mass ROC curves and optimize AMS (Tab 11, Fig 9)
python run_pipeline.py --stage ams
```

---

## 6. Key Methodological Implementations

### A. Signal Mass Equalization Scheme (Section 3.2)
Because detector acceptance and reconstruction efficiencies drop at low masses, raw signal events at $m_a = 12\text{ GeV}$ carry $18\times$ less weight than at $60\text{ GeV}$. The analysis implements:
$$\sum_{i \in m_a} \omega_i^{\text{train,sig}} = \frac{1}{11}, \qquad \omega_i^{\text{train,sig}} = \frac{1}{11 \times N_{m_a}}$$
This forces balanced gradient updates across all mass hypotheses, resulting in a **+122.3% gain in AMS at 12 GeV** and **+90.9% at 15 GeV**.

### B. Negative Weight Rectification (Section 3.1)
Collinear/soft subtraction terms in NLO event generators produce negative weights (e.g. $29.4\%$ in `TTG1Jets`, $36.0\%$ in `DYto2E50`). Negative weights induce negative Hessian sums in XGBoost candidate leaf nodes. Weights are rectified via:
$$\omega_i^{\text{train,bkg}} = |\text{evt\_wgt}_i| \times \frac{\sum_j \text{evt\_wgt}_j}{\sum_j |\text{evt\_wgt}_j|}$$
preserving the total physical cross section while guaranteeing numerical stability during gradient boosting.

### C. Per-Hypothesis Background Inference Fix (Section 6.2)
Legacy inference pipelines evaluated background using a single randomly assigned mass hypothesis. In this suite, every background event is scored across all 11 mass hypotheses:
$$\forall m_a \in \{12, 15, \dots, 60\}\text{ GeV}: \quad \text{Score}_{\text{bkg}}(m_a) = F(\mathbf{x}_{\text{bkg}}, m_a)$$
ensuring mathematically consistent, hypothesis-dependent background rejection.
