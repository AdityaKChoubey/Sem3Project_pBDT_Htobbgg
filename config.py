"""
config.py — Single source of truth for the HtoAATo2b2g BDT pipeline.

All scripts import from here so that variable lists, binning, physics
normalisation, and weight column names remain consistent across:

    - BDT training
    - BDT inference
    - input-variable plotting
    - mass sculpting studies
    - per-mass-point studies
    - AMS optimisation

INPUT_VARS below is kept in lock-step with the standalone training script
(xgboost_TrainMVA_v2.py). That script owns the ground truth for variable
names / order; this file mirrors it so every downstream script (inference,
plotting) automatically stays consistent with whatever model was trained.

Weight philosophy
─────────────────

TRAINING
    Signal:
        The original evt_wgt is replaced by a flat per-event weight for each
        mass point. Every signal mass point contributes the same total weight:

            1 / N_mass_points

        Therefore the total signal training weight summed over all mass points
        is exactly 1.

    Background:
        Negative evt_wgt values are converted to absolute values and globally
        rescaled so that the total background weight remains equal to the
        original signed sum.

        This reproduces the weighting used in xgboost_TrainMVA_v2.py.

PLOTTING / PHYSICS YIELDS
    Uses the raw 'weight' branch from the ROOT file.

    Physics event yield:

        weight × cross section [pb] × luminosity [fb^-1] × 1000

    because:

        1 pb × 1 fb^-1 = 1000 events

    Background cross section is selected using process_id.

    Signal is assumed to have:

        sigma(signal) = 1% × sigma(WH)

    and the same signal cross section is used for every mass hypothesis.
"""

import numpy as np


# =============================================================================
# Input variables
# =============================================================================

# Variables follow the BDT training-variable definition in
# xgboost_TrainMVA_v2.py (inputVars / inputVarsMC). Keep this list, and its
# order, identical to that script's — it is the authoritative source.
#
# Lepton
#   pT
#   eta
#
# b-jets
#   eta of leading and subleading b-jets
#   pT(b1) / Mbb
#   pT(b2) / Mbb
#   b-tag probability (DeepJet prob-b) of leading and subleading b-jets
#
# Event
#   Number of AK4 jets
#
# Photons
#   eta of leading and subleading photons
#   pT(gamma1) / Mgg
#   pT(gamma2) / Mgg
#   photon MVA ID of leading and subleading photons
#
# Multi-object variables
#   DeltaPhi(gamma, gamma)
#   DeltaPhi(b, b)
#   DeltaPhi(bb, gammagamma)
#   DeltaPhi(gammagamma, MET)
#
# Parametric input
#   mass_point


INPUT_VARS = [
    #Lepton
    "leppt",
    "lepeta",

    #b-jets
    "first_jet_eta",
    "second_jet_eta",
    "pT1_by_mbb",
    "pT2_by_mbb",
    "first_jet_probb",
    "second_jet_probb",

    #AK4 jet multiplicity
    "Njets",

    #Photons
    "pholead_eta",
    "phosublead_eta",
    "pT1_by_mgg",
    "pT2_by_mgg",
    "pholead_mvaID",
    "phosublead_mvaID",

    #Multi-object variables
    "delphi_gg",
    "delphi_bb",
    "delphi_bbgg",
    "delphi_ggMET_PF",

    #Parametric mass hypothesis
    "mass_point",
]


# =============================================================================
# ROOT branches
# =============================================================================

# Training weight branch
TRAIN_WEIGHT_BRANCH = "evt_wgt"

# Raw physics event weight branch
PLOT_WEIGHT_BRANCH = "weight"

# Backward compatibility with older training scripts
WEIGHT_BRANCH = TRAIN_WEIGHT_BRANCH

# Diphoton invariant mass
MASS_BRANCH = "CMS_hgg_mass"


# Branches required during training
LOAD_VARS = (
    INPUT_VARS
    + [
        MASS_BRANCH,
        TRAIN_WEIGHT_BRANCH,
    ]
)


# Branches required for signal physics plotting
LOAD_VARS_PLOT_SIG = (
    INPUT_VARS
    + [
        MASS_BRANCH,
        PLOT_WEIGHT_BRANCH,
    ]
)


# Branches required for background physics plotting
LOAD_VARS_PLOT_BKG = (
    INPUT_VARS
    + [
        MASS_BRANCH,
        PLOT_WEIGHT_BRANCH,
        "process_id",
    ]
)


# =============================================================================
# Histogram binning
# =============================================================================

# BDT score histograms
N_BINS_BDT = 25

# Input-feature distributions
N_BINS_FEATURE = 50

# Diphoton mass distributions
N_BINS_MGG = 180


# =============================================================================
# Plot ranges
# =============================================================================

BDT_RANGE = (0.0, 1.0)

# Diphoton invariant-mass window [GeV]
MGG_RANGE = (0.0, 180.0)

# Delta-phi is defined in [0, pi]
DELPHI_RANGE = (0.0, np.pi)


# =============================================================================
# Signal mass hypotheses
# =============================================================================

MASS_POINTS = [
    12,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
]


# =============================================================================
# Signal training weights
# =============================================================================

# Every mass point contributes the same total training weight.
#
# With 11 mass points:
#
#       weight sum per mass point = 1 / 11
#
# Therefore:
#
#       total signal training weight = 1

SIG_WEIGHT_PER_MASS_POINT = 1.0 / len(MASS_POINTS)


def get_signal_weights_per_mass_point(
    mass_point_array,
    mass_points=None,
    target_sum=None,
):
    """
    Recreate the per-event signal weights used during BDT training.

    This must reproduce:

        normaliseSignalWeightsPerMassPoint()

    from xgboost_TrainMVA_v2.py.

    For every mass point m:

        event_weight = target_sum / N_m

    where N_m is the number of signal events at mass point m.

    Therefore:

        sum(weights for mass point m) = target_sum

    By default:

        target_sum = 1 / len(MASS_POINTS)

    and therefore the total signal training weight is exactly 1.

    IMPORTANT
    ---------
    These are TRAINING weights.

    Do not use this function for physics-yield plots.
    Use get_physics_weights_signal() instead.
    """

    if mass_points is None:
        mass_points = MASS_POINTS

    if target_sum is None:
        target_sum = SIG_WEIGHT_PER_MASS_POINT

    mass_point_array = np.asarray(mass_point_array)

    weights = np.zeros(
        len(mass_point_array),
        dtype=float,
    )

    for mass_point in mass_points:

        mask = mass_point_array == mass_point

        n_events = np.count_nonzero(mask)

        if n_events == 0:
            print(
                "  [get_signal_weights_per_mass_point] WARNING: "
                f"no signal events for mass_point={mass_point}"
            )
            continue

        weights[mask] = target_sum / n_events

    return weights


# =============================================================================
# Background training weights
# =============================================================================


def get_background_training_weights(evt_wgt_array):
    """
    Recreate the background weights used during BDT training.

    The training procedure is:

        original_sum = sum(evt_wgt)

        abs_weight = abs(evt_wgt)

        abs_sum = sum(abs_weight)

        scale = original_sum / abs_sum

        new_weight = abs_weight * scale

    Therefore:

        sum(new_weight) == sum(original evt_wgt)

    while every individual training weight is non-negative.

    This reproduces the negative-weight handling in
    xgboost_TrainMVA_v2.py.

    IMPORTANT
    ---------
    These are TRAINING weights.

    Do not use this function for physics-yield plots.
    Use get_physics_weights_background() instead.
    """

    evt_wgt_array = np.asarray(
        evt_wgt_array,
        dtype=float,
    )

    original_sum = evt_wgt_array.sum()

    abs_weight = np.abs(evt_wgt_array)

    abs_sum = abs_weight.sum()

    if abs_sum == 0:
        print(
            "  [get_background_training_weights] WARNING: "
            "sum(|evt_wgt|) == 0"
        )

        return abs_weight

    scale = original_sum / abs_sum

    new_weight = abs_weight * scale

    return new_weight


# =============================================================================
# Integrated luminosity
# =============================================================================

# Run-3 target integrated luminosity
#
# Units: fb^-1

LUMI = 109.0


# =============================================================================
# Background process definitions
# =============================================================================

# process_id mapping:
#
#   0   TTG1Jets
#   1   TTto2L2Nu
#   2   TTtoLNu2Q
#   3   WGtoLNuG
#   4   DYto2Mu50
#   5   DYto2E50


PROCESS_MAP = {
    "TTG1Jets": 0,
    "TTto2L2Nu": 1,
    "TTtoLNu2Q": 2,
    "WGtoLNuG": 3,
    "DYto2Mu50": 4,
    "DYto2E50": 5,
}


PROCESS_NAMES = {
    process_id: process_name
    for process_name, process_id in PROCESS_MAP.items()
}


# =============================================================================
# Background cross sections
# =============================================================================

# Cross sections in pb

XSEC_BKG = {
    0: 4.634,       # TTG1Jets
    1: 98.04,       # TTto2L2Nu
    2: 405.87,      # TTtoLNu2Q
    3: 671.5,       # WGtoLNuG
    4: 2124.08,     # DYto2Mu50
    5: 2124.08,     # DYto2E50
}


# =============================================================================
# Signal cross section
# =============================================================================

# Total WH production cross section [pb].
#
# IMPORTANT:
# Keep this value consistent with the cross-section source used by the
# analysis. The previous config contained a mismatch between the numerical
# value and the explanatory comment.
#
# Using:
#
#       sigma(WH) = 1.458 pb
#
# and assuming:
#
#       BR / signal fraction = 1%
#
# gives:
#
#       sigma(signal) = 0.01 × 1.458 pb
#                     = 0.01458 pb


WH_XSEC_TOTAL = 1.458

SIG_XSEC_FRAC = 0.01

XSEC_SIG = WH_XSEC_TOTAL * SIG_XSEC_FRAC


# =============================================================================
# Signal physics weights
# =============================================================================


def get_physics_weights_signal(weight_branch_array):
    """
    Compute physics plotting weights for signal.

    For every signal event:

        physics_weight
            = weight
            × XSEC_SIG [pb]
            × LUMI [fb^-1]
            × 1000

    Unit conversion:

        1 pb × 1 fb^-1 = 1000

    Therefore the resulting weighted sum corresponds to the expected
    event yield.

    The same XSEC_SIG is currently applied to every mass point.
    """

    weight_array = np.asarray(
        weight_branch_array,
        dtype=float,
    )

    return (
        weight_array
        * XSEC_SIG
        * LUMI
        * 1000.0
    )


# =============================================================================
# Background physics weights
# =============================================================================


def get_physics_weights_background(
    weight_branch_array,
    process_id_array,
):
    """
    Compute physics plotting weights for background.

    For every event:

        physics_weight
            = weight
            × XSEC_BKG[process_id] [pb]
            × LUMI [fb^-1]
            × 1000

    The process_id branch determines which production cross section is
    assigned to each event.
    """

    weight_array = np.asarray(
        weight_branch_array,
        dtype=float,
    )

    process_id_array = np.asarray(
        process_id_array,
        dtype=int,
    )

    output_weights = np.zeros(
        len(weight_array),
        dtype=float,
    )

    known_process_ids = set(XSEC_BKG.keys())

    unique_process_ids = set(
        np.unique(process_id_array).tolist()
    )

    unknown_process_ids = (
        unique_process_ids - known_process_ids
    )

    if unknown_process_ids:
        print(
            "  [get_physics_weights_background] WARNING: "
            "unknown process_id values found: "
            f"{sorted(unknown_process_ids)}"
        )

    for process_id, cross_section in XSEC_BKG.items():

        mask = process_id_array == process_id

        n_events = np.count_nonzero(mask)

        if n_events == 0:
            print(
                "  [get_physics_weights_background] WARNING: "
                f"no background events for process_id={process_id}"
            )
            continue

        output_weights[mask] = (
            weight_array[mask]
            * cross_section
            * LUMI
            * 1000.0
        )

        process_name = PROCESS_NAMES.get(
            process_id,
            f"process_{process_id}",
        )

        print(
            "  [get_physics_weights_background] "
            f"process_id={process_id}  "
            f"process={process_name:<12}  "
            f"n_events={n_events:>8}  "
            f"xsec={cross_section:.6g} pb  "
            f"weight_sum={output_weights[mask].sum():.6f}"
        )

    return output_weights


# =============================================================================
# Human-readable feature labels
# =============================================================================

FEATURE_LABELS = {
    # ── Lepton ───────────────────────────────────────────────────────────────
    "leppt":
        r"Lepton $p_T$ [GeV]",

    "lepeta":
        r"Lepton $\eta$",

    # ── b-jets ───────────────────────────────────────────────────────────────
    "first_jet_eta":
        r"Leading b-jet $\eta$",

    "second_jet_eta":
        r"Subleading b-jet $\eta$",

    "pT1_by_mbb":
        r"$p_T^{b_1}/M_{bb}$",

    "pT2_by_mbb":
        r"$p_T^{b_2}/M_{bb}$",

    "first_jet_probb":
        r"Leading b-jet DeepJet prob-b",

    "second_jet_probb":
        r"Subleading b-jet DeepJet prob-b",

    # ── AK4 jets ─────────────────────────────────────────────────────────────
    "Njets":
        r"Number of AK4 jets",

    # ── Photons ──────────────────────────────────────────────────────────────
    "pholead_eta":
        r"Leading photon $\eta$",

    "phosublead_eta":
        r"Subleading photon $\eta$",

    "pT1_by_mgg":
        r"$p_T^{\gamma_1}/M_{\gamma\gamma}$",

    "pT2_by_mgg":
        r"$p_T^{\gamma_2}/M_{\gamma\gamma}$",

    "pholead_mvaID":
        r"Leading photon MVA ID",

    "phosublead_mvaID":
        r"Subleading photon MVA ID",

    # ── Multi-object variables ────────────────────────────────────────────────
    "delphi_gg":
        r"$\Delta\phi_{\gamma\gamma}$",

    "delphi_bb":
        r"$\Delta\phi_{bb}$",

    "delphi_bbgg":
        r"$\Delta\phi_{bb,\gamma\gamma}$",

    "delphi_ggMET_PF":
        r"$\Delta\phi_{\gamma\gamma,MET}$",

    # ── Parametric mass ──────────────────────────────────────────────────────
    "mass_point":
        r"Mass hypothesis $m_a$ [GeV]",
}


# =============================================================================
# Variable-level plot settings
# =============================================================================

# Maps:
#
#       variable -> (x_range, log_y)
#
# x_range = None
#       Plotting script determines the range automatically.
#
# log_y = True
#       Use logarithmic y-axis.


PLOT_SETTINGS = {
    # ── Lepton ───────────────────────────────────────────────────────────────
    "leppt": (
        (0.0, 250.0),
        True,
    ),

    "lepeta": (
        (-3.0, 3.0),
        False,
    ),

    # ── b-jets ───────────────────────────────────────────────────────────────
    "first_jet_eta": (
        (-3.0, 3.0),
        False,
    ),

    "second_jet_eta": (
        (-3.0, 3.0),
        False,
    ),

    "pT1_by_mbb": (
        (0.0, 5.0),
        True,
    ),

    "pT2_by_mbb": (
        (0.0, 5.0),
        True,
    ),

    "first_jet_probb": (
        (0.0, 1.0),
        False,
    ),

    "second_jet_probb": (
        (0.0, 1.0),
        False,
    ),

    # ── AK4 jet multiplicity ─────────────────────────────────────────────────
    "Njets": (
        (0.0, 10.0),
        False,
    ),

    # ── Photons ──────────────────────────────────────────────────────────────
    "pholead_eta": (
        (-3.0, 3.0),
        False,
    ),

    "phosublead_eta": (
        (-3.0, 3.0),
        False,
    ),

    "pT1_by_mgg": (
        (0.0, 5.0),
        True,
    ),

    "pT2_by_mgg": (
        (0.0, 5.0),
        True,
    ),

    "pholead_mvaID": (
        (-1.0, 1.0),
        False,
    ),

    "phosublead_mvaID": (
        (-1.0, 1.0),
        False,
    ),

    # ── Multi-object variables ────────────────────────────────────────────────
    "delphi_gg": (
        DELPHI_RANGE,
        False,
    ),

    "delphi_bb": (
        DELPHI_RANGE,
        False,
    ),

    "delphi_bbgg": (
        DELPHI_RANGE,
        False,
    ),

    "delphi_ggMET_PF": (
        DELPHI_RANGE,
        False,
    ),

    # ── Parametric mass ──────────────────────────────────────────────────────
    "mass_point": (
        None,
        False,
    ),
}


# =============================================================================
# Configuration sanity checks
# =============================================================================


def validate_config():
    """
    Perform lightweight consistency checks on the configuration.

    Raises ValueError if duplicate variables or missing plot labels/settings
    are found.
    """

    # Check duplicate input variables
    duplicate_vars = {
        variable
        for variable in INPUT_VARS
        if INPUT_VARS.count(variable) > 1
    }

    if duplicate_vars:
        raise ValueError(
            "Duplicate INPUT_VARS found: "
            f"{sorted(duplicate_vars)}"
        )

    # Check labels
    missing_labels = [
        variable
        for variable in INPUT_VARS
        if variable not in FEATURE_LABELS
    ]

    if missing_labels:
        raise ValueError(
            "Missing FEATURE_LABELS entries for: "
            f"{missing_labels}"
        )

    # Check plotting settings
    missing_plot_settings = [
        variable
        for variable in INPUT_VARS
        if variable not in PLOT_SETTINGS
    ]

    if missing_plot_settings:
        raise ValueError(
            "Missing PLOT_SETTINGS entries for: "
            f"{missing_plot_settings}"
        )

    # Check cross-section definitions
    missing_xsecs = [
        process_id
        for process_id in PROCESS_NAMES
        if process_id not in XSEC_BKG
    ]

    if missing_xsecs:
        raise ValueError(
            "Missing XSEC_BKG entries for process IDs: "
            f"{missing_xsecs}"
        )

    print(
        "[config] Configuration validation successful"
    )

    print(
        f"[config] Number of BDT input variables: "
        f"{len(INPUT_VARS)}"
    )

    print(
        f"[config] Number of signal mass points: "
        f"{len(MASS_POINTS)}"
    )

    print(
        f"[config] Signal training weight per mass point: "
        f"{SIG_WEIGHT_PER_MASS_POINT:.8f}"
    )

    print(
        f"[config] WH cross section: "
        f"{WH_XSEC_TOTAL:.6f} pb"
    )

    print(
        f"[config] Signal cross section: "
        f"{XSEC_SIG:.6f} pb"
    )

    print(
        f"[config] Integrated luminosity: "
        f"{LUMI:.3f} fb^-1"
    )


# Run configuration validation when this file is executed directly.
#
# Example:
#
#       python config.py

if __name__ == "__main__":
    validate_config()


# =============================================================================
# Engineered Kinematic Variables (Semester Report Section 7 / Table 9)
# =============================================================================

ENGINEERED_VARS_ALL = [
    'alpha_gg', 'dR_gg', 'dR_bb', 'dR_bb_gg', 'dR_l_bnearest',
    'pt_asym_gg', 'pt_asym_bb', 'pt_asym_bbgg',
    'pt_diff_gg', 'pt_diff_bb', 'pt_diff_bbgg',
    'pt_gg', 'pt_bb', 'pt_bbgg',
    'pt_ratio_gg_bb', 'pt_ratio_bbgg_sum',
    'M_bb', 'M_gg', 'M_bbgg', 'abs_mbb_minus_mgg',
    'dphi_l_MET', 'mT_l_MET', 'met_significance', 'met_pf_puppi_pull',
    'mass_pull_bb', 'mass_pull_gg'
]

# Variables fed into BDT training (excludes direct m_gg correlates M_gg and abs_mbb_minus_mgg)
ENGINEERED_BDT_FEATURES = [
    'alpha_gg', 'pt_asym_gg', 'pt_diff_gg', 'pt_asym_bb', 'pt_diff_bb',
    'pt_bb', 'pt_gg', 'pt_bbgg', 'pt_diff_bbgg', 'pt_asym_bbgg',
    'pt_ratio_gg_bb', 'pt_ratio_bbgg_sum', 'dphi_l_MET', 'M_bbgg',
    'mT_l_MET', 'met_significance', 'met_pf_puppi_pull',
    'mass_pull_bb', 'mass_pull_gg', 'dR_bb', 'dR_gg', 'dR_bb_gg', 'dR_l_bnearest',
    'mass_point'
]

def find_data_file(filename, search_dirs=None):
    """
    Searches for a data file across standard project directories.
    """
    import os
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    
    if search_dirs is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        search_dirs = [
            os.getcwd(),
            this_dir,
            os.path.join(this_dir, '..', 'RUN2'),
            os.path.join(this_dir, '..', 'data'),
            os.path.join(this_dir, 'data'),
            os.path.join(os.getcwd(), 'RUN2'),
            r'c:\Users\Aditya Kr Choubey\Downloads\RUN2\RUN2'
        ]
    
    for d in search_dirs:
        cand = os.path.normpath(os.path.join(d, filename))
        if os.path.exists(cand):
            return cand
            
    return filename
