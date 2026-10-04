"""
kinematic_features.py
=====================
Kinematic feature engineering library for WH -> l nu a a -> l nu b b~ gamma gamma.

Corresponds to Semester Report:
- Section 7.1: Constructed Kinematic Observables (Table 9)

Derives all 24 engineered features:
1.  alpha_gg: 3D opening angle between photons
2.  dR_gg: Delta-R between leading and subleading photons
3.  dR_bb: Delta-R between leading and subleading b-jets
4.  dR_bb_gg: Delta-R between bb system and gamma-gamma system
5.  dR_l_bnearest: Delta-R between lepton and nearest b-jet
6.  pt_asym_gg: Transverse momentum asymmetry between photons
7.  pt_asym_bb: Transverse momentum asymmetry between b-jets
8.  pt_asym_bbgg: Transverse momentum asymmetry between bb and gamma-gamma systems
9.  pt_diff_gg: |pT(gamma1) - pT(gamma2)|
10. pt_diff_bb: |pT(b1) - pT(b2)|
11. pt_diff_bbgg: |pT(bb) - pT(gamma-gamma)|
12. pt_gg: Transverse momentum of gamma-gamma system
13. pt_bb: Transverse momentum of bb system
14. pt_bbgg: Combined transverse momentum of 4-body bb-gamma-gamma system
15. pt_ratio_gg_bb: pT(gamma-gamma) / pT(bb)
16. pt_ratio_bbgg_sum: pT(bb-gamma-gamma) / (pT(bb) + pT(gamma-gamma))
17. M_bb: Dijet invariant mass
18. M_gg: Diphoton invariant mass (diagnostic only, excluded from BDT inputs)
19. M_bbgg: Four-body invariant mass m(bb-gamma-gamma)
20. abs_mbb_minus_mgg: |m(bb) - m(gamma-gamma)| (diagnostic only, excluded from BDT inputs)
21. dphi_l_MET: Delta-phi between lepton and MET
22. mT_l_MET: Lepton-MET transverse mass sqrt(2*pT(l)*MET*(1-cos(dphi)))
23. met_significance: MET / sqrt(sumET)
24. met_pf_puppi_pull: |PF-MET - PUPPI-MET| / mean(PF-MET, PUPPI-MET)
"""

import numpy as np
import awkward as ak


def four_vector(pt, eta, phi, mass):
    """Computes (px, py, pz, E) from (pt, eta, phi, mass)."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    p2 = px**2 + py**2 + pz**2
    e = np.sqrt(p2 + mass**2)
    return px, py, pz, e


def invariant_mass(components_list):
    """Computes invariant mass for a sum of (px, py, pz, E) four-vectors."""
    px = sum(c[0] for c in components_list)
    py = sum(c[1] for c in components_list)
    pz = sum(c[2] for c in components_list)
    e = sum(c[3] for c in components_list)
    m2 = e**2 - px**2 - py**2 - pz**2
    m2 = ak.where(m2 < 0.0, 0.0, m2)
    return np.sqrt(m2)


def delta_phi(phi1, phi2):
    """Computes difference in azimuthal angle folded into [-pi, pi]."""
    dphi = phi1 - phi2
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def delta_r(eta1, phi1, eta2, phi2):
    """Computes Delta-R between two direction vectors."""
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


def derive_all_features(arr):
    """
    Computes all 24 engineered kinematic variables from an awkward array
    containing reconstructed photons, jets, leptons, and MET branches.
    """
    # 1. Photons kinematics
    pt1, eta1, phi1 = arr["pholead_pt"], arr["pholead_eta"], arr["pholead_phi"]
    pt2, eta2, phi2 = arr["phosublead_pt"], arr["phosublead_eta"], arr["phosublead_phi"]
    m_g1 = arr["pholead_mass"] if "pholead_mass" in arr.fields else 0.0
    m_g2 = arr["phosublead_mass"] if "phosublead_mass" in arr.fields else 0.0

    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    p1 = pt1 * np.cosh(eta1)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    p2 = pt2 * np.cosh(eta2)

    # 3D photon opening angle (alpha_gg)
    cos_alpha = (px1 * px2 + py1 * py2 + pz1 * pz2) / (p1 * p2 + 1e-6)
    cos_alpha = ak.where(cos_alpha > 1.0, 1.0, cos_alpha)
    cos_alpha = ak.where(cos_alpha < -1.0, -1.0, cos_alpha)
    arr["alpha_gg"] = np.arccos(cos_alpha)

    # Photon pT asymmetries & differences
    arr["pt_asym_gg"] = (pt1 - pt2) / (pt1 + pt2 + 1e-6)
    arr["pt_diff_gg"] = np.abs(pt1 - pt2)
    arr["dR_gg"] = delta_r(eta1, phi1, eta2, phi2)

    # 2. b-jets kinematics
    pt_b1, eta_b1, phi_b1 = arr["first_jet_pt"], arr["first_jet_eta"], arr["first_jet_phi"]
    pt_b2, eta_b2, phi_b2 = arr["second_jet_pt"], arr["second_jet_eta"], arr["second_jet_phi"]
    m_b1 = arr["first_jet_mass"] if "first_jet_mass" in arr.fields else 0.0
    m_b2 = arr["second_jet_mass"] if "second_jet_mass" in arr.fields else 0.0

    arr["pt_asym_bb"] = (pt_b1 - pt_b2) / (pt_b1 + pt_b2 + 1e-6)
    arr["pt_diff_bb"] = np.abs(pt_b1 - pt_b2)
    arr["dR_bb"] = delta_r(eta_b1, phi_b1, eta_b2, phi_b2)

    # 3. bb system & gamma-gamma system vector sums
    px_bb = pt_b1 * np.cos(phi_b1) + pt_b2 * np.cos(phi_b2)
    py_bb = pt_b1 * np.sin(phi_b1) + pt_b2 * np.sin(phi_b2)
    pt_bb = np.sqrt(px_bb**2 + py_bb**2)

    px_gg = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
    py_gg = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
    pt_gg = np.sqrt(px_gg**2 + py_gg**2)

    arr["pt_bb"] = pt_bb
    arr["pt_gg"] = pt_gg
    arr["pt_bbgg"] = pt_bb + pt_gg
    arr["pt_diff_bbgg"] = np.abs(pt_bb - pt_gg)
    arr["pt_asym_bbgg"] = (pt_bb - pt_gg) / (pt_bb + pt_gg + 1e-6)

    arr["pt_ratio_gg_bb"] = pt_gg / (pt_bb + 1e-6)
    arr["pt_ratio_bbgg_sum"] = (pt_bb + pt_gg) / (pt_bb + pt_gg + 1e-6)

    # Invariant masses & 4-vectors
    v_g1 = four_vector(pt1, eta1, phi1, m_g1)
    v_g2 = four_vector(pt2, eta2, phi2, m_g2)
    v_b1 = four_vector(pt_b1, eta_b1, phi_b1, m_b1)
    v_b2 = four_vector(pt_b2, eta_b2, phi_b2, m_b2)

    arr["M_gg"] = invariant_mass([v_g1, v_g2])
    arr["M_bb"] = invariant_mass([v_b1, v_b2])
    arr["M_bbgg"] = invariant_mass([v_g1, v_g2, v_b1, v_b2])
    arr["abs_mbb_minus_mgg"] = np.abs(arr["M_bb"] - arr["M_gg"])

    # Mass pulls relative to hypothesis
    if "mass_point" in arr.fields:
        arr["mass_pull_bb"] = np.abs(arr["M_bb"] - arr["mass_point"]) / (arr["mass_point"] + 1e-6)
        arr["mass_pull_gg"] = np.abs(arr["M_gg"] - arr["mass_point"]) / (arr["mass_point"] + 1e-6)

    # 4. Angular separation dR(bb, gg)
    phi_bb = np.arctan2(py_bb, px_bb)
    phi_gg = np.arctan2(py_gg, px_gg)
    eta_bb = (eta_b1 + eta_b2) / 2.0
    eta_gg = (eta1 + eta2) / 2.0
    arr["dR_bb_gg"] = delta_r(eta_bb, phi_bb, eta_gg, phi_gg)

    # 5. Lepton and MET Observables
    lep_pt = arr["leppt"] if "leppt" in arr.fields else arr["electron_pt"]
    lep_eta = arr["lepeta"] if "lepeta" in arr.fields else arr["electron_eta"]
    lep_phi = ak.where(arr["electron_pt"] > 0, arr["electron_phi"], arr["muon_phi"])

    # dR between lepton and nearest b-jet
    dr_lb1 = delta_r(lep_eta, lep_phi, eta_b1, phi_b1)
    dr_lb2 = delta_r(lep_eta, lep_phi, eta_b2, phi_b2)
    arr["dR_l_bnearest"] = np.minimum(dr_lb1, dr_lb2)

    # Delta-phi(lepton, MET) and Transverse Mass mT(l, MET)
    met_pt = arr["PFMET_pt"]
    met_phi = arr["PFMET_phi"]
    dphi_l_met = delta_phi(lep_phi, met_phi)
    arr["dphi_l_MET"] = dphi_l_met

    mt2 = 2.0 * lep_pt * met_pt * (1.0 - np.cos(dphi_l_met))
    mt2 = ak.where(mt2 < 0.0, 0.0, mt2)
    arr["mT_l_MET"] = np.sqrt(mt2)

    # MET significance
    sum_et = arr["PFMET_sumEt"] if "PFMET_sumEt" in arr.fields else 1.0
    arr["met_significance"] = met_pt / np.sqrt(np.maximum(sum_et, 1.0))

    # Pull between PF-MET and PUPPI-MET
    puppi_pt = arr["PuppiMET_pt"] if "PuppiMET_pt" in arr.fields else met_pt
    mean_met = (met_pt + puppi_pt) / 2.0 + 1e-6
    arr["met_pf_puppi_pull"] = np.abs(met_pt - puppi_pt) / mean_met

    return arr
