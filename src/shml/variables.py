"""Module to return aliases for variables in .root files for use with uproot."""
from __future__ import annotations

__all__ = ["variable_names", "ml_vars"]

from .preselection import cut_vars


def ml_vars() -> list[str]:
    return [
        "photon2_phi",
        "bb_phi",
        "yy_phi",
        "met_phi",
        "bjet1_pt",
        "bjet1_eta",
        "bjet1_bin",
        "bjet1_phi",
        "bjet2_pt",
        "bjet2_eta",
        "bjet2_bin",
        "bjet2_phi",
        "dphi_yy",
        "dphi_bb",
        "dphi_bbyy",
        "deta_bbyy",
        "dR_yy",
        "dR_bb",
        "dR_bbyy",
        "photon1_eta",
        "photon2_eta",
        "bb_eta",
        "yy_eta",
        "bjet1_idx",
        "bjet2_idx",
        "mass_yy",
        "rel_photon1_pt",
        "rel_photon2_pt",
        "met_pt",
        "bb_pt",
        "bb_m",
        "HT",
        "topness",
        "N_j_central",
        "N_j",
        "N_bjets",
        "m_jj_BCal",
        "m_yyjj_BCal",
        "X_mass",
        "S_mass",
        "category",
        "weight",
    ]


def variable_names(cut_strength: str) -> dict[str, str]:
    """Function that returns aliases for analysis .root files for use with uproot,
    depending on desired cut strength. Manual preselections aren't supported."""
    variables = [
        "TotalWeight",
        "HGamEventInfoAuxDyn.m_yy",
        "HGamPhotonsAuxDyn.pt[:, 0]/HGamEventInfoAuxDyn.m_yy",
        "HGamPhotonsAuxDyn.eta[:, 0]",
        "HGamPhotonsAuxDyn.phi[:, 0]",
        "HGamPhotonsAuxDyn.pt[:, 1]/HGamEventInfoAuxDyn.m_yy",
        "HGamPhotonsAuxDyn.eta[:, 1]",
        "HGamPhotonsAuxDyn.phi[:, 1]",
        "HGamEventInfoAuxDyn.met_TST",
        "HGamEventInfoAuxDyn.phi_TST",
        "HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.pt",
        "HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.eta",
        "HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.phi",
        "HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.DL1r_bin",
        "bb_pt",
        "bb_eta",
        "bb_phi",
        "bb_m",
        "HT",
        "topness",
        "yy_phi",
        "yy_eta",
        "yy_pt",
        "HGamEventInfoAuxDyn.N_j_central",
        "HGamEventInfoAuxDyn.N_j",
        "HGamEventInfoAuxDyn.N_j_btag",
        "HGamEventInfoAuxDyn.yybb_candidate_jet1_fix",
        "HGamEventInfoAuxDyn.yybb_candidate_jet2_fix",
        "HGamEventInfoAuxDyn.yybb_BCal_m_jj",
        "HGamEventInfoAuxDyn.yybb_BCal_m_yyjj",
        "EventNumber",
    ]
    var_names = [
        "weight",
        "mass_yy",
        "rel_photon1_pt",
        "photon1_eta",
        "photon1_phi",
        "rel_photon2_pt",
        "photon2_eta",
        "photon2_phi",
        "met_pt",
        "met_phi",
        "bjet_pt",
        "bjet_eta",
        "bjet_phi",
        "bjet_bin",
        "bb_pt",
        "bb_eta",
        "bb_phi",
        "bb_m",
        "HT",
        "topness",
        "yy_phi",
        "yy_eta",
        "yy_pt",
        "N_j_central",
        "N_j",
        "N_bjets",
        "bjet1_idx",
        "bjet2_idx",
        "m_jj_BCal",
        "m_yyjj_BCal",
        "EventNumber",
    ]
    preprocess_dict = {
        "photon1_eta": "HGamPhotonsAuxDyn.eta[:, 0]",
        "photon1_phi": "HGamPhotonsAuxDyn.phi[:, 0]",
        "photon2_eta": "HGamPhotonsAuxDyn.eta[:, 1]",
        "photon2_phi": "HGamPhotonsAuxDyn.phi[:, 1]",
        "bjet_pt": "HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.pt",
        "bjet_eta": "HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.eta",
        "bjet_phi": "HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.phi",
        "bjet_bin": "HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.DL1r_bin",
        "bb_eta": "bb_eta",
        "bb_phi": "bb_phi",
        "yy_phi": "yy_phi",
        "yy_eta": "yy_eta",
        "bjet1_idx": "HGamEventInfoAuxDyn.yybb_candidate_jet1_fix",
        "bjet2_idx": "HGamEventInfoAuxDyn.yybb_candidate_jet2_fix",
    }

    all_cut_vars = cut_vars(cut_strength)

    cut_vars_dict = {v: v for v in all_cut_vars if v not in preprocess_dict.values()}

    most_vars = {**preprocess_dict, **cut_vars_dict}
    other_vars = {
        k: v
        for k, v in zip(
            var_names,
            variables,
        )
        if k not in most_vars.keys()
    }
    return {**most_vars, **other_vars}
