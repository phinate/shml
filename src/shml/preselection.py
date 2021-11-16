"""Defines preselections. Currently implements 'loose' and 'tight' strategies."""
from __future__ import annotations

__all__ = ["preselection_mask", "cut_vars"]

from functools import singledispatch

import awkward as ak
import numpy as np
from uproot.behaviors.TTree import TTree


def cut_vars(strength: str) -> list[str]:
    """Returns a list of relevant variables to cut on, depending on cut strength."""
    loose = [
        "HGamEventInfoAuxDyn.yybb_btag77_cutFlow",
        "HGamEventInfoAuxDyn.isPassed",
        "HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.DL1r_FixedCutBEff_85",
        "HGamEventInfoAuxDyn.yybb_candidate_jet1_fix",
    ]

    tight = [
        "HGamEventInfoAuxDyn.yybb_btag77_cutFlow",
        "HGamEventInfoAuxDyn.isPassed",
        "HGamEventInfoAuxDyn.passCrackVetoCleaning",
        "HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.DL1r_FixedCutBEff_77",
        "HGamEventInfoAuxDyn.yybb_candidate_jet1_fix",
        "HGamEventInfoAuxDyn.yybb_candidate_jet2_fix",
    ]

    if strength == "loose":
        return loose
    elif strength == "tight":
        return tight
    elif strength == "all":
        return list(set(loose + tight))
    else:
        raise NotImplementedError(f"no protocol for cut strength '{strength}'")


# "2bjets"
def _mask_tight(
    arrs: ak.Array,
) -> np.ndarray:

    # go from arr[a, b ,c ,...] to arr[[a], [b], [c], ...] (fancy jagged indexing)
    jet1_ind = ak.from_regular(
        arrs["bjet1_idx"][:, np.newaxis],
        axis=1,
    )
    jet2_ind = ak.from_regular(
        arrs["bjet2_idx"][:, np.newaxis],
        axis=1,
    )
    bjet1_req = (
        ak.flatten(
            arrs["HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.DL1r_FixedCutBEff_77"][
                jet1_ind
            ],
        )
        == 1
    )
    bjet2_req = (
        ak.flatten(
            arrs["HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.DL1r_FixedCutBEff_77"][
                jet2_ind
            ],
        )
        == 1
    )

    mask = (
        (arrs["HGamEventInfoAuxDyn.yybb_btag77_cutFlow"] == 6)
        * (arrs["HGamEventInfoAuxDyn.isPassed"] == 1)
        * (arrs["HGamEventInfoAuxDyn.passCrackVetoCleaning"] == 1)
        * bjet1_req
        * bjet2_req
    )
    return mask.to_numpy().astype("bool")


def _from_tree_tight(
    tree: TTree,
) -> np.ndarray:

    arrs = tree.arrays(cut_vars("tight"))

    return _mask_tight(arrs)


# "1bjet_2jets"
def _mask_loose(
    arrs: ak.Array,
) -> np.ndarray:

    # go from arr[a, b ,c ,...] to arr[[a], [b], [c], ...] (fancy jagged indexing)
    jet1_ind = ak.from_regular(
        arrs["bjet1_idx"][:, np.newaxis],
        axis=1,
    )
    bjet_req = (
        ak.flatten(
            arrs["HGamAntiKt4PFlowCustomVtxHggJetsAuxDyn.DL1r_FixedCutBEff_85"][
                jet1_ind
            ],
        )
        == 1
    )
    mask = (
        (arrs["HGamEventInfoAuxDyn.yybb_btag77_cutFlow"] > 3)
        * (arrs["HGamEventInfoAuxDyn.isPassed"] == 1)
        * bjet_req
    )
    return mask.to_numpy().astype("bool")


def _from_tree_loose(
    tree: TTree,
) -> np.ndarray:

    arrs = tree.arrays(cut_vars("loose"))

    return _mask_loose(arrs)


@singledispatch
def preselection_mask(data, strength: str) -> np.ndarray:  # type: ignore
    """Constructs the mask representing a specified preselection strength.
    Usage:
        ```
        # my_data can be an awkward/numpy array or an uproot TTree
        mask = shml.preselection_mask(my_data, 'loose')

        # if my_data is an array:
        cut_data = my_data[mask]

        # or an uproot TTree:
        data = my_data.arrays(list_of_my_variables, my_aliases)
        cut_data = data[mask]
        ```
    """

    raise NotImplementedError(
        f"{type(data)} is not a supported type; "
        + "please provide an uproot tree or awkward array",
    )


@preselection_mask.register
def _from_array(
    data: ak.Array,
    strength: str,
) -> np.ndarray:  # usage: array[preselection.array_mask(array)]
    if strength == "loose":
        return _mask_loose(data)
    elif strength == "tight":
        return _mask_tight(data)
    else:
        raise NotImplementedError(f"no protocol for cut strength '{strength}'")


@preselection_mask.register
def _from_tree(
    data: TTree,
    strength: str,
) -> np.ndarray:  # usage: tree.arrays()[preselection.array_mask(tree)]
    if strength == "loose":
        return _from_tree_loose(data)
    elif strength == "tight":
        return _from_tree_tight(data)
    else:
        raise NotImplementedError(f"no protocol for cut strength '{strength}'")
