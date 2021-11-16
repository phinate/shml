"""Routine that applies a preselection and constructs useful quantities from data."""
from __future__ import annotations

__all__ = ["preprocess"]

from typing import Callable

import awkward as ak
import numpy as np

from .kinematics import bound_phi, delta_R
from .preselection import preselection_mask


def preprocess(
    arr: ak.Array,
    # user can provide own expression
    preselection: Callable[[ak.Array], ak.Array] | str,
    filename: str,
    blind: bool = False,
) -> ak.Array:
    """Applies preselection, and builds useful variables, such as:
    - delta_R
    - delta_phi
    - delta_eta
    - S mass, X mass (if background, both = 0. Will be modified during training.)
    - signal/background category (1 or 0)

    Returns unmodified entries in addition to augmented entries."""

    # apply preselection
    if callable(preselection):
        mask = preselection(arr).to_numpy().astype("bool")
    elif isinstance(preselection, str):  # 'loose' or 'tight' for now
        mask = preselection_mask(arr, preselection)
    else:
        raise ValueError(
            f"incorrect type {type(preselection)} provided for preselection",
        )
    cutarr = arr[mask]

    # construct high-level features
    # first, construct masks for the first 2 bjet candidates
    bjet1_idx = ak.from_regular(
        cutarr["bjet1_idx"][:, np.newaxis],
        axis=1,
    )
    bjet2_idx = ak.from_regular(
        cutarr["bjet2_idx"][:, np.newaxis],
        axis=1,
    )
    # use these to collect individual bjet info
    bjets = cutarr[["bjet_pt", "bjet_eta", "bjet_phi", "bjet_bin"]]
    bjet1 = ak.Array(
        {
            k: v
            for k, v in zip(
                ["pt", "eta", "phi", "bin"],
                [ak.flatten(bjets[bjet1_idx][f]) for f in bjets.fields],
            )
        },
    )
    bjet2 = ak.Array(
        {
            k: v
            for k, v in zip(
                ["pt", "eta", "phi", "bin"],
                [ak.flatten(bjets[bjet2_idx][f]) for f in bjets.fields],
            )
        },
    )
    dphi_yy = bound_phi(cutarr["photon2_phi"] - cutarr["photon1_phi"])
    dphi_bb = bound_phi(bjet2["phi"] - bjet1["phi"])
    dphi_bbyy = bound_phi(cutarr["bb_phi"] - cutarr["yy_phi"])

    deta_bbyy = cutarr["bb_eta"] - cutarr["yy_eta"]

    dR_yy = delta_R(
        dphi=dphi_yy,
        deta=cutarr["photon2_eta"] - cutarr["photon1_eta"],
    )
    dR_bb = delta_R(dphi=dphi_bb, deta=bjet2["eta"] - bjet1["eta"])
    dR_bbyy = delta_R(dphi=dphi_bbyy, deta=deta_bbyy)

    # rotate all phi angles relative to photon 1 phi (which becomes the 0-point)
    # bjet_x variables are placeholders to extract the 2 bjets, so add them separately
    phi_fields = [
        field for field in cutarr.fields if "phi" in field and "bjet" not in field
    ]

    # set up dictionaries to export to awkward
    phis_relative_to_y1 = {
        k: v
        for k, v in zip(
            phi_fields,
            [bound_phi(cutarr[field] - cutarr["photon1_phi"]) for field in phi_fields],
        )
    }

    bjet1_dict = {
        k: v
        for k, v in zip(
            ["bjet1_pt", "bjet1_eta", "bjet1_bin", "bjet1_phi"],
            [
                bjet1["pt"],
                bjet1["eta"],
                bjet1["bin"],
                bound_phi(
                    bjet1["phi"] - cutarr["photon1_phi"],
                ),
            ],
        )
    }
    bjet2_dict = {
        k: v
        for k, v in zip(
            ["bjet2_pt", "bjet2_eta", "bjet2_bin", "bjet2_phi"],
            [
                bjet2["pt"],
                bjet2["eta"],
                bjet2["bin"],
                bound_phi(
                    bjet2["phi"] - cutarr["photon1_phi"],
                ),
            ],
        )
    }
    aux_var_dict = dict(
        dphi_yy=dphi_yy,
        dphi_bb=dphi_bb,
        dphi_bbyy=dphi_bbyy,
        deta_bbyy=deta_bbyy,
        dR_yy=dR_yy,
        dR_bb=dR_bb,
        dR_bbyy=dR_bbyy,
    )

    big_dict = {
        **phis_relative_to_y1,
        **bjet1_dict,
        **bjet2_dict,
        **aux_var_dict,
    }

    # handle unprocessed elements while we're here
    leftovers = tuple(
        filter(
            lambda item: item not in big_dict.keys() and "bjet_" not in item,
            cutarr.fields,
        ),
    )
    for field in leftovers:
        big_dict[field] = cutarr[field]

    # convention for signal filenames (X{X_mass}_S{S_mass}), could be stricter regex :p
    if filename[0] == "X":  # signal
        name = filename.replace(".root", "")
        x_idx = name.find("X")
        s_idx = name.find("S")

        big_dict["X_mass"] = ak.Array(
            np.zeros(len(cutarr)) + int(name[x_idx + 1 : s_idx - 1]),
        )
        big_dict["S_mass"] = ak.Array(
            np.zeros(len(cutarr)) + int(name[s_idx + 1 :]),
        )

        big_dict["category"] = ak.Array(np.ones(len(cutarr)))
    else:  # background = 0
        big_dict["X_mass"] = ak.Array(np.zeros(len(cutarr)))
        big_dict["S_mass"] = ak.Array(np.zeros(len(cutarr)))
        big_dict["category"] = ak.Array(np.zeros(len(cutarr)))

    big_dict["filename"] = [filename] * len(cutarr)

    return ak.Array(big_dict)
