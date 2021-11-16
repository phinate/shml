"""Helper functions to compute vectorized versions of useful kinematic quantities."""
from __future__ import annotations

__all__ = ["bound_phi", "delta_R"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import awkward as ak

import numpy as np


def bound_phi(
    phi: float | ak.Array,
) -> float | ak.Array:  # -pi to pi (thanks to github.com/scikit-hep/vector)
    """Maps phi from -pi to pi."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def delta_R(dphi: float | ak.Array, deta: float | ak.Array) -> float | ak.Array:
    """Computes the distance in (eta, phi) space between particles."""
    return np.sqrt(bound_phi(dphi) ** 2 + deta ** 2)
