from __future__ import annotations

__all__ = ["signal_masses_from_filenames"]

from typing import Iterable

import numpy as np


def signal_masses_from_filenames(filenames: Iterable[str]) -> np.ndarray:
    arr = []
    for filename in filenames:
        name = filename.replace(".root", "")
        x_idx = name.find("X")
        s_idx = name.find("S")
        x = int(name[x_idx + 1 : s_idx - 1])
        s = int(name[s_idx + 1 :])
        arr.append([x, s])
    return np.array(arr)
