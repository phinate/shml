from __future__ import annotations

__all__ = ["EventDataset"]

import json

import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import signal_masses_from_filenames


class EventDataset(Dataset):
    def __init__(
        self,
        data: ak.Array | str,
        fields: list[str] | None = None,
        config_path: str | None = None,
        use_truth_masses: bool = True,
    ) -> None:
        if isinstance(data, str):
            akarr = ak.from_parquet(data)
        else:
            akarr = data

        fields = fields or akarr.fields

        # cast array to float32 so numpy doesn't complain about mixed dtypes
        self.awkward_array = ak.values_astype(akarr[fields], np.float64)
        self.length = len(self.awkward_array)

        config_path = config_path or "config.json"
        with open(config_path) as f:
            self.config = json.load(f)

        # grab relative signal proportions to randomly sample for bkg later on
        sig_prop_dict = self.config["signal_proportions"]
        self.sig_props = list(sig_prop_dict.values())
        self.sig_masses = signal_masses_from_filenames(sig_prop_dict.keys())

        self.use_truth_masses = use_truth_masses  # parametrized nn

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Assumes single integer used for indexing."""
        awk = self.awkward_array[idx]
        fields = self.awkward_array.fields
        cat = awk["category"]
        weight = awk["weight"]

        # remove entries we've already processed from fields
        fields.remove("category")
        fields.remove("weight")
        batcharr = awk[fields]

        # pnn
        if self.use_truth_masses:
            if cat == 0.0:
                # background has no truth masses (both set to proxy val of 0)
                # draw random truth masses using signal proportions from training data
                X_mass, S_mass = self.sig_masses[
                    np.random.choice(
                        np.arange(0, len(self.sig_props)),
                        p=self.sig_props,
                    )
                ]
            elif cat == 1.0:
                X_mass, S_mass = batcharr["X_mass"], batcharr["S_mass"]
            else:
                raise NotImplementedError(
                    f"unreachable... val = {cat}, type = {type(cat)}",
                )

            # longer boilerplate since awkward arrays are immutable
            fields.remove("X_mass")
            fields.remove("S_mass")
            batcharr = ak.with_field(
                ak.with_field(
                    batcharr[fields],
                    X_mass,
                    "X_mass",
                ),
                S_mass,
                "S_mass",
            )

        # normalize
        scalers = self.config["scalers"]
        features = []
        for field in fields:
            features.append(
                ak.to_numpy(
                    (batcharr[field] - scalers[field]["mean"]) / scalers[field]["std"],
                ),
            )

        X = np.asarray(features)
        y = ak.to_numpy(cat)
        # supply weight separately for use with loss calc
        w = ak.to_numpy(weight)

        return torch.Tensor(X), y, w

    def __len__(self) -> int:
        return self.length
