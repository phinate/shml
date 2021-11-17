from __future__ import annotations

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np
import uproot

import shml


def initial(writeout_dir: str, filepath: str, cut_type: str) -> ak.Array:
    print(f"running preprocessing with {cut_type} preselection")
    filenames = list(filter(lambda item: "root" in item, os.listdir(filepath)))
    filenames.remove("yyjj.root")

    allarrs = []
    for filename in filenames:
        print(f"preprocessing {filename}...", end="")
        abs_path = filepath + "/" + filename
        tree = uproot.open(abs_path)["CollectionTree"]
        all_vars = shml.variable_names(cut_type)
        # get arrays from uproot
        allarrs.append(
            shml.preprocess(
                tree.arrays(all_vars.keys(), aliases=all_vars),
                preselection=cut_type,
                filename=filename,
            ),
        )
        print("done!")

    # save to file
    ak.to_parquet(
        ak.concatenate(allarrs),
        os.path.join(
            writeout_dir,
            "without_yyjj.parquet",
        ),
    )

    del allarrs

    # make this not hardcoded...
    print("now preprocessing yyjj.root in 176 chunks")
    filename = "yyjj.root"
    abs_path = filepath + "/" + filename
    all_vars = shml.variable_names(cut_type)
    tree = uproot.open(abs_path)["CollectionTree"]

    for i, arrs in enumerate(
        tree.iterate(all_vars.keys(), aliases=all_vars, step_size=100000),
    ):
        print(f"processing chunk {i}...", end=" ")
        ak.to_parquet(
            shml.preprocess(
                arrs,
                preselection=cut_type,
                filename=filename,
            ),
            os.path.join(writeout_dir, f"preprocessed_yyjj_chunk_{i}.parquet"),
        )
        print("done!")
        sys.stdout.write("\033[2K\033[1G")  # erase and go to start of line

    print("loading files and concatenating...", end=" ")
    a = ak.from_parquet(os.path.join(writeout_dir, "without_yyjj.parquet"))
    b = [
        ak.from_parquet(
            os.path.join(writeout_dir, f"preprocessed_yyjj_chunk_{i}.parquet"),
        )
        for i in range(176 + 1)
    ]
    outfile = os.path.join(writeout_dir, "preprocessed_data.parquet")
    array = ak.concatenate(b + [a])
    ak.to_parquet(array, outfile)
    print("done!")
    print(f"preprocessed data written out to {outfile}")
    print("cleaning up...", end=" ")
    for i in range(176 + 1):
        os.remove(
            os.path.join(
                writeout_dir,
                f"preprocessed_yyjj_chunk_{i}.parquet",
            ),
        )
    os.remove(os.path.join(writeout_dir, "without_yyjj.parquet"))
    print("done!")
    return array


def weight(arrs: ak.Array, writeout_dir: str, filepath: str, cut_type: str) -> ak.Array:
    print("performing weight normalization...")
    # weight normalization per-category
    filenames = list(filter(lambda item: "root" in item, os.listdir(filepath)))
    signal_files = tuple(filter(lambda item: item[0] == "X", filenames))

    list_of_arrs = []
    for file in signal_files:
        arr = arrs[arrs["filename"] == file]
        fields = arr.fields
        weights = arr["weight"] / ak.sum(arr["weight"], axis=0)
        fields.remove("weight")
        list_of_arrs.append(ak.with_field(arr[fields], weights, "weight"))

    arr = arrs[arrs["category"] == 0]
    fields = arr.fields
    weights = arr["weight"] / ak.sum(arr["weight"], axis=0)
    fields.remove("weight")
    bkg = ak.with_field(arr[fields], weights, "weight")

    all_normed = ak.concatenate(list_of_arrs + [bkg])
    ak.to_parquet(
        all_normed,
        os.path.join(
            writeout_dir,
            "all_data_normed_weights.parquet",
        ),
    )
    print("done!")
    return all_normed


def split(
    all_normed: ak.Array, writeout_dir: str, filepath: str, cut_type: str
) -> ak.Array:
    print("performing train/test split...")
    train_mask = all_normed["EventNumber"] % 4 <= 1
    test_mask = all_normed["EventNumber"] % 4 == 2
    train = all_normed[train_mask]
    ak.to_parquet(
        train,
        os.path.join(
            writeout_dir,
            "train.parquet",
        ),
    )
    ak.to_parquet(
        all_normed[test_mask],
        os.path.join(
            writeout_dir,
            "test.parquet",
        ),
    )
    print("done!")
    print(
        f"files written to {writeout_dir}/train.parquet, {writeout_dir}/test.parquet",
    )
    return train


def cfg(train: ak.Array, writeout_dir: str, filepath: str, cut_type: str) -> ak.Array:

    print("getting relative signal proportions from training data...")
    filenames = list(filter(lambda item: "root" in item, os.listdir(filepath)))
    signal_files = tuple(filter(lambda item: item[0] == "X", filenames))
    prop_dict = {}
    for name in signal_files:
        prop_dict[name] = len(train[train["filename"] == name])
    tot = sum(prop_dict.values())
    sig_props = {k: v / tot for k, v in prop_dict.items()}
    print("done!")
    print("caclulating mean and std of input features...")
    fields = shml.ml_vars()
    fields.remove("weight")
    fields.remove("category")

    store: dict[str, dict[str, float]] = {}

    for field in fields:
        feature = train[field]
        if field in ["X_mass", "S_mass"]:
            # don't include background proxy masses
            feature = feature[feature != 0]
        mean = np.mean(feature)
        std = np.std(feature)

        store[field] = dict(mean=mean, std=std)
    config = dict(signal_proportions=sig_props, scalers=store)
    print("done!")
    print("saving to config.json...")
    with open("config.json", "w+") as f:
        f.write(json.dumps(config))
    print("done!")
    print("have a good rest of your day! ^_^")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        help="Choose the directory to write the output files to.",
    )
    parser.add_argument(
        "--rootpath",
        help="Specify the directory of the .root files you want to convert.",
        default="/eos/user/h/hhsukth/GroupSpace/MiniTree/h027/",
    )
    parser.add_argument(
        "--preselection-type",
        help="Specify the type of preselection you'd like to use.",
        default="loose",
    )
    parser.add_argument(
        "--entrypoint",
        help="Run from this step!",
        default="start",
    )

    args = parser.parse_args()
    cut_type = args.preselection_type
    writeout_dir = args.outdir
    filepath = args.rootpath
    entry = args.entrypoint
    all_args = (writeout_dir, filepath, cut_type)

    if entry == "start":
        cfg(split(weight(initial(*all_args), *all_args), *all_args), *all_args)
    elif entry == "weight":
        cfg(
            split(
                weight(
                    ak.from_parquet(
                        os.path.join(writeout_dir, "preprocessed_data.parquet")
                    ),
                    *all_args,
                ),
                *all_args,
            ),
            *all_args,
        )
    elif entry == "split":
        cfg(
            split(
                ak.from_parquet(
                    os.path.join(writeout_dir, "all_data_normed_weights.parquet")
                ),
                *all_args,
            ),
            *all_args,
        )
    elif entry == "config":
        cfg(ak.from_parquet(os.path.join(writeout_dir, "train.parquet")), *all_args)
    else:
        print("doing nothing!")
