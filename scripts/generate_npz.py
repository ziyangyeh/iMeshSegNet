import argparse
import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import pandas as pd

from dataset import iMeshDataset


def gen_csv(input_dir, output_dir="npz"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataframe = pd.DataFrame()
    dataframe["npz_file"] = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files
        if file.endswith(".npz")
    ]
    dataframe.to_csv(os.path.join(output_dir, "npz_dataset.csv"), index=False)


def gen_npz(csv_file, output_dir="npz"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df = pd.read_csv(csv_file)
    ds = iMeshDataset(df)
    for idx, item in enumerate(ds):
        np.savez_compressed(
            os.path.join(output_dir, f"{idx+1}.npz"),
            input=item["input"],
            label=item["label"],
            KG_6=item["KG_6"],
            KG_12=item["KG_12"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv", "--csv_file", type=str, required=True)
    parser.add_argument("-out", "--output_dir", type=str, default="npz")
    args = parser.parse_args()
    gen_npz(args.csv_file, args.output_dir)
    gen_csv(args.output_dir, args.output_dir)
