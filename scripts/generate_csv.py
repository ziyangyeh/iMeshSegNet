import os
import pandas as pd
import argparse


def gen_csv(input_dir, output_dir="data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataframe = pd.DataFrame()
    dataframe["vtp_file"] = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files
        if file.endswith(".vtp")
    ]
    dataframe.to_csv(os.path.join(output_dir, "dataset.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--input_dir", type=str, required=True)
    parser.add_argument("-out", "--output_dir", type=str, default="data")
    args = parser.parse_args()
    gen_csv(args.input_dir, args.output_dir)
