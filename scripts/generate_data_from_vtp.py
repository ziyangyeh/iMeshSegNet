import os
import sys

sys.path.append(os.getcwd())

import argparse
import glob
from multiprocessing import cpu_count

import h5py
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
from tqdm.contrib import tzip

from utils import Easy_Mesh

from .generate_data import augment, centring, gen_metadata, rearrange


def flip(mesh_path: str, out_dir: str):
    flipped_result = Easy_Mesh(
        os.path.join(out_dir, os.path.splitext(os.path.basename(mesh_path))[0] + ".vtp")
    )
    flipped_result.mesh_reflection(ref_axis="y")
    flipped_result.to_vtp(
        os.path.join(
            out_dir, os.path.splitext(os.path.basename(mesh_path))[0] + "_f.vtp"
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H5 Dataset Script")
    parser.add_argument(
        "-in",
        "--input_dir",
        type=str,
        metavar="",
        help="Input Directory",
        required=True,
    )
    parser.add_argument(
        "-out",
        "--output_dir",
        type=str,
        metavar="",
        help="Output Directory",
        required=True,
    )
    parser.add_argument(
        "-off", "--offset", type=int, metavar="", help="Offset Value", default=0
    )
    parser.add_argument(
        "-aug", "--augment", type=int, metavar="", help="Augment Value", default=20
    )
    parser.add_argument(
        "-patch", "--patch_size", type=int, metavar="", help="Patch Size", default=7000
    )
    parser.add_argument(
        "-cfg", "--cfg_file", type=str, metavar="", help="CFG File Path", required=True
    )

    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        print("Output Directory exists.")
        pass
    else:
        os.mkdir(args.output_dir)
        os.mkdir(os.path.join(args.output_dir, "upper"))
        os.mkdir(os.path.join(args.output_dir, "lower"))

    print("Assembling...")
    Parallel(n_jobs=cpu_count())(
        delayed(flip)(i, j, os.path.join(args.output_dir, "upper"), args.offset)
        for i, j in tzip(
            glob.glob(f"./{args.input_dir}/*/上/*.stl"),
            glob.glob(f"./{args.input_dir}/*/上/*.txt"),
            desc="upper",
        )
    )
    Parallel(n_jobs=cpu_count())(
        delayed(flip)(i, j, os.path.join(args.output_dir, "lower"), args.offset)
        for i, j in tzip(
            glob.glob(f"./{args.input_dir}/*/下/*.stl"),
            glob.glob(f"./{args.input_dir}/*/下/*.txt"),
            desc="lower",
        )
    )

    print("Renaming...")
    for idx, file_name in enumerate(
        tqdm(glob.glob(f"{args.output_dir}/upper/*_f.vtp"), desc="upper")
    ):
        os.rename(
            os.path.join(
                args.output_dir, "upper", os.path.basename(file_name)[:-6] + ".vtp"
            ),
            os.path.join(args.output_dir, "upper", "Sample_%04d.vtp" % idx),
        )
        os.rename(
            os.path.join(args.output_dir, "upper", os.path.basename(file_name)),
            os.path.join(args.output_dir, "upper", "Sample_%04d.vtp" % (1000 + idx)),
        )
    for idx, file_name in enumerate(
        tqdm(glob.glob(f"{args.output_dir}/lower/*_f.vtp"), desc="lower")
    ):
        os.rename(
            os.path.join(
                args.output_dir, "lower", os.path.basename(file_name)[:-6] + ".vtp"
            ),
            os.path.join(args.output_dir, "lower", "Sample_%04d.vtp" % idx),
        )
        os.rename(
            os.path.join(args.output_dir, "lower", os.path.basename(file_name)),
            os.path.join(args.output_dir, "lower", "Sample_%04d.vtp" % (1000 + idx)),
        )

    print("Augmenting...")
    aug_dir = "augmentation_vtk_data"
    if os.path.exists(aug_dir):
        print("Augmentation Directory exists.")
        pass
    else:
        os.mkdir(aug_dir)
        os.mkdir(os.path.join(aug_dir, "upper"))
        os.mkdir(os.path.join(aug_dir, "lower"))

    Parallel(n_jobs=cpu_count())(
        delayed(augment)(item, os.path.join(aug_dir, "upper"), args.augment)
        for item in tqdm(glob.glob(f"{args.output_dir}/upper/*.vtp"), desc="upper")
    )
    Parallel(n_jobs=cpu_count())(
        delayed(augment)(item, os.path.join(aug_dir, "lower"), args.augment)
        for item in tqdm(glob.glob(f"{args.output_dir}/lower/*.vtp"), desc="lower")
    )

    print("Splitting Upper...")
    u_train_list, tmp_list = train_test_split(
        glob.glob(f"{aug_dir}/upper/*.vtp"), train_size=0.8
    )
    u_val_list, u_test_list = train_test_split(tmp_list, test_size=0.5)
    print("Train Size: ", len(u_train_list))
    print("Val Size: ", len(u_val_list))
    print("Test Size: ", len(u_test_list))

    print("Spliting Lower...")
    l_train_list, tmp_list = train_test_split(
        glob.glob(f"{aug_dir}/lower/*.vtp"), train_size=0.8
    )
    l_val_list, l_test_list = train_test_split(tmp_list, test_size=0.5)
    print("Train Size: ", len(l_train_list))
    print("Val Size: ", len(l_val_list))
    print("Test Size: ", len(l_test_list))

    print("Generating HDF5...")
    h5_dir = "h5"
    if os.path.exists(h5_dir):
        print("HDF5 Directory exists.")
        pass
    else:
        os.mkdir(h5_dir)

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.cfg_file)

    for jaw_type, abbr in zip(["upper", "lower"], ["u", "l"]):
        f = h5py.File(os.path.join(h5_dir, f"{jaw_type}.hdf5"), "a")
        for item in ["train", "val", "test"]:
            length = len(eval(f"{abbr}_{item}_list"))
            data_group = f.create_group(item)
            data_group.create_dataset(
                "label",
                data=np.zeros((length, 1, args.patch_size), dtype=np.int32),
                dtype="int32",
            )
            data_group.create_dataset(
                "input",
                data=np.zeros(
                    (length, cfg.model.num_channels, args.patch_size), dtype=np.float32
                ),
                dtype="float32",
            )
            data_group.create_dataset(
                "KG_6",
                data=np.zeros((length, 3 * 2, args.patch_size, 6), dtype=np.float32),
                dtype="float32",
            )
            data_group.create_dataset(
                "KG_12",
                data=np.zeros((length, 3 * 2, args.patch_size, 12), dtype=np.float32),
                dtype="float32",
            )
            data_group.create_dataset(
                "A_S",
                data=np.zeros(
                    (length, args.patch_size, args.patch_size), dtype=np.float32
                ),
                dtype="float32",
            )
            data_group.create_dataset(
                "A_L",
                data=np.zeros(
                    (length, args.patch_size, args.patch_size), dtype=np.float32
                ),
                dtype="float32",
            )
            for idx, i in enumerate(
                tqdm(eval(f"{abbr}_{item}_list"), desc=f"{jaw_type}_{item}")
            ):
                gen_metadata(idx, i, args.patch_size, item, f)
        f.close()
