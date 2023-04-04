import os
import sys

sys.path.append(os.getcwd())

import argparse
import glob
from multiprocessing import cpu_count

import h5py
import numpy as np
import pandas as pd
import torch
import trimesh
import vedo
from joblib import Parallel, delayed
from scipy.spatial import distance_matrix
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
from tqdm.contrib import tzip

from utils import Easy_Mesh, GetVTKTransformationMatrix, get_graph_feature


def rearrange(nparry: np.ndarray) -> np.ndarray:
    nparry[nparry == 17] = 1
    nparry[nparry == 37] = 1
    nparry[nparry == 16] = 2
    nparry[nparry == 36] = 2
    nparry[nparry == 15] = 3
    nparry[nparry == 35] = 3
    nparry[nparry == 14] = 4
    nparry[nparry == 34] = 4
    nparry[nparry == 13] = 5
    nparry[nparry == 33] = 5
    nparry[nparry == 12] = 6
    nparry[nparry == 32] = 6
    nparry[nparry == 11] = 7
    nparry[nparry == 31] = 7
    nparry[nparry == 21] = 8
    nparry[nparry == 41] = 8
    nparry[nparry == 22] = 9
    nparry[nparry == 42] = 9
    nparry[nparry == 23] = 10
    nparry[nparry == 43] = 10
    nparry[nparry == 24] = 11
    nparry[nparry == 44] = 11
    nparry[nparry == 25] = 12
    nparry[nparry == 45] = 12
    nparry[nparry == 26] = 13
    nparry[nparry == 46] = 13
    nparry[nparry == 27] = 14
    nparry[nparry == 47] = 14
    return nparry


def centring(mesh):
    if isinstance(mesh, trimesh.Trimesh):
        mesh.vertices -= mesh.centroid
    elif isinstance(mesh, vedo.Mesh):
        mesh.points(pts=mesh.points() - mesh.center_of_mass())
    else:
        raise NotImplementedError
    return mesh


def add_label(mesh: trimesh.Trimesh, label: np.ndarray) -> vedo.Mesh:
    mesh = vedo.trimesh2vedo(mesh)
    mesh.celldata["Label"] = label
    return mesh


def assemble(mesh_path: str, label_path: str, out_dir: str, offset: int):
    mesh = trimesh.load_mesh(mesh_path)
    label = np.loadtxt(label_path)[offset:]
    result = add_label(centring(mesh), rearrange(label))
    vedo.io.write(
        result,
        os.path.join(
            out_dir, os.path.splitext(os.path.basename(mesh_path))[0] + ".vtp"
        ),
    )

    flipped_result = Easy_Mesh(
        os.path.join(out_dir, os.path.splitext(os.path.basename(mesh_path))[0] + ".vtp")
    )
    flipped_result.mesh_reflection(ref_axis="y")
    flipped_result.to_vtp(
        os.path.join(
            out_dir, os.path.splitext(os.path.basename(mesh_path))[0] + "_f.vtp"
        )
    )


def augment(mesh_path: str, out_dir: str, aug_num: int):
    for i in range(aug_num):
        mesh = vedo.Mesh(mesh_path)
        vtk_matrix = GetVTKTransformationMatrix(
            rotate_X=[-180, 180],
            rotate_Y=[-180, 180],
            rotate_Z=[-180, 180],
            translate_X=[-10, 10],
            translate_Y=[-10, 10],
            translate_Z=[-10, 10],
            scale_X=[0.8, 1.2],
            scale_Y=[0.8, 1.2],
            scale_Z=[0.8, 1.2],
        )  # use default random setting
        mesh.apply_transform(vtk_matrix)
        mesh = centring(mesh)
        mesh.celldata["Normal"] = vedo.vedo2trimesh(mesh).face_normals
        vedo.io.write(
            mesh, os.path.join(out_dir, "A%02d_" % i + os.path.basename(mesh_path))
        )


def gen_metadata(
    idx: int, mesh_path: str, patch_size: int, stage: str, hdf5: h5py.File
):
    mesh = vedo.Mesh(mesh_path)
    N = mesh.ncells
    points = vedo.vtk2numpy(mesh.polydata().GetPoints().GetData())
    ids = vedo.vtk2numpy(mesh.polydata().GetPolys().GetData()).reshape((N, -1))[:, 1:]
    cells = points[ids].reshape(N, 9).astype(dtype="float32")
    labels = mesh.celldata["Label"].astype("int32").reshape(-1, 1)
    normals = mesh.celldata["Normal"]
    barycenters = mesh.cell_centers()

    # normalized data
    maxs = points.max(axis=0)
    mins = points.min(axis=0)
    means = points.mean(axis=0)
    stds = points.std(axis=0)
    nmeans = normals.mean(axis=0)
    nstds = normals.std(axis=0)

    for i in range(3):
        cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
        cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
        cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3
        barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
        normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

    X = np.column_stack((cells, barycenters, normals))
    Y = labels

    # initialize batch of input and label
    X_train = np.zeros([patch_size, X.shape[1]], dtype="float32")
    Y_train = np.zeros([patch_size, Y.shape[1]], dtype="int32")

    # calculate number of valid cells (tooth instead of gingiva)
    positive_idx = np.argwhere(labels > 0)[:, 0]  # tooth idx
    negative_idx = np.argwhere(labels == 0)[:, 0]  # gingiva idx

    num_positive = len(positive_idx)  # number of selected tooth cells

    if num_positive > patch_size:  # all positive_idx in this patch
        positive_selected_idx = np.random.choice(
            positive_idx, size=patch_size, replace=False
        )
        selected_idx = positive_selected_idx
    else:  # patch contains all positive_idx and some negative_idx
        num_negative = patch_size - num_positive  # number of selected gingiva cells
        positive_selected_idx = np.random.choice(
            positive_idx, size=num_positive, replace=False
        )
        negative_selected_idx = np.random.choice(
            negative_idx, size=num_negative, replace=False
        )
        selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

    selected_idx = np.sort(selected_idx, axis=None)

    X_train[:] = X[selected_idx, :]
    Y_train[:] = Y[selected_idx, :]

    X_train = X_train.transpose(1, 0)
    Y_train = Y_train.transpose(1, 0)

    KG_6 = (
        get_graph_feature(torch.from_numpy(X_train[9:12, :]).unsqueeze(0), k=6)
        .squeeze(0)
        .numpy()
    )
    KG_12 = (
        get_graph_feature(torch.from_numpy(X_train[9:12, :]).unsqueeze(0), k=12)
        .squeeze(0)
        .numpy()
    )

    hdf5[stage]["input"][idx] = X_train
    hdf5[stage]["label"][idx] = Y_train
    hdf5[stage]["KG_6"][idx] = KG_6
    hdf5[stage]["KG_12"][idx] = KG_12


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
        delayed(assemble)(i, j, os.path.join(args.output_dir, "upper"), args.offset)
        for i, j in tzip(
            glob.glob(f"./{args.input_dir}/*/上/*.stl"),
            glob.glob(f"./{args.input_dir}/*/上/*.txt"),
            desc="upper",
        )
    )
    Parallel(n_jobs=cpu_count())(
        delayed(assemble)(i, j, os.path.join(args.output_dir, "lower"), args.offset)
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
            for idx, i in enumerate(
                tqdm(eval(f"{abbr}_{item}_list"), desc=f"{jaw_type}_{item}")
            ):
                gen_metadata(idx, i, args.patch_size, item, f)
        f.close()
