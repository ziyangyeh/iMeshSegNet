from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import vedo
from torch.utils.data import Dataset

from utils import get_graph_feature_cpu, rearrange


class iMeshDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_classes: int = 15,
        patch_size: int = 7000,
        rearrange: bool = False,
        transform: Optional[Callable] = None,
        mode: str = "csv",
    ) -> None:
        super(iMeshDataset, self).__init__()
        self.dataframe = dataframe
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.rearrange = rearrange
        self.mode = mode
        self.transform = transform if mode == "csv" else None

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx) -> dict:
        if self.mode == "csv":
            mesh = vedo.Mesh(self.dataframe.loc[idx, "vtp_file"])

            if self.rearrange:
                mesh.celldata["Label"] = rearrange(mesh.celldata["Label"])

            labels = mesh.celldata["Label"].astype("int32").reshape(-1, 1)

            if self.transform:
                mesh = self.transform(mesh)

            points = mesh.points()
            mean_cell_centers = mesh.center_of_mass()
            points[:, 0:3] -= mean_cell_centers[0:3]

            ids = np.array(mesh.faces())
            cells = points[ids].reshape(mesh.ncells, 9).astype(dtype="float32")

            mesh.compute_normals()
            normals = mesh.normals(cells=True)

            barycenters = mesh.cell_centers()
            barycenters -= mean_cell_centers[0:3]

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
            X_train = np.zeros([self.patch_size, X.shape[1]], dtype="float32")
            Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype="int32")

            # calculate number of valid cells (tooth instead of gingiva)
            positive_idx = np.argwhere(labels > 0)[:, 0]  # tooth idx
            negative_idx = np.argwhere(labels == 0)[:, 0]  # gingiva idx

            num_positive = len(positive_idx)  # number of selected tooth cells

            if num_positive > self.patch_size:  # all positive_idx in this patch
                positive_selected_idx = np.random.choice(
                    positive_idx, size=self.patch_size, replace=False
                )
                selected_idx = positive_selected_idx
            else:  # patch contains all positive_idx and some negative_idx
                num_negative = (
                    self.patch_size - num_positive
                )  # number of selected gingiva cells
                positive_selected_idx = np.random.choice(
                    positive_idx, size=num_positive, replace=False
                )
                negative_selected_idx = np.random.choice(
                    negative_idx, size=num_negative, replace=False
                )
                selected_idx = np.concatenate(
                    (positive_selected_idx, negative_selected_idx)
                )

            selected_idx = np.sort(selected_idx, axis=None)

            X_train[:] = X[selected_idx, :]
            Y_train[:] = Y[selected_idx, :]

            X_train = X_train.transpose(1, 0)
            Y_train = Y_train.transpose(1, 0)

            KG_6 = get_graph_feature_cpu(X_train[9:12, :], k=6)

            KG_12 = get_graph_feature_cpu(X_train[9:12, :], k=12)
        else:
            npz_data = np.load(self.dataframe.loc[idx, "npz_file"])
            X_train = npz_data["input"]
            Y_train = npz_data["label"]
            KG_6 = npz_data["KG_6"]
            KG_12 = npz_data["KG_12"]

        return {
            "input": X_train,
            "label": Y_train.astype(np.int64),
            "KG_6": KG_6,
            "KG_12": KG_12,
        }
