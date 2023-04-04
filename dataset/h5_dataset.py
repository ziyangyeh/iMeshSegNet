from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5_Mesh_Dataset(Dataset):
    def __init__(
        self,
        cfg,
        data_type: str,
        file_path: str,
    ):
        super(H5_Mesh_Dataset, self).__init__()
        self.cfg = cfg
        self.data_type = data_type
        self.file_path = file_path
        self.hdf5 = h5py.File(self.file_path, "r")
        self.length = len(self.hdf5[self.data_type]["label"])
        self.dataset = self.hdf5[self.data_type]

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not hasattr(self, "hdf5"):
            raise AttributeError
        data = dict()
        data["input"] = torch.from_numpy(self.dataset["input"][idx])
        data["label"] = torch.from_numpy(self.dataset["label"][idx].astype(np.int64))
        if self.cfg.model.name == "iMeshSegNet":
            data["KG_6"] = torch.from_numpy(self.dataset["KG_6"][idx])
            data["KG_12"] = torch.from_numpy(self.dataset["KG_12"][idx])
        elif self.cfg.model.name == "MeshSegNet":
            data["A_S"] = torch.from_numpy(self.dataset["A_S"][idx])
            data["A_L"] = torch.from_numpy(self.dataset["A_L"][idx])
        else:
            raise NotImplementedError
        return data


if __name__ == "__main__":
    import torch.nn as nn
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("config/default.yaml")
    ds = H5_Mesh_Dataset(cfg, "val", cfg.dataset.hdf5_path)
    one_batch_label = ds[0]["label"].unsqueeze(0)
    print(one_batch_label.shape)
    print(one_batch_label.min())
    print(one_batch_label.max())
    np_one_batch_label = one_batch_label.numpy()
    zero_ind = torch.from_numpy(np.where(np_one_batch_label == 0)[-1])
    print(zero_ind)
    one_hot_labels = nn.functional.one_hot(
        one_batch_label[:, 0, :], num_classes=cfg.model.num_classes
    )
    print(one_hot_labels[0][zero_ind])
