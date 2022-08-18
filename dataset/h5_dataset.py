from typing import Dict, List, Optional, Tuple, Callable

import torch
import h5py
from mpi4py import MPI
from torch.utils.data import Dataset

class H5_Mesh_Dataset(Dataset):
    def __init__(self,
                data_type: str,
                file_path: str,):
        super(H5_Mesh_Dataset, self).__init__()
        self.data_type = data_type
        self.file_path = file_path
        self.hdf5 = h5py.File(self.file_path, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        self.length = len(self.hdf5[self.data_type]["label"])
        self.dataset = self.hdf5[self.data_type]

    def __len__(self):
        
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not hasattr(self, 'hdf5'):
            raise AttributeError 
        data = dict()
        data["input"] = torch.from_numpy(self.dataset["input"][idx])
        data["label"] = torch.from_numpy(self.dataset["label"][idx])
        data["KG_6"] = torch.from_numpy(self.dataset["KG_6"][idx])
        data["KG_12"] = torch.from_numpy(self.dataset["KG_12"][idx])
        return data

    def __del__(self):
        self.hdf5.close()
