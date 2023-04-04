import os
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split

from .h5_dataset import H5_Mesh_Dataset


class LitDataModule(pl.LightningDataModule):
    def __init__(self, cfg: OmegaConf):
        super(LitDataModule, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.dataloader.batch_size
        self.num_workers = cfg.dataloader.num_workers
        self.hdf5_path = cfg.dataset.hdf5_path

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        if os.path.exists(self.hdf5_path):
            train_data = H5_Mesh_Dataset(self.cfg, "train", self.hdf5_path)
            val_data = H5_Mesh_Dataset(self.cfg, "val", self.hdf5_path)
            test_data = H5_Mesh_Dataset(self.cfg, "test", self.hdf5_path)
        else:
            raise FileNotFoundError
        if stage == "fit" or stage is None:
            self.train_dataset = train_data
            self.val_dataset = val_data
        if stage == "test" or stage is None:
            self.test_dataset = test_data

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True, val=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, train=False, val=True)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(
        self, dataset: H5_Mesh_Dataset, train: bool = False, val: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if train and val else False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if train and val else False,
        )
