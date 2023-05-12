from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import vedo
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils import GetVTKTransformationMatrix

from .imeshdataset import iMeshDataset


def augment(mesh: vedo.Mesh):
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
    )
    mesh.apply_transform(vtk_matrix)
    return mesh


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fold: int,
        cfg_ds: OmegaConf,
        cfg_dl: OmegaConf,
        dataframe: pd.DataFrame,
        num_classes: int,
        mode: str,
    ):
        super(LitDataModule, self).__init__()
        self.fold = fold
        self.num_classes = num_classes
        self.path_size = cfg_ds.patch_size
        self.batch_size = cfg_dl.batch_size
        self.rearrange = cfg_ds.rearrange
        self.num_workers = cfg_dl.num_workers
        self.dataframe = dataframe
        self.mode = mode
        self.transform = augment if cfg_ds.transform else None

        self.save_hyperparameters(ignore=["cfg", "dataframe"])

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_df = self.dataframe[self.dataframe["fold"] != self.fold].reset_index()
            val_df = self.dataframe[self.dataframe["fold"] == self.fold].reset_index()
            self.train_dataset = iMeshDataset(
                dataframe=train_df,
                num_classes=self.num_classes,
                patch_size=self.path_size,
                rearrange=self.rearrange,
                transform=self.transform,
                mode=self.mode,
            )
            self.val_dataset = iMeshDataset(
                dataframe=val_df,
                num_classes=self.num_classes,
                patch_size=self.path_size,
                rearrange=self.rearrange,
                mode=self.mode,
            )
        if stage == "test" or stage is None:
            self.test_dataset = iMeshDataset(
                dataframe=self.dataframe,
                num_classes=self.num_classes,
                patch_size=self.path_size,
                rearrange=self.rearrange,
                mode=self.mode,
            )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True, val=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, train=False, val=True)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(
        self, dataset: iMeshDataset, train: bool = False, val: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if train and val else False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if train and val else False,
        )
