import argparse
import gc

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold

from dataset import LitDataModule
from models import LitModule

torch.set_float32_matmul_precision("high")


def train(fold: int, cfg: OmegaConf, dataframe=pd.DataFrame):
    pl.seed_everything(cfg.seed)

    datamodule = LitDataModule(
        fold=fold,
        cfg_ds=cfg.dataset,
        cfg_dl=cfg.dataloader,
        dataframe=dataframe,
        num_classes=cfg.model.num_classes,
        mode="csv" if cfg.dataset.mode == "csv" else "npy",
    )

    datamodule.setup()

    module = LitModule(cfg.train, cfg.model)

    dsc_model_checkpoint = ModelCheckpoint(
        dirpath=cfg.train.checkpoint_dir,
        monitor="val_DSC",
        mode="max",
        filename=f"{module.model.__class__.__name__}_{cfg.model.num_classes}_Classes_{cfg.train.precision}_f_best_DSC",
        verbose="True",
    )
    loss_model_checkpoint = ModelCheckpoint(
        dirpath=cfg.train.checkpoint_dir,
        monitor="val_loss",
        mode="min",
        filename=f"{module.model.__class__.__name__}_{cfg.model.num_classes}_Classes_{cfg.train.precision}_f_best_loss",
        verbose="True",
    )

    trainer = pl.Trainer(
        callbacks=[dsc_model_checkpoint, loss_model_checkpoint],
        benchmark=cfg.train.benchmark,
        # deterministic=cfg.train.deterministic,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy="ddp" if cfg.train.ddp else "auto",
        max_epochs=cfg.train.epochs,
        precision=cfg.train.precision,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        logger=WandbLogger(
            project=cfg.logger.project, name=f"iMeshSegNet-{cfg.train.precision}f"
        ),
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
    )

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "-cfg",
        "--config_file",
        type=str,
        metavar="",
        help="configuration file",
        default="config/default.yaml",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)
    dataframe = pd.read_csv(cfg.dataset.csv_path)

    for fold, (train_idx, valid_idx) in enumerate(
        KFold(n_splits=cfg.k_fold, random_state=cfg.seed, shuffle=True).split(dataframe)
    ):
        dataframe.loc[valid_idx, "fold"] = fold

    for i in range(cfg.k_fold):
        trainer = train(i, cfg, dataframe)
        wandb.finish()
        del trainer
        gc.collect()
