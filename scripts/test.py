import os
import sys

sys.path.append(os.getcwd())

import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf

from dataset import LitDataModule
from models import LitModule


def test(cfg_path: str, ckpt_path: str):
    cfg = OmegaConf.load(cfg_path)

    pl.seed_everything(cfg.seed)

    datamodule = LitDataModule(cfg)

    datamodule.setup()

    module = LitModule(cfg)

    trainer = pl.Trainer(
        benchmark=cfg.train.benchmark,
        deterministic=cfg.train.deterministic
        if cfg.model.name == "MeshSegNet"
        else False,
        accelerator="gpu" if cfg.train.accelerator == "gpu" else "cpu",
        devices=1,
        max_epochs=cfg.train.epochs,
        precision=cfg.train.precision,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        auto_lr_find=cfg.train.auto_lr_find,
        auto_scale_batch_size=cfg.train.auto_scale_batch_size,
        fast_dev_run=cfg.train.fast_dev_run,
    )

    trainer.tune(module, datamodule=datamodule)

    trainer.test(module, dataloaders=datamodule, ckpt_path=ckpt_path)


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
    parser.add_argument(
        "-ckpt", "--ckpt_path", type=str, metavar="", help="ckpt file", required=True
    )

    args = parser.parse_args()

    test(args.config_file, args.ckpt_path)
