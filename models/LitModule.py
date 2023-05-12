from typing import Callable, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.losses import GeneralizedDiceLoss
from omegaconf import OmegaConf
from torch.optim import *
from torch.optim.lr_scheduler import *

from losses_and_metrics import *

from .imeshsegnet import iMeshSegNet


class LitModule(pl.LightningModule):
    def __init__(self, cfg_train_params: OmegaConf, cfg_model: OmegaConf) -> None:
        super(LitModule, self).__init__()
        self.cfg_train = cfg_train_params
        self.cfg_model = cfg_model

        self.model = iMeshSegNet(
            num_classes=self.cfg_model.num_classes,
            num_channels=self.cfg_model.num_channels,
            with_dropout=self.cfg_model.with_dropout,
            dropout_p=self.cfg_model.dropout,
        )
        self.dice_fn = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
        self.lovasz_fn = LovaszSoftmax()
        self.ce_fn = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def forward(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(X["input"], X["KG_12"], X["KG_6"])
        return outputs

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg_train.learning_rate,
            weight_decay=self.cfg_train.weight_decay,
        )

        scheduler = StepLR(
            optimizer,
            step_size=self.cfg_train.step_size,
            gamma=self.cfg_train.gamma,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self(batch).transpose(2, 1).softmax(dim=-1)
    
    def _step(self, batch: Dict[str, torch.Tensor], step: str) -> torch.Tensor:
        outputs = self(batch)

        class_weights = torch.ones(self.cfg_model.num_classes, device=self.device)
        one_hot_labels = nn.functional.one_hot(
            batch["label"][:, 0, :], num_classes=self.cfg_model.num_classes
        )

        dice_loss = self.dice_fn(outputs, batch["label"])
        self.log(f"{step}_dice_loss", dice_loss, sync_dist=True)

        lovasz_loss = self.lovasz_fn(
            outputs.unsqueeze(-1),
            batch["label"].squeeze(1).unsqueeze(-1),
        )
        self.log(f"{step}_lovasz_loss", lovasz_loss, sync_dist=True)

        ce_loss = self.ce_fn(outputs, batch["label"].squeeze(1))
        self.log(f"{step}_ce_loss", ce_loss, sync_dist=True)

        loss = dice_loss * 0.4 + lovasz_loss * 0.4 + ce_loss * 0.2
        self.log(f"{step}_loss", loss, sync_dist=True, prog_bar=True)

        outputs = outputs.detach().transpose(2, 1).softmax(dim=-1)
        dsc = weighting_DSC(outputs.softmax(dim=1), one_hot_labels, class_weights)
        self.log(f"{step}_DSC", dsc, sync_dist=True, prog_bar=True)
        sen = weighting_SEN(outputs, one_hot_labels, class_weights)
        self.log(f"{step}_SEN", sen, sync_dist=True)
        ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
        self.log(f"{step}_PPV", ppv, sync_dist=True)

        return loss
