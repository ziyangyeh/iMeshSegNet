from typing import Dict, List, Optional, Tuple, Callable
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pl_bolts.optimizers import lr_scheduler

from .imeshsegnet import iMeshSegNet
from losses_and_metrics import *

class LitModule(pl.LightningModule):
    def __init__(
        self,
        cfg,
    ):
        super(LitModule, self).__init__()

        self.cfg = cfg.model
        self.cfg_optimizer = self.cfg.optimizer
        self.cfg_scheduler = self.cfg.scheduler
        self.cfg_scheduler.epochs = self.cfg.epochs
        self.batch_size = self.cfg.batch_size
        self.learning_rate = self.cfg.learning_rate

        self.model = iMeshSegNet(num_classes=15, num_channels=15, with_dropout=True, dropout_p=0.5)

        self.loss_fn = Generalized_Dice_Loss()

    def forward(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        dof = self.model(X)
        return dof

    def configure_optimizers(self):
        # Setup the optimizer
        optimizer = create_optimizer_v2(self.parameters(),
                                        opt=self.cfg_optimizer.NAME,
                                        lr=self.cfg_optimizer.learning_rate,
                                        weight_decay=self.cfg_optimizer.weight_decay,
                                        )

        # Setup the schedulerwarmup_epochs: int,
        scheduler = lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer,
                                                            warmup_epochs=self.cfg_scheduler.warmup_epochs,
                                                            max_epochs=self.cfg_scheduler.epochs,
                                                            warmup_start_lr=self.cfg_scheduler.warmup_start_lr,
                                                            eta_min=self.cfg_scheduler.eta_min,
                                                            last_epoch=-1,
                                                            )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def _step(self, batch: Dict[str, torch.Tensor], step: str) -> torch.Tensor:
        class_weights = torch.ones(15, device=self.device)
        KG_12 = batch['KG_12']
        KG_6 = batch['KG_6']
        one_hot_labels = nn.functional.one_hot(batch['labels'][:, 0, :], num_classes=15)

        outputs = self(batch['cells'], KG_12, KG_6)
        loss = self.loss_fn(outputs, one_hot_labels, class_weights)
        self.log(f"{step}_loss", loss)
        dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
        self.log(f"{step}_DSC", dsc)
        sen = weighting_SEN(outputs, one_hot_labels, class_weights)
        self.log(f"{step}_SEN", sen)
        ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
        self.log(f"{step}_PPV", ppv)

        return loss