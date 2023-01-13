from typing import Callable, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
from torch.optim.lr_scheduler import StepLR

from losses_and_metrics import *

from .imeshsegnet import iMeshSegNet
from .meshsegnet import MeshSegNet


class LitModule(pl.LightningModule):
    def __init__(self,cfg: OmegaConf):
        super(LitModule, self).__init__()
        self.cfg = cfg
        self.cfg_train = cfg.train
        self.cfg_model = cfg.model
        
        if self.cfg.model.name == "iMeshSegNet":
            self.model = iMeshSegNet(num_classes=self.cfg_model.num_classes, 
                                    num_channels=self.cfg_model.num_channels, 
                                    with_dropout=self.cfg_model.with_dropout, 
                                    dropout_p=self.cfg_model.dropout_p)
            self.lovasz_fn = Lovasz_Softmax_Flat()
            self.ce_fn = nn.CrossEntropyLoss()
        elif self.cfg.model.name == "MeshSegNet":
            self.model = MeshSegNet(num_classes=self.cfg_model.num_classes, 
                                    num_channels=self.cfg_model.num_channels, 
                                    with_dropout=self.cfg_model.with_dropout, 
                                    dropout_p=self.cfg_model.dropout_p)
            self.loss_fn = Generalized_Dice_Loss()
        else:
            raise AttributeError
        
        self.save_hyperparameters()
        
    def forward(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(self.model, iMeshSegNet):
            outputs = self.model(X['input'], X['KG_12'], X['KG_6'])
        elif isinstance(self.model, MeshSegNet):
            outputs = self.model(X['input'], X['A_S'], X['A_L'])
        else:
            raise NotImplementedError
        
        return outputs
    
    def configure_optimizers(self):
        # Setup the optimizer
        optimizer = create_optimizer_v2(self.parameters(),
                                        opt=self.cfg_train.optimizer,
                                        lr=self.cfg_train.learning_rate,
                                        weight_decay=self.cfg_train.weight_decay,
                                        )
        
        # # Setup the schedulerwarmup_epochs: int,
        # scheduler = lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer,
        #                                                     warmup_epochs=self.cfg_train.warmup_epochs,
        #                                                     max_epochs=self.cfg_train.epochs,
        #                                                     warmup_start_lr=self.cfg_train.warmup_start_lr,
        #                                                     eta_min=self.cfg_train.eta_min,
        #                                                     last_epoch=-1,
        #                                                     )
        scheduler = StepLR(optimizer,
                        step_size=self.cfg.train.step_size,
                        gamma=self.cfg.train.gamma,
                        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        # return optimizer
        
    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")
    
    def _step(self, batch: Dict[str, torch.Tensor], step: str) -> torch.Tensor:
        outputs = self(batch)
        
        class_weights = torch.ones(self.cfg_model.num_classes, device=self.device)
        one_hot_labels = nn.functional.one_hot(batch['label'][:, 0, :], num_classes=self.cfg_model.num_classes)
        
        if self.cfg.model.name == "iMeshSegNet":
            lovasz_loss = self.lovasz_fn(outputs.view(-1, self.cfg_model.num_classes), batch["label"].view(-1))
            ce_loss = self.ce_fn(outputs.view(-1, self.cfg_model.num_classes), batch["label"].view(-1))
            loss = 0.5 * lovasz_loss + 0.5 * ce_loss
        elif self.cfg.model.name == "MeshSegNet":
            loss = self.loss_fn(outputs, one_hot_labels, class_weights)
        else:
            raise AttributeError
        self.log(f"{step}_loss", loss, sync_dist=True)
        
        dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
        self.log(f"{step}_DSC", dsc, sync_dist=True, prog_bar=True)
        sen = weighting_SEN(outputs, one_hot_labels, class_weights)
        self.log(f"{step}_SEN", sen, sync_dist=True, prog_bar=True)
        ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
        self.log(f"{step}_PPV", ppv, sync_dist=True, prog_bar=True)
        
        return loss