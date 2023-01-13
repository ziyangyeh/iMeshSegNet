import torch.nn as nn

from .losses_and_metrics import (Generalized_Dice_Loss, weighting_DSC,
                                 weighting_PPV, weighting_SEN)
from .lovasz_losses import *


class Lovasz_Softmax_Flat(nn.Module):
    def __init__(self):
        super(Lovasz_Softmax_Flat, self).__init__()
    
    def forward(self, preds, labels):
        return lovasz_softmax_flat(preds, labels)
