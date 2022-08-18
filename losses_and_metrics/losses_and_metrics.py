from MeshSegNet.losses_and_metrics_for_mesh import Generalized_Dice_Loss as gdl
from MeshSegNet.losses_and_metrics_for_mesh import weighting_DSC, weighting_SEN, weighting_PPV

import torch
import torch.nn as nn

class Generalized_Dice_Loss(nn.Module):
    def __init__(self,
                smooth = 1.0,
                ):
        super(Generalized_Dice_Loss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true, class_weights):
        return gdl(y_pred, y_true, class_weights, self.smooth)