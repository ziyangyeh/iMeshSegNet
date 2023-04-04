import torch
import torch.nn as nn
import torch.nn.functional as F

from .meshsegnet import STN3d, STNkd


class iMeshSegNet(nn.Module):
    def __init__(
        self, num_classes=15, num_channels=15, with_dropout=True, dropout_p=0.5
    ):
        super(iMeshSegNet, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.with_dropout = with_dropout
        self.dropout_p = dropout_p

        # MLP-1 [64, 64]
        self.mlp1_conv1 = torch.nn.Conv1d(self.num_channels, 64, 1)
        self.mlp1_conv2 = torch.nn.Conv1d(64, 64, 1)
        self.mlp1_bn1 = nn.BatchNorm1d(64)
        self.mlp1_bn2 = nn.BatchNorm1d(64)
        # FTM (feature-transformer module)
        self.fstn = STNkd(k=64)
        # GLM-1 (graph-contrained learning modulus)
        self.glm1_conv1 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_conv2_1 = torch.nn.Conv2d(3 * 2, 64, 1, bias=False)
        self.glm1_conv2_2 = torch.nn.Conv2d(64, 32, 1, bias=False)
        self.glm1_bn1 = nn.BatchNorm1d(32)
        self.glm1_bn2_1 = nn.BatchNorm2d(64)
        self.glm1_bn2_2 = nn.BatchNorm2d(32)
        self.glm1_conv2 = torch.nn.Conv1d(32 + 32, 64, 1)
        self.glm1_bn2 = nn.BatchNorm1d(64)
        # MLP-2
        self.mlp2_conv1 = torch.nn.Conv1d(64, 64, 1)
        self.mlp2_bn1 = nn.BatchNorm1d(64)
        self.mlp2_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.mlp2_bn2 = nn.BatchNorm1d(128)
        self.mlp2_conv3 = torch.nn.Conv1d(128, 512, 1)
        self.mlp2_bn3 = nn.BatchNorm1d(512)
        # GLM-2 (graph-contrained learning modulus)
        self.glm2_conv1 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv2_1 = torch.nn.Conv2d(3 * 2, 256, 1, bias=False)
        self.glm2_conv2_2 = torch.nn.Conv2d(256, 128, 1, bias=False)
        self.glm2_conv3_1 = torch.nn.Conv2d(3 * 2, 256, 1, bias=False)
        self.glm2_conv3_2 = torch.nn.Conv2d(256, 128, 1, bias=False)
        self.glm2_bn1 = nn.BatchNorm1d(128)
        self.glm2_bn2_1 = nn.BatchNorm2d(256)
        self.glm2_bn2_2 = nn.BatchNorm2d(128)
        self.glm2_bn3_1 = nn.BatchNorm2d(256)
        self.glm2_bn3_2 = nn.BatchNorm2d(128)
        self.glm2_conv4 = torch.nn.Conv1d(128 * 3, 512, 1)
        self.glm2_bn4 = nn.BatchNorm1d(512)
        # MLP-3
        self.mlp3_conv1 = torch.nn.Conv1d(64 + 512 + 512 + 512, 256, 1)
        self.mlp3_conv2 = torch.nn.Conv1d(256, 256, 1)
        self.mlp3_bn1_1 = nn.BatchNorm1d(256)
        self.mlp3_bn1_2 = nn.BatchNorm1d(256)
        self.mlp3_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.mlp3_conv4 = torch.nn.Conv1d(128, 128, 1)
        self.mlp3_bn2_1 = nn.BatchNorm1d(128)
        self.mlp3_bn2_2 = nn.BatchNorm1d(128)
        # output
        self.output_conv = torch.nn.Conv1d(128, self.num_classes, 1)
        if self.with_dropout:
            self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x, kg_12, kg_6):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # MLP-1
        x = F.relu(self.mlp1_bn1(self.mlp1_conv1(x)))
        x = F.relu(self.mlp1_bn2(self.mlp1_conv2(x)))
        # FTM
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x_ftm = torch.bmm(x, trans_feat)
        # GLM-1
        x_ftm = x_ftm.transpose(2, 1)
        x = F.relu(self.glm1_bn1(self.glm1_conv1(x_ftm)))
        ec12 = self.glm1_bn2_1(self.glm1_conv2_1(kg_12))
        ec12 = self.glm1_bn2_2(self.glm1_conv2_2(ec12))
        ec12 = F.leaky_relu(ec12, negative_slope=0.2)
        ec12 = ec12.max(dim=-1, keepdim=False)[0]
        x = torch.cat([x, ec12], dim=1)
        x = F.relu(self.glm1_bn2(self.glm1_conv2(x)))
        # MLP-2
        x = F.relu(self.mlp2_bn1(self.mlp2_conv1(x)))
        x = F.relu(self.mlp2_bn2(self.mlp2_conv2(x)))
        x_mlp2 = F.relu(self.mlp2_bn3(self.mlp2_conv3(x)))
        if self.with_dropout:
            x_mlp2 = self.dropout(x_mlp2)
        # GLM-2
        x = F.relu(self.glm2_bn1(self.glm2_conv1(x_mlp2)))
        ec12 = self.glm2_bn2_1(self.glm2_conv2_1(kg_12))
        ec12 = self.glm2_bn2_2(self.glm2_conv2_2(ec12))
        ec12 = F.leaky_relu(ec12, negative_slope=0.2)
        ec12 = ec12.max(dim=-1, keepdim=False)[0]
        ec6 = self.glm2_bn3_1(self.glm2_conv3_1(kg_6))
        ec6 = self.glm2_bn3_2(self.glm2_conv3_2(ec6))
        ec6 = F.leaky_relu(ec6, negative_slope=0.2)
        ec6 = ec6.max(dim=-1, keepdim=False)[0]
        x = torch.cat([x, ec12, ec6], dim=1)
        x_glm2 = F.relu(self.glm2_bn4(self.glm2_conv4(x)))
        # GMP
        x = torch.max(x_glm2, 2, keepdim=True)[0]
        # Upsample
        x = torch.nn.Upsample(n_pts)(x)
        # Dense fusion
        x = torch.cat([x, x_ftm, x_mlp2, x_glm2], dim=1)
        # MLP-3
        x = F.relu(self.mlp3_bn1_1(self.mlp3_conv1(x)))
        x = F.relu(self.mlp3_bn1_2(self.mlp3_conv2(x)))
        x = F.relu(self.mlp3_bn2_1(self.mlp3_conv3(x)))
        if self.with_dropout:
            x = self.dropout(x)
        x = F.relu(self.mlp3_bn2_2(self.mlp3_conv4(x)))
        # output
        x = self.output_conv(x)
        x = x.transpose(2, 1).contiguous()
        x = torch.nn.Softmax(dim=-1)(x.view(-1, self.num_classes))
        x = x.view(batchsize, n_pts, self.num_classes)

        return x
