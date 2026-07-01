import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from monai.losses import DiceCELoss
from monai.networks.nets import UNETR, UNet, ViT


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ReceptiveFieldBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch0 = nn.Sequential(BasicConv(in_channels, out_channels, 1))
        self.branch1 = nn.Sequential(
            BasicConv(in_channels, out_channels, 1), BasicConv(out_channels, out_channels, 3, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_channels, out_channels, 1), BasicConv(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_channels, out_channels, 1), BasicConv(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.conv_cat = BasicConv(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        return F.relu(x_cat + self.conv_res(x), inplace=True)


class PartialDecoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up1 = BasicConv(channel, channel, 3, padding=1)
        self.conv_up2 = BasicConv(channel, channel, 3, padding=1)
        self.conv_concat = BasicConv(3 * channel, channel, 3, padding=1)

    def forward(self, f2, f3, f4):
        f3_up = self.upsample(f4)
        f3_out = self.conv_up1(f3_up * f3)
        f2_up = self.upsample(f3_out)
        f2_out = self.conv_up2(f2_up * f2)
        f4_up_2x = self.upsample(self.upsample(f4))
        f3_up_1x = self.upsample(f3_out)
        return self.conv_concat(torch.cat((f2_out, f3_up_1x, f4_up_2x), 1))


class ReverseAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Sequential(BasicConv(channel, channel, 3, padding=1), nn.Conv2d(channel, 1, 1))

    def forward(self, features, prior_mask):
        reverse_mask = -1 * torch.sigmoid(prior_mask) + 1
        return self.conv(features * reverse_mask)


class PraNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.rfb2 = ReceptiveFieldBlock(512, 32)
        self.rfb3 = ReceptiveFieldBlock(1024, 32)
        self.rfb4 = ReceptiveFieldBlock(2048, 32)

        self.pd = PartialDecoder(32)
        self.ra3 = ReverseAttention(32)
        self.ra2 = ReverseAttention(32)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.map_out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        f2 = self.rfb2(l2)
        f3 = self.rfb3(l3)
        f4 = self.rfb4(l4)

        global_map = self.pd(f2, f3, f4)
        global_pred = self.map_out(global_map)

        ra3_out = self.ra3(f3, global_pred)
        ra3_pred = global_pred + self.upsample2(ra3_out)

        ra2_out = self.ra2(f2, ra3_pred)
        ra2_pred = ra3_pred + self.upsample2(ra2_out)

        return self.upsample4(ra2_pred)


def build_loss():
    return DiceCELoss(
        include_background=False, sigmoid=True, jaccard=False, reduction="mean", weight=torch.tensor([1.0])
    )
