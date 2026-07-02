import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.networks.nets import UNETR, UNet


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype="normal"):
        super().__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.scale != 1 and self.stype == "normal":
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == "stage":
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4):
        super().__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [
            block(self.inplanes, planes, stride, downsample, stype="stage", baseWidth=self.baseWidth, scale=self.scale)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)


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
        resized_mask = F.interpolate(prior_mask, size=features.shape[2:], mode="bilinear", align_corners=True)
        reverse_mask = -1 * torch.sigmoid(resized_mask) + 1
        return self.conv(features * reverse_mask)


class PraNet(nn.Module):
    def __init__(self, backbone_weights=None, pranet_weights=None):
        super().__init__()
        self.resnet = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4)

        if backbone_weights and os.path.exists(backbone_weights):
            self.resnet.load_state_dict(torch.load(backbone_weights))

        self.rfb2 = ReceptiveFieldBlock(512, 32)
        self.rfb3 = ReceptiveFieldBlock(1024, 32)
        self.rfb4 = ReceptiveFieldBlock(2048, 32)

        self.pd = PartialDecoder(32)
        self.ra3 = ReverseAttention(32)
        self.ra2 = ReverseAttention(32)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.map_out = nn.Conv2d(32, 1, 1)

        if pranet_weights and os.path.exists(pranet_weights):
            self.load_state_dict(torch.load(pranet_weights))

    def forward(self, x):
        l0 = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        l0_pool = self.resnet.maxpool(l0)
        l1 = self.resnet.layer1(l0_pool)
        l2 = self.resnet.layer2(l1)
        l3 = self.resnet.layer3(l2)
        l4 = self.resnet.layer4(l3)

        f2 = self.rfb2(l2)
        f3 = self.rfb3(l3)
        f4 = self.rfb4(l4)

        global_map = self.pd(f2, f3, f4)
        global_pred = self.map_out(global_map)

        ra3_out = self.ra3(f3, global_pred)
        ra3_pred = global_pred + self.upsample2(ra3_out)

        ra2_out = self.ra2(f2, ra3_pred)
        ra2_pred = ra3_pred + ra2_out

        return self.upsample8(ra2_pred)


def build_loss():
    return DiceCELoss(
        include_background=False, sigmoid=True, jaccard=False, reduction="mean", weight=torch.tensor([1.0])
    )
