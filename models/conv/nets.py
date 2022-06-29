import torch
import torch.nn.functional as F
from torch import nn


class ConvLayers(nn.Module):
    """Convolutional feature extractor model for (natural) images."""

    def __init__(self, image_channels):
        super(ConvLayers, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=(4, 4), stride=2, padding=(15, 15))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=(15, 15))
        self.out_channels = 128
        self.out_feature_dim = 128 * 28 * 28

    def forward(self, x):
        x = F.relu(self.conv1(x))
        feature = F.relu(self.conv2(x))

        return feature


class DeconvLayers(nn.Module):
    """'Deconvolutional' feature decoder model for (natural) images."""

    def __init__(self, image_channels):
        super(DeconvLayers, self).__init__()
        self.image_channels = image_channels
        self.in_channel = 128
        self.in_size = 7
        self.in_feature_dim = 7 * 7 * 128
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=self.image_channels, kernel_size=4, padding=1, stride=2)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))

        return x


# ---------------------------------------------------------------------------------------------------


class ResBlock(nn.Module):
    """
    Input:  [batch_size] x [dim] x [image_size] x [image_size] tensor
    Output: [batch_size] x [dim] x [image_size] x [image_size] tensor
    """

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel_num=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(channel_num, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.out_channels = 512
        self.out_feature_dim = 512 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return out


def ResNet18(channel_num=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], channel_num=channel_num)


class DeconvResnet(nn.Module):
    """'Deconvolutional' feature decoder model for (natural) images using ResBlock as the backbone"""

    def __init__(self, channel_num, dim=512):
        super(DeconvResnet, self).__init__()
        self.image_channels = channel_num
        self.in_channel = dim
        self.in_size = 4
        self.in_feature_dim = dim * 4 * 4
        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),

            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),

            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),

            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, channel_num, 4, 2, 1)
        )

    def forward(self, x):
        x = self.decoder(x)
        x = torch.sigmoid(x)

        return x
