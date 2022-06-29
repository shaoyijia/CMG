import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv.nets import ConvLayers, DeconvLayers, ResNet18, DeconvResnet
from models.utils import modules


class ConditionalVAE(nn.Module):
    """Variational Auto-Encoder with class conditional information"""

    def __init__(self, z_dim=32, image_channels=1, image_size=28, class_num=10, dataset='mnist'):
        super(ConditionalVAE, self).__init__()
        self.class_num = class_num
        self.image_size = image_size
        self.image_channels = image_channels

        if dataset == 'mnist':
            self.convE = ConvLayers(image_channels)
            self.convD = DeconvLayers(image_channels)
        elif dataset == 'cifar10':
            self.convE = ResNet18()
            self.convD = DeconvResnet(image_channels)
        else:
            raise NotImplementedError
        self.flatten = modules.Flatten()
        self.fcE = nn.Linear(self.convE.out_feature_dim, 1024)
        self.z_dim = z_dim
        self.fcE_mean = nn.Linear(1024, self.z_dim)
        self.fcE_logvar = nn.Linear(1024, self.z_dim)
        self.fromZ = nn.Linear(2 * self.z_dim, 1024)
        self.fcD = nn.Linear(1024, self.convD.in_feature_dim)
        self.to_image = modules.Reshape(image_channels=self.convE.out_channels)
        self.device = None

        self.class_embed = nn.Linear(class_num, self.z_dim)

    def encode(self, x):
        hidden_x = self.convE(x)
        feature = self.flatten(hidden_x)

        hE = F.relu(self.fcE(feature))

        z_mean = self.fcE_mean(hE)
        z_logvar = self.fcE_logvar(hE)

        return z_mean, z_logvar, hE, hidden_x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(self.device)
        z = torch.randn(std.size()).to(self.device) * std + mu.to(self.device)
        return z

    def decode(self, z, y_embed):
        z = torch.cat([z, y_embed], dim=1)  # add label information
        hD = F.relu(self.fromZ(z))
        feature = self.fcD(hD)
        image_recon = self.convD(feature.view(-1, self.convD.in_channel, self.convD.in_size, self.convD.in_size))

        return image_recon

    def forward(self, x, y_tensor):
        mu, logvar, hE, hidden_x = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y_embed = self.class_embed(y_tensor)
        x_recon = self.decode(z, y_embed)
        return mu, logvar, x_recon
