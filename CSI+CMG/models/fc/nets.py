import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """
    Module for a multi-layer perceptron (MLP).

    input: <2D tensor> [batch_size] * [input_dim]
    output: <2D tensor> [batch_size] * [classes]

    """

    def __init__(self, input_dim, classes, latent_dim=512):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.classes = classes
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, classes)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        result = self.fc2(x)

        return result
