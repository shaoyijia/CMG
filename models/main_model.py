import torch.nn as nn
import torch.nn.functional as F

from models.conv.cnn import ConvEncoder
from models.conv.nets import ResNet18
from models.fc.nets import MLP
from models.utils import loss_functions as lf
from models.utils import modules


class MainModel(nn.Module):
    """
    feature_extractor(CNN) -> classifier (MLP)
    """

    def __init__(self, image_size, image_channels, classes, dataset='mnist'):
        super(MainModel, self).__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes

        # for encoder
        if dataset == 'mnist':
            self.convE = ConvEncoder(image_channels=image_channels)
        elif dataset == 'cifar10':
            self.convE = ResNet18()
        else:
            raise NotImplementedError
        self.flatten = modules.Flatten()

        # classifier
        self.classifier = MLP(self.convE.out_feature_dim, classes)

        self.optimizer = None  # needs to be set before training starts

        self.device = None  # needs to be set before using the model

    # --------- FROWARD FUNCTIONS ---------#
    def encode(self, x):
        """
        pass input through feed-forward connections to get [image_features]
        """
        # Forward-pass through conv-layers
        hidden_x = self.convE(x)

        return hidden_x

    def classify(self, x):
        """
        For input [x] (image or extracted "internal“ image features),
        return predicted scores (<2D tensor> [batch_size] * [classes])
        """
        result = self.classifier(x)
        return result

    def forward(self, x):
        """
        Forward function to propagate [x] through the encoder and the classifier.
        """
        hidden_x = self.encode(x)
        prediction = self.classifier(hidden_x)
        return prediction

    # ------------------TRAINING FUNCTIONS----------------------#
    def train_a_batch(self, x, y):
        """
        Train model for one batch ([x], [y])
        """
        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        # Run the model
        hidden_x = self.encode(x)
        prediction = self.classifier(hidden_x)
        predL = F.cross_entropy(prediction, y, reduction='none')
        loss = lf.weighted_average(predL, weights=None, dim=0)

        loss.backward()

        self.optimizer.step()

        return loss.item()
