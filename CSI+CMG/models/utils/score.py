import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def softmax_result(fx, y):
    """Calculates roc_auc using softmax score of the unseen slot.

    Args:
        fx: Last layer output of the model, assumes the unseen slot to be the last one.
        y: Class Label, assumes the label of unseen data to be -1.
    Returns:
        roc_auc: Unseen data as positive, seen data as negative.
    """
    score = F.softmax(fx, dim=1)[:, -1]
    rocauc = roc_auc_score((y == -1).cpu().detach().numpy(), score.cpu().detach().numpy())

    return rocauc


def energy_result(fx, y):
    """Calculates roc_auc using energy score.

    Args:
        fx: Last layer output of the model, assumes the unseen slot to be the last one.
        y: Class Label, assumes the label of unseen data to be -1.
    Returns:
        roc_auc: Unseen data as positive, seen data as negative.
    """
    energy_score = - torch.logsumexp(fx[:, :-1], dim=1)
    rocauc = roc_auc_score((y == -1).cpu().detach().numpy(), energy_score.cpu().detach().numpy())

    return rocauc
