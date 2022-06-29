import torch


def weighted_average(tensor, weights=None, dim=0):
    """Computes weighted average of [tensor] over dimension [dim]."""
    if weights is None:
        mean = torch.mean(tensor, dim=dim)
    else:
        batch_size = tensor.size(dim) if len(tensor.size()) > 0 else 1
        assert len(weights) == batch_size
        norm_weights = torch.tensor([weight for weight in weights]).to(tensor.device)
        mean = torch.mean(norm_weights * tensor, dim=dim)
    return mean
