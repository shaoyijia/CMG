import torch

from models.utils import loss_functions as lf


def train_cvae(model, train_loader, device, params_dict_name, dataset='mnist'):
    """Trains CVAE on the given dataset."""

    if dataset == 'mnist':
        n_epochs = 100
    elif dataset == 'cifar10':
        n_epochs = 200
    else:
        raise ValueError
    LR = 0.001

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    for epoch in range(n_epochs):
        for data, y in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            y = y.long()
            y_onehot = torch.Tensor(y.shape[0], model.class_num)
            y_onehot.zero_()
            y_onehot.scatter_(1, y.view(-1, 1), 1)
            y_onehot = y_onehot.to(device)
            mu, logvar, recon = model(data, y_onehot)

            variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            variatL = lf.weighted_average(variatL, weights=None, dim=0)
            variatL /= (model.image_channels * model.image_size * model.image_size)

            data_resize = data.reshape(-1, model.image_channels * model.image_size * model.image_size)
            recon_resize = recon.reshape(-1, model.image_channels * model.image_size * model.image_size)
            reconL = (data_resize - recon_resize) ** 2
            reconL = torch.mean(reconL, 1)
            reconL = lf.weighted_average(reconL, weights=None, dim=0)

            loss = variatL + reconL

            loss.backward()
            optimizer.step()

        print("epoch: {}, loss = {}, reconL = {}, variaL = {}".format(epoch, loss, reconL, variatL))

    torch.save(model.state_dict(), params_dict_name)
