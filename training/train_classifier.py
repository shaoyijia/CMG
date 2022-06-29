import torch


def train_classifier(model, train_loader, device, params_dict_name, dataset='mnist'):
    """Trains the IND classifier on the given dataset."""

    if dataset == 'mnist':
        n_epochs = 100
    elif dataset == 'cifar10':
        n_epochs = 200
    else:
        raise ValueError
    LR = 0.001
    model.optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    for epoch in range(n_epochs):
        for data, target in train_loader:
            data = data.to(device)
            target = target.long().to(device)
            train_loss = model.train_a_batch(data, target)

        print('Epoch: {}, loss = {}'.format(epoch, train_loss))

    torch.save(model.state_dict(), params_dict_name)
