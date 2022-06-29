import torch
import torch.nn.functional as F

from models.utils import loss_functions as lf
from models.utils import score


def generate_pseudo_data(vae, class_num, scalar, neg_item_per_batch, device):
    """Generates pseudo data for CMG stage 2."""

    # prepare for class embedding
    y1 = torch.Tensor(neg_item_per_batch, vae.class_num)
    y1.zero_()
    y2 = torch.Tensor(neg_item_per_batch, vae.class_num)
    y2.zero_()
    ind = torch.randint(0, class_num, (neg_item_per_batch, 1))
    ind2 = torch.randint(0, class_num, (neg_item_per_batch, 1))
    y1.scatter_(1, ind.view(-1, 1), 1)
    y2.scatter_(1, ind2.view(-1, 1), 1)
    y1 = y1.to(device)
    y2 = y2.to(device)
    class_embed1 = vae.class_embed(y1)
    class_embed2 = vae.class_embed(y2)
    rv = torch.randint(0, 2, class_embed1.shape).to(device)
    class_embed = torch.where(rv == 0, class_embed1, class_embed2)

    # sample in N(0, sigma^2)
    random_z = torch.randn(neg_item_per_batch, vae.z_dim).to(device) * scalar

    x_generate = vae.decode(random_z, class_embed)

    return x_generate


def get_m(train_loader, classifier, vae, class_num, scalar, device):
    """Gets hyper-parameters for CMG-energy using training data."""

    print("=====get_m======")
    with torch.no_grad():
        Ec_out, Ec_in = None, None
        for data, target in train_loader:
            classifier.eval()
            classifier.classifier.train()

            data = data.to(device)
            prediction = classifier(data)

            x_generate = generate_pseudo_data(vae, class_num, scalar, 128, device)

            prediction_generate = classifier(x_generate)

            # calculate energy of the training data and generated negative data
            T = 1
            if Ec_in is None and Ec_out is None:
                Ec_in = -T * torch.logsumexp(prediction / T, dim=1)
                Ec_out = -T * torch.logsumexp(prediction_generate / T, dim=1)
            else:
                Ec_in = torch.cat((Ec_in, (-T * torch.logsumexp(prediction / T, dim=1))), dim=0)
                Ec_out = torch.cat((Ec_out, (-T * torch.logsumexp(prediction_generate / T, dim=1))), dim=0)

        Ec_in = Ec_in.sort()[0]
        Ec_out = Ec_out.sort()[0]
        in_size = Ec_in.size(0)
        out_size = Ec_out.size(0)
        m_in, m_out = Ec_in[int(in_size * 0.2)], Ec_out[int(out_size * 0.8)]
        print("m_in = ", m_in, ",m_out=", m_out)

        return m_in, m_out


def fine_tune_same_dataset(
        classifier, vae, train_loader, test_loader_seen, test_loader_unseen, device, dataset='mnist',
        mode='CMG-energy'):
    """
    Performs CMG stage2 by fine-tuning the OOD detector using generated pseudo data on Setting 1.
    """
    for p in classifier.convE.parameters():
        p.requires_grad = False

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    n_epochs = 15
    LR = 0.0001
    if dataset == 'mnist':
        scalar = 3.0
    elif dataset == 'cifar10':
        scalar = 5.0
    else:
        raise NotImplementedError
    neg_item_per_batch = 128
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=LR)

    if mode == 'CMG-energy':
        # define hyper-parameters for CMG-energy
        mu = 0.1
        m_in, m_out = get_m(train_loader, classifier, vae, 6, scalar, device)

    index = -1
    max_rocauc = 0
    for epoch in range(n_epochs):
        for data, target in train_loader:

            classifier.eval()
            classifier.classifier.train()
            optimizer.zero_grad()

            index += 1
            data = data.to(device)
            target = target.long().to(device)
            prediction = classifier(data)
            x_generate = generate_pseudo_data(vae, 6, scalar, neg_item_per_batch, device)
            prediction_generate = classifier(x_generate)

            if mode == 'CMG-softmax':
                loss_input = F.cross_entropy(prediction, target, reduction='none')
                loss_input = lf.weighted_average(loss_input, weights=None, dim=0)

                y_generate = torch.ones(neg_item_per_batch) * 6
                y_generate = y_generate.long().to(device)

                loss_generate = F.cross_entropy(prediction_generate[:, 0:7], y_generate)

                loss = loss_generate + loss_input

                loss.backward()
                optimizer.step()

            elif mode == 'CMG-energy':
                loss_ce = F.cross_entropy(prediction, target)

                Ec_in = - torch.logsumexp(prediction, dim=1)
                Ec_out = - torch.logsumexp(prediction_generate, dim=1)
                loss_energy = torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean()

                loss = loss_ce + mu * loss_energy

                loss.backward()
                optimizer.step()

            else:
                raise NotImplementedError

            # evaluate
            if index % 20 == 0:
                classifier.eval()
                with torch.no_grad():
                    fx = []
                    labels = []

                    for x, y in test_loader_seen:
                        x = x.to(device)
                        y = y.long().to(device)
                        output = classifier(x)

                        fx.append(output[:, 0:7])
                        labels.append(y)

                    for x, _ in test_loader_unseen:
                        x = x.to(device)
                        output = classifier(x)

                        fx.append(output[:, 0:7])
                        labels.append(-torch.ones(x.size(0)).long().to(device))

                    labels = torch.cat(labels, 0)
                    fx = torch.cat(fx, 0)

                    if mode == 'CMG-softmax':
                        roc_auc = score.softmax_result(fx, labels)
                    elif mode == 'CMG-energy':
                        roc_auc = score.energy_result(fx, labels)
                    else:
                        raise NotImplementedError

                    if roc_auc > max_rocauc:
                        max_rocauc = roc_auc

                    if mode == 'CMG-softmax':
                        print('Epoch:', epoch, 'Index:', index, 'loss input', loss_input.item(),
                              'loss neg', loss_generate.item())
                    elif mode == 'CMG-energy':
                        print('Epoch:', epoch, 'Index:', index, 'loss ce', loss_ce.item(),
                              'loss energy', loss_energy.item())
                    else:
                        raise NotImplementedError

                    print('max roc auc:', max_rocauc)

    return max_rocauc


def fine_tune_different_dataset(
        classifier, vae, train_loader, test_loader_seen, test_loader_unseen, device, mode='CMG-energy'):
    """
    Performs CMG stage2 by fine-tuning the OOD detector using generated pseudo data on Setting 2.
    """
    for p in classifier.convE.parameters():
        p.requires_grad = False

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    n_epochs = 15
    LR = 0.0001
    scalar = 5.0
    neg_item_per_batch = 128
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=LR)

    if mode == 'CMG-energy':
        # define hyper-parameters for CMG-energy
        mu = 0.1
        m_in, m_out = get_m(train_loader, classifier, vae, 10, scalar, device)

    index = -1
    max_rocauc = 0
    for epoch in range(n_epochs):
        for data, target in train_loader:

            classifier.eval()
            classifier.classifier.train()
            optimizer.zero_grad()

            index += 1
            data = data.to(device)
            target = target.long().to(device)
            prediction = classifier(data)
            x_generate = generate_pseudo_data(vae, 10, scalar, neg_item_per_batch, device)
            prediction_generate = classifier(x_generate)

            if mode == 'CMG-softmax':
                loss_input = F.cross_entropy(prediction, target, reduction='none')
                loss_input = lf.weighted_average(loss_input, weights=None, dim=0)

                y_generate = torch.ones(neg_item_per_batch) * 10
                y_generate = y_generate.long().to(device)

                loss_generate = F.cross_entropy(prediction_generate[:, 0:11], y_generate)

                loss = loss_generate + loss_input

                loss.backward()
                optimizer.step()

            elif mode == 'CMG-energy':
                loss_ce = F.cross_entropy(prediction, target)

                Ec_in = - torch.logsumexp(prediction, dim=1)
                Ec_out = - torch.logsumexp(prediction_generate, dim=1)
                loss_energy = torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean()

                loss = loss_ce + mu * loss_energy

                loss.backward()
                optimizer.step()

            else:
                raise NotImplementedError

            # evaluate
            if index % 100 == 0:
                classifier.eval()
                with torch.no_grad():
                    fx = []
                    labels = []

                    for x, y in test_loader_seen:
                        x = x.to(device)
                        y = y.long().to(device)
                        output = classifier(x)

                        fx.append(output[:, 0:11])
                        labels.append(y)

                    for x, _ in test_loader_unseen:
                        x = x.to(device)
                        output = classifier(x)

                        fx.append(output[:, 0:11])
                        labels.append(-torch.ones(x.size(0)).long().to(device))

                    labels = torch.cat(labels, 0)
                    fx = torch.cat(fx, 0)

                    if mode == 'CMG-softmax':
                        roc_auc = score.softmax_result(fx, labels)
                    elif mode == 'CMG-energy':
                        roc_auc = score.energy_result(fx, labels)
                    else:
                        raise NotImplementedError

                    if roc_auc > max_rocauc:
                        max_rocauc = roc_auc

                    if mode == 'CMG-softmax':
                        print('Epoch:', epoch, 'Index:', index, 'loss input', loss_input.item(),
                              'loss neg', loss_generate.item())
                    elif mode == 'CMG-energy':
                        print('Epoch:', epoch, 'Index:', index, 'loss ce', loss_ce.item(),
                              'loss energy', loss_energy.item())
                    else:
                        raise NotImplementedError

                    print('max roc auc:', max_rocauc)

    return max_rocauc
