import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import CSI_model.classifier as C
from models.vae import ConditionalVAE2
from utils import get_args

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def energy_result(fx, y):
    """Calculates roc_auc using energy score.

    Args:
        fx: Last layer output of the model.
        y: Class Label, assumes the label of unseen data to be -1.
    Returns:
        roc_auc: Unseen data as positive, seen data as negative.
    """
    energy_score = - torch.logsumexp(fx, dim=1)
    rocauc = roc_auc_score((y == -1).cpu().detach().numpy(), energy_score.cpu().detach().numpy())

    return rocauc


# Set up seed -----------------------------------------------------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(222)

# Define hyper parameters and model -------------------------------------------
args = get_args()
batch_size = 128
n_epochs = 15
LR = 0.0001
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(device)

classifier = C.get_classifier('resnet18', n_classes=10).to(device)
checkpoint = torch.load(args.params_dict_name)
classifier.load_state_dict(checkpoint, strict=False)

# freeze the encoder part
for p in classifier.layer1.parameters():
    p.requires_grad = False
for p in classifier.layer2.parameters():
    p.requires_grad = False
for p in classifier.layer3.parameters():
    p.requires_grad = False
for p in classifier.layer4.parameters():
    p.requires_grad = False
for p in classifier.conv1.parameters():
    p.requires_grad = False

# load the CVAE model
vae = ConditionalVAE2()
vae.load_state_dict(torch.load(args.params_dict_name2, map_location='cpu'))
vae.to(device)
vae.eval()
for p in vae.parameters():
    p.requires_grad = False


def generate_pseudo_data(vae):
    scalar = 5.0
    neg_item_per_batch = 128

    # prepare for class embedding
    y1 = torch.Tensor(neg_item_per_batch, vae.class_num)
    y1.zero_()
    y2 = torch.Tensor(neg_item_per_batch, vae.class_num)
    y2.zero_()
    ind = torch.randint(0, 10, (neg_item_per_batch, 1))
    ind2 = torch.randint(0, 10, (neg_item_per_batch, 1))
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


def get_m(train_loader, classifier, vae):
    print("=====get_m======")
    with torch.no_grad():
        Ec_out, Ec_in = None, None
        for data, target in train_loader:
            classifier.eval()
            classifier.linear.train()

            data = data.to(device)
            prediction = classifier(data)

            x_generate = generate_pseudo_data(vae)

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


def tune_main_model():
    """
    seen: cifar10
    unseen: SVHN / LSUN / ImagenNet / LSUN(FIX) / ImageNet(FIX) / CIFAR100.
    """

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=LR)

    # prepare data ------------------------------------------------------------
    transform_train = transforms.ToTensor()

    train_data = datasets.CIFAR10(
        root='./data/cifar10', train=True, download=True,
        transform=transform_train)
    test_data = datasets.CIFAR10(
        root='./data/cifar10', train=False, download=True,
        transform=transforms.ToTensor())

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader_seen = DataLoader(test_data, batch_size=batch_size, num_workers=4)
    test_dir_svhn = os.path.join('./data', 'svhn')
    svhn_data = datasets.SVHN(
        root=test_dir_svhn, split='test', download=True,
        transform=transforms.ToTensor())
    test_loader_svhn = DataLoader(svhn_data, batch_size=512, num_workers=4)
    test_dir_lsun = os.path.join('./data', 'LSUN_resize')
    lsun_data = datasets.ImageFolder(test_dir_lsun, transform=transforms.ToTensor())
    test_loader_lsun = DataLoader(lsun_data, batch_size=512, num_workers=4)
    test_dir_imagenet = os.path.join('./data', 'Imagenet_resize')
    imagenet_data = datasets.ImageFolder(test_dir_imagenet, transform=transforms.ToTensor())
    test_loader_imagenet = DataLoader(imagenet_data, batch_size=512, num_workers=4)
    test_dir_lsun_fix = os.path.join('./data', 'LSUN_fix')
    lsun_fix_data = datasets.ImageFolder(test_dir_lsun_fix, transform=transforms.ToTensor())
    test_loader_lsun_fix = DataLoader(lsun_fix_data, batch_size=512, num_workers=4)
    test_dir_imagenet_fix = os.path.join('./data', 'Imagenet_fix')
    imagenet_data = datasets.ImageFolder(test_dir_imagenet_fix, transform=transforms.ToTensor())
    test_loader_imagenet_fix = DataLoader(imagenet_data, batch_size=512, num_workers=4)
    test_dir_cifar100 = os.path.join('./data', 'cifar100')
    cifar100_data = datasets.CIFAR100(
        root=test_dir_cifar100, train=False, transform=transforms.ToTensor(), download=True)
    test_loader_cifar100 = DataLoader(cifar100_data, batch_size=512, num_workers=4)

    # hyper params for energy loss
    mu = 0.1
    m_in, m_out = get_m(train_loader, classifier, vae)

    # fine-tuning the classification head ----------------------------------------------------
    index = -1
    max_roc_auc = {'svhn': 0, 'lsun': 0, 'imagenet': 0, 'lsun_fix': 0, 'imagenet_fix': 0, 'cifar100': 0}

    for epoch in range(n_epochs):

        for data, target in train_loader:

            classifier.eval()
            classifier.linear.train()
            optimizer.zero_grad()
            index += 1

            data = data.to(device)
            target = target.long().to(device)
            prediction = classifier(data)
            loss_ce = F.cross_entropy(prediction, target)
            Ec_in = -torch.logsumexp(prediction, dim=1)

            x_generate = generate_pseudo_data(vae)

            prediction_generate = classifier(x_generate)[:, 0:10]

            Ec_out = -torch.logsumexp(prediction_generate, dim=1)

            # energy loss
            loss_energy = torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean()

            loss = loss_ce + mu * loss_energy

            loss.backward()
            optimizer.step()

            # evaluate (every 100 batches) ------------------------------------
            if index % 100 == 0:
                classifier.eval()
                with torch.no_grad():

                    output_ind = []
                    output_svhn, output_lsun, output_imagenet, output_lsun_fix, output_imagenet_fix, output_cifar100, \
                        = [], [], [], [], [], []

                    for x, _ in test_loader_seen:
                        x = x.to(device)
                        output = classifier(x)
                        output_ind.append(output)

                    for x, _ in test_loader_svhn:
                        x = x.to(device)
                        output = classifier(x)
                        output_svhn.append(output)

                    for x, _ in test_loader_lsun:
                        x = x.to(device)
                        output = classifier(x)
                        output_lsun.append(output)

                    for x, _ in test_loader_imagenet:
                        x = x.to(device)
                        output = classifier(x)
                        output_imagenet.append(output)

                    for x, _ in test_loader_lsun_fix:
                        x = x.to(device)
                        output = classifier(x)
                        output_lsun_fix.append(output)

                    for x, _ in test_loader_imagenet_fix:
                        x = x.to(device)
                        output = classifier(x)
                        output_imagenet_fix.append(output)

                    for x, _ in test_loader_cifar100:
                        x = x.to(device)
                        output = classifier(x)
                        output_cifar100.append(output)

                    output_ind = torch.cat(output_ind, 0)
                    output_svhn = torch.cat(output_svhn, 0)
                    output_lsun = torch.cat(output_lsun, 0)
                    output_imagenet = torch.cat(output_imagenet, 0)
                    output_lsun_fix = torch.cat(output_lsun_fix, 0)
                    output_imagenet_fix = torch.cat(output_imagenet_fix, 0)
                    output_cifar100 = torch.cat(output_cifar100, 0)

                    roc_auc_svhn = energy_result(torch.cat([output_ind, output_svhn]), torch.cat(
                        [torch.ones(output_ind.size(0)), -torch.ones(output_svhn.size(0))]).long().to(device))
                    roc_auc_lsun = energy_result(torch.cat([output_ind, output_lsun]), torch.cat(
                        [torch.ones(output_ind.size(0)), -torch.ones(output_lsun.size(0))]).long().to(device))
                    roc_auc_imagenet = energy_result(torch.cat([output_ind, output_imagenet]), torch.cat(
                        [torch.ones(output_ind.size(0)), -torch.ones(output_imagenet.size(0))]).long().to(device))
                    roc_auc_lsun_fix = energy_result(torch.cat([output_ind, output_lsun_fix]), torch.cat(
                        [torch.ones(output_ind.size(0)), -torch.ones(output_lsun_fix.size(0))]).long().to(device))
                    roc_auc_imagenet_fix = energy_result(torch.cat([output_ind, output_imagenet_fix]), torch.cat(
                        [torch.ones(output_ind.size(0)), -torch.ones(output_imagenet_fix.size(0))]).long().to(device))
                    roc_auc_cifar100 = energy_result(torch.cat([output_ind, output_cifar100]), torch.cat(
                        [torch.ones(output_ind.size(0)), -torch.ones(output_cifar100.size(0))]).long().to(device))

                    max_roc_auc['svhn'] = max(max_roc_auc['svhn'], roc_auc_svhn)
                    max_roc_auc['lsun'] = max(max_roc_auc['lsun'], roc_auc_lsun)
                    max_roc_auc['imagenet'] = max(max_roc_auc['imagenet'], roc_auc_imagenet)
                    max_roc_auc['lsun_fix'] = max(max_roc_auc['lsun_fix'], roc_auc_lsun_fix)
                    max_roc_auc['imagenet_fix'] = max(max_roc_auc['imagenet_fix'], roc_auc_imagenet_fix)
                    max_roc_auc['cifar100'] = max(max_roc_auc['cifar100'], roc_auc_cifar100)

                    print('Epoch: {}  Index: {}'.format(epoch, index))
                    print('Max rocauc result')
                    print(max_roc_auc)


if __name__ == '__main__':
    tune_main_model()
