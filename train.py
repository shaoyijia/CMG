"""CMG Stage 1: IND classifier building & CVAE training."""
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from datasets.datasetsHelper import get_dataset
from models.main_model import MainModel
from models.vae import ConditionalVAE
from training.train_classifier import train_classifier
from training.train_cvae import train_cvae
from utils import get_args

args = get_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)

batch_size = 512

# gpu
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(device)


def main():
    # prepare data
    if args.task == 'same_dataset_mnist':
        if args.partition == 'partition1':
            train_data, _, _ = get_dataset('mnist', transforms.ToTensor(), transforms.ToTensor(), seen='012345')
        elif args.partition == 'partition2':
            train_data, _, _ = get_dataset('mnist', transforms.ToTensor(), transforms.ToTensor(), seen='123456')
        elif args.partition == 'partition3':
            train_data, _, _ = get_dataset('mnist', transforms.ToTensor(), transforms.ToTensor(), seen='234567')
        elif args.partition == 'partition4':
            train_data, _, _ = get_dataset('mnist', transforms.ToTensor(), transforms.ToTensor(), seen='345678')
        elif args.partition == 'partition5':
            train_data, _, _ = get_dataset('mnist', transforms.ToTensor(), transforms.ToTensor(), seen='456789')
        else:
            raise NotImplementedError
    elif args.task == 'same_dataset_cifar10':
        if args.command == 'train_classifier':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif args.command == 'train_cvae':
            train_transform = transforms.ToTensor()

        if args.partition == 'partition1':
            train_data, _, _ = get_dataset('cifar10', train_transform=train_transform,
                                           test_transform=transforms.ToTensor(), seen='012345')
        elif args.partition == 'partition2':
            train_data, _, _ = get_dataset('cifar10', train_transform=train_transform,
                                           test_transform=transforms.ToTensor(), seen='123456')
        elif args.partition == 'partition3':
            train_data, _, _ = get_dataset('cifar10', train_transform=train_transform,
                                           test_transform=transforms.ToTensor(), seen='234567')
        elif args.partition == 'partition4':
            train_data, _, _ = get_dataset('cifar10', train_transform=train_transform,
                                           test_transform=transforms.ToTensor(), seen='345678')
        elif args.partition == 'partition5':
            train_data, _, _ = get_dataset('cifar10', train_transform=train_transform,
                                           test_transform=transforms.ToTensor(), seen='456789')
        else:
            raise NotImplementedError
    elif args.task == 'different_dataset':
        if args.command == 'train_classifier':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif args.command == 'train_cvae':
            train_transform = transforms.ToTensor()

        root_path = os.path.dirname(__file__)
        data_path = os.path.join(root_path, 'datasets/cifar10')
        train_data = datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

    # CMG stage 1
    if args.command == 'train_classifier':
        if args.task == 'same_dataset_mnist':
            model = MainModel(28, 1, 11, dataset='mnist')
            model.to(device)
            train_classifier(model, train_loader, device, args.params_dict_name, dataset='mnist')
        elif args.task == 'same_dataset_cifar10':
            model = MainModel(32, 3, 11, dataset='cifar10')
            model.to(device)
            train_classifier(model, train_loader, device, args.params_dict_name, dataset='cifar10')
        elif args.task == 'different_dataset':
            model = MainModel(32, 3, 110, dataset='cifar10')
            model.to(device)
            train_classifier(model, train_loader, device, args.params_dict_name, dataset='cifar10')

    elif args.command == 'train_cvae':
        if args.task == 'same_dataset_mnist':
            model = ConditionalVAE(image_channels=1, image_size=28, dataset='mnist')
            model.device = device
            model.to(device)
            train_cvae(model, train_loader, device, args.params_dict_name, dataset='mnist')
        elif args.task == 'same_dataset_cifar10':
            model = ConditionalVAE(image_channels=3, image_size=32, dataset='cifar10')
            model.device = device
            model.to(device)
            train_cvae(model, train_loader, device, args.params_dict_name, dataset='cifar10')
        elif args.task == 'different_dataset':
            model = ConditionalVAE(image_channels=3, image_size=32, dataset='cifar10')
            model.device = device
            model.to(device)
            train_cvae(model, train_loader, device, args.params_dict_name, dataset='cifar10')

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
