"""CMG Stage 2: Fine-tuning the classification head using IND data and pseudo-OOD data generated by the CVAE."""
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from datasets.datasetsHelper import get_dataset, get_ood_dataset
from models.main_model import MainModel
from models.vae import ConditionalVAE
from training.fine_tune import fine_tune_same_dataset, fine_tune_different_dataset
from utils import get_args

args = get_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)

batch_size = 128

# gpu
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(device)


def main():
    if args.task == 'same_dataset_mnist':
        # prepare data
        if args.partition == 'partition1':
            train_data, test_data_seen, test_data_unseen = get_dataset('mnist', transforms.ToTensor(),
                                                                       transforms.ToTensor(), seen='012345')
        elif args.partition == 'partition2':
            train_data, test_data_seen, test_data_unseen = get_dataset('mnist', transforms.ToTensor(),
                                                                       transforms.ToTensor(), seen='123456')
        elif args.partition == 'partition3':
            train_data, test_data_seen, test_data_unseen = get_dataset('mnist', transforms.ToTensor(),
                                                                       transforms.ToTensor(), seen='234567')
        elif args.partition == 'partition4':
            train_data, test_data_seen, test_data_unseen = get_dataset('mnist', transforms.ToTensor(),
                                                                       transforms.ToTensor(), seen='345678')
        elif args.partition == 'partition5':
            train_data, test_data_seen, test_data_unseen = get_dataset('mnist', transforms.ToTensor(),
                                                                       transforms.ToTensor(), seen='456789')
        else:
            raise NotImplementedError

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
        test_loader_seen = DataLoader(test_data_seen, batch_size=512, num_workers=8)
        test_loader_unseen = DataLoader(test_data_unseen, batch_size=512, num_workers=8)

        # prepare model
        classifier = MainModel(28, 1, 11, dataset='mnist')
        classifier.load_state_dict(torch.load(args.params_dict_name, map_location='cpu'))
        classifier.to(device)
        vae = ConditionalVAE(image_channels=1, image_size=28, dataset='mnist')
        vae.load_state_dict(torch.load(args.params_dict_name2, map_location='cpu'))
        vae.to(device)
        vae.device = device

        # CMG Stage 2
        result = fine_tune_same_dataset(
            classifier, vae, train_loader, test_loader_seen, test_loader_unseen, device, dataset='mnist',
            mode=args.mode)

        print('{}, same dataset mnist {}: max roc auc = {}'.format(args.mode, args.partition, result))

    elif args.task == 'same_dataset_cifar10':
        # prepare data
        if args.partition == 'partition1':
            train_data, test_data_seen, test_data_unseen = get_dataset('cifar10', transforms.ToTensor(),
                                                                       transforms.ToTensor(), seen='012345')
        elif args.partition == 'partition2':
            train_data, test_data_seen, test_data_unseen = get_dataset('cifar10', transforms.ToTensor(),
                                                                       transforms.ToTensor(), seen='123456')
        elif args.partition == 'partition3':
            train_data, test_data_seen, test_data_unseen = get_dataset('cifar10', transforms.ToTensor(),
                                                                       transforms.ToTensor(), seen='234567')
        elif args.partition == 'partition4':
            train_data, test_data_seen, test_data_unseen = get_dataset('cifar10', transforms.ToTensor(),
                                                                       transforms.ToTensor(), seen='345678')
        elif args.partition == 'partition5':
            train_data, test_data_seen, test_data_unseen = get_dataset('cifar10', transforms.ToTensor(),
                                                                       transforms.ToTensor(), seen='456789')
        else:
            raise NotImplementedError

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
        test_loader_seen = DataLoader(test_data_seen, batch_size=512, num_workers=8)
        test_loader_unseen = DataLoader(test_data_unseen, batch_size=512, num_workers=8)

        # prepare model
        classifier = MainModel(32, 3, 11, dataset='cifar10')
        classifier.load_state_dict(torch.load(args.params_dict_name, map_location='cpu'))
        classifier.to(device)
        vae = ConditionalVAE(image_channels=3, image_size=32, dataset='cifar10')
        vae.load_state_dict(torch.load(args.params_dict_name2, map_location='cpu'))
        vae.to(device)
        vae.device = device

        # CMG Stage 2
        result = fine_tune_same_dataset(
            classifier, vae, train_loader, test_loader_seen, test_loader_unseen, device, dataset='cifar10',
            mode=args.mode)

        print('{}, same dataset cifar10 {}: max roc auc = {}'.format(args.mode, args.partition, result))

    elif args.task == 'different_dataset':
        # prepare data
        root_path = os.path.dirname(__file__)
        data_path = os.path.join(root_path, 'datasets/cifar10')
        train_data = datasets.CIFAR10(root=data_path, train=True, transform=transforms.ToTensor())
        test_data_seen = datasets.CIFAR10(root=data_path, train=False, transform=transforms.ToTensor())
        test_data_unseen = get_ood_dataset(args.ood_dataset)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
        test_loader_seen = DataLoader(test_data_seen, batch_size=512, num_workers=8)
        test_loader_unseen = DataLoader(test_data_unseen, batch_size=512, num_workers=8)

        # prepare model
        classifier = MainModel(32, 3, 110, dataset='cifar10')
        classifier.load_state_dict(torch.load(args.params_dict_name, map_location='cpu'))
        classifier.to(device)
        vae = ConditionalVAE(image_channels=3, image_size=32, dataset='cifar10')
        vae.load_state_dict(torch.load(args.params_dict_name2, map_location='cpu'))
        vae.to(device)
        vae.device = device

        # CMG Stage 2
        result = fine_tune_different_dataset(
            classifier, vae, train_loader, test_loader_seen, test_loader_unseen, device, mode=args.mode)

        print('{}, different dataset, ood dataset {}: max roc auc = {}'.format(args.mode, args.ood_dataset, result))

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
