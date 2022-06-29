import os

from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

root_path = os.path.dirname(__file__)


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]
    indices = []
    for idx, data in enumerate(dataset):
        if data[1] in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


def get_dataset(dataset, train_transform, test_transform, download=False, seen=None):
    """Get datasets for setting 1 (OOD Detection on the Same Dataset)."""

    if dataset == 'cifar10':
        DATA_PATH = os.path.join(root_path, 'cifar10')
        class_idx = [int(num) for num in seen]
        for i in range(10):
            if i in class_idx:
                continue
            class_idx.append(i)
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform,
                                     target_transform=lambda x: class_idx.index(x))
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform,
                                    target_transform=lambda x: class_idx.index(x))
        seen_class_idx = [0, 1, 2, 3, 4, 5]
        unseen_class_idx = [6, 7, 8, 9]

        train_set = get_subclass_dataset(train_set, seen_class_idx)
        test_set_seen = get_subclass_dataset(test_set, seen_class_idx)
        test_set_unseen = get_subclass_dataset(test_set, unseen_class_idx)

    elif dataset == 'mnist':
        DATA_PATH = os.path.join(root_path, 'mnist')
        class_idx = [int(num) for num in seen]
        for i in range(10):
            if i in class_idx:
                continue
            class_idx.append(i)
        train_set = datasets.MNIST(DATA_PATH, train=True, download=download, transform=train_transform,
                                   target_transform=lambda x: class_idx.index(x))
        test_set = datasets.MNIST(DATA_PATH, train=False, download=download, transform=test_transform,
                                  target_transform=lambda x: class_idx.index(x))
        seen_class_idx = [0, 1, 2, 3, 4, 5]
        unseen_class_idx = [6, 7, 8, 9]

        train_set = get_subclass_dataset(train_set, seen_class_idx)
        test_set_seen = get_subclass_dataset(test_set, seen_class_idx)
        test_set_unseen = get_subclass_dataset(test_set, unseen_class_idx)

    else:
        raise NotImplementedError

    return train_set, test_set_seen, test_set_unseen


def get_ood_dataset(dataset):
    """Get datasets for setting 2 (OOD Detection on Different Datasets)."""

    if dataset == 'SVHN':
        dir = os.path.join(root_path, 'svhn')
        data = datasets.SVHN(root=dir, split='test', download=True, transform=transforms.ToTensor())
    elif dataset == 'LSUN':
        dir = os.path.join(root_path, 'LSUN_resize')
        data = datasets.ImageFolder(dir, transform=transforms.ToTensor())
    elif dataset == 'tinyImageNet':
        dir = os.path.join(root_path, 'Imagenet_resize')
        data = datasets.ImageFolder(dir, transform=transforms.ToTensor())
    elif dataset == 'LSUN-FIX':
        dir = os.path.join(root_path, 'LSUN_fix')
        data = datasets.ImageFolder(dir, transform=transforms.ToTensor())
    elif dataset == 'ImageNet-FIX':
        dir = os.path.join(root_path, 'Imagenet_fix')
        data = datasets.ImageFolder(dir, transform=transforms.ToTensor())
    elif dataset == 'CIFAR100':
        dir = os.path.join(root_path, 'cifar100')
        data = datasets.CIFAR100(
            root=dir, train=False, transform=transforms.ToTensor(), download=True)
    else:
        raise NotImplementedError

    return data
