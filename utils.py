import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='same_dataset_mnist',
                        choices=['same_dataset_mnist', 'same_dataset_cifar10', 'different_dataset'],
                        help='the current task: same_dataset_mnist/same_dataset_cifar10/different_dataset')
    parser.add_argument('--partition', type=str, default='partition1')
    parser.add_argument('--command', type=str, default='train_classifier',
                        choices=['train_classifier', 'train_cvae'],
                        help='command for CMG stage 1: train_classifier/train_cvae')
    parser.add_argument('--ood-dataset', type=str,
                        choices=['SVHN', 'LSUN', 'LSUN-FIX', 'tinyImageNet', 'ImageNet-FIX', 'CIFAR100'],
                        help='OOD dataset for setting 2: SVHN/LSUN/LSUN-FIX/tinyImageNet/ImageNet-FIX/CIFAR100')
    parser.add_argument('--mode', type=str, default='CMG-energy', choices=['CMG-softmax', 'CMG-energy'],
                        help="CMG-softmax/CMG-energy")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for training')
    parser.add_argument('--params-dict-name', type=str,
                        help='name of the classifier checkpoint file')
    parser.add_argument('--params-dict-name2', type=str,
                        help='name of the CVAE checkpoint file')
    parser.add_argument('--seed', type=int, default=123, help='set random seed')
    return parser.parse_args()
