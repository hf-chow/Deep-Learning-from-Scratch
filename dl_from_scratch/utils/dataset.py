from os.path import abspath, join

import torch
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

#TODO Refactor this into an object to reduce repetition
def load_mnist(save_path, transform_func=None):
    if transform_func is None:
        train_data = datasets.MNIST(
                        root=save_path,
                        download=True, 
                        train=True,
                        transform=ToTensor()
                               )
        test_data = datasets.MNIST(
                        root=save_path,
                        download=True,
                        train=False,
                        transform=ToTensor()
                        )
    else:
        train_data = datasets.MNIST(
                        root=save_path,
                        download=True, 
                        train=True,
                        transform=transform_func
                               )
        test_data = datasets.MNIST(
                        root=save_path,
                        download=True,
                        train=False,
                        transform=transform_func
                        )

    return train_data, test_data

def load_cifar100(save_path, transform_func=None):
    if transform_func is None:
        train_data = datasets.CIFAR100(
                root=save_path,
                download=True,
                train=True,
                transform=ToTensor()
                )

        test_data = datasets.CIFAR100(
                root=save_path,
                download=True,
                train=False,
                transform=ToTensor()
                )
    else:
        train_data = datasets.CIFAR100(
                root=save_path,
                download=True,
                train=True,
                transform=transform_func
                )

        test_data = datasets.CIFAR100(
                root=save_path,
                download=True,
                train=False,
                transform=transform_func
                )

    return train_data, test_data

def load_caltech101(save_path, train_test_split=0.8, seed=64, transform_func=None):
    if transform_func is None:
        caltech = datasets.Caltech101(
                root=save_path,
                download=True,
                transform=ToTensor()
                )
    else:
        caltech = datasets.Caltech101(
                root=save_path,
                download=True,
                transform=transform_func
                )
    generator = torch.Generator().manual_seed(seed)
    train_data = random_split(caltech, train_test_split, generator=generator)[0]
    test_data = random_split(caltech, train_test_split, generator=generator)[1]

    return train_data, test_data
        
#def load_imagenet(save_path):
#    train_data = datasets.ImageNet(
#            root=save_path,
#            download=True,
#            train=True,
#            transform=ToTensor()
#            )
#
#    test_data = datasets.ImageNet(
#            root=save_path,
#            download=True,
#            train=False,
#            transform=ToTensor()
#            )
#    return train_data, test_data
