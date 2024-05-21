from os.path import abspath, join

import torch
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

#TODO Refactor this into an object to reduce repetition

class Dataset:
    def __init__(self, save_path, transform_func=None):
        self.save_path = save_path
        self.transform_func = transform_func

    def train_test_split(self, _dataset, train_test_ratio=0.8, seed=64):
        generator = torch.Generator().manual_seed(self.seed)
        data = random_split(_dataset, train_test_ratio, generator=generator)
        train_data, test_data = data[0], data[1]

    def load_mnist(self, save_path, transform_func=None):
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

        train_data, test_data = self.train_test_split(caltech, train_test_split=train_test_split, seed=seed)
    
        return train_data, test_data

    def load_imagenette(save_path, download=True, train_test_split=0.8, seed=64, transform_func=None):
        if transform_func is None:
            train_data = datasets.Imagenette(
                    root=save_path,
                    split = "train",
                    size = "320px",
                    download=download,
                    transform=ToTensor()
                    )
            test_data = datasets.Imagenette(
                    root=save_path,
                    split = "test",
                    size = "320px",
                    download=download,
                    transform=ToTensor()
                    )

        else:
            train_data = datasets.Imagenette(
                    root=save_path,
                    split = "train",
                    size = "320px",
                    download=download,
                    transform=transform_func
                    )
            test_data = datasets.Imagenette(
                    root=save_path,
                    split = "val",
                    size = "320px",
                    download=download,
                    transform=transform_func
                    )

        return train_data, test_data

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
