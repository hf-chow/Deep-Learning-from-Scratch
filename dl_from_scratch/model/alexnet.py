from dl_from_scratch.utils.plot import acc_loss_plot

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Implementation
# Technically this is not a "pure" AlexNet since the original paper has error in the Conv2D dimensions
# We follow the "one werid trick" paper's implementation instead
# https://arxiv.org/abs/1404.5997
# https://github.com/soumith/imagenet-multiGPU.torch/blob/master/models/alexnetowt.lua
# Architeture:
# - Conv2D 3
# - LocalNorm
# - MaxPool
# - Conv2D 48
# - LocalNorm
# - MaxPool
# - Conv2D 128
# - Conv2D 192
# - Conv2D 192
# - MaxPool
# - Linear 4096
# - Dropout 0.5
# - Linear 4096
# - Dropout 0.5
# - Linear 1000
# -- Relu after every layer

# Augmentation
# Horizontal relects
# 224 x 224 random cropping
# Normalizing RGB

device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
        )

def _layer_checker():
    # Helper function to check the layers dimensions 
    pass

def _get_output_size(input_size, kernel_size, stride, padding, dilation=1):
    # Helper function to calculate the out_channels of each conv layers
    # This function assumes padding and stride are both single dimensional
    # and the input image is square
    output_size = math.floor((input_size[0] + 2*padding - dilation*(kernel_size[0] - 1) - 1) / stride) + 1
    return output_size


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(in_channels=3, 
                          out_channels=64,      # Original paper does not yield the correct out_channels due to its implementation for dual GPUs
                          kernel_size=(11, 11), 
                          stride=4,
                          padding=(2,2)
                          ),
                nn.ReLu(inplace=True),  # Inplace preserve a little bit more memory
                nn.LocalResponseNorm(size=5, k=2),
                nn.MaxPool2d(kernel_size=(3,3), stride=2), #Section 3.4, the paper opt for overlapping maxpool
                nn.Conv2d(in_channels=64, 
                          out_channels=192, 
                          kernel_size=(5,5),
                          stride=1,
                          padding=(2,2)),
                nn.ReLu(inplace=True),
                nn.LocalResponseNorm(size=5, k=2),
                nn.MaxPool2d(kernel_size=(3,3), stride=2),
                nn.Conv2d(in_channels=192, 
                          out_channels=384, 
                          kernel_size=(3,3),
                          stride=1,
                          padding=(1,1)),
                nn.ReLu(inplace=True),
                nn.Conv2d(in_channels=384, 
                          out_channels=256, 
                          kernel_size=(3,3),
                          stride=1,
                          padding=(1,1)),
                nn.ReLu(inplace=True),
                nn.Conv2d(in_channels=256, 
                          out_channels=256, 
                          kernel_size=(3,3),
                          stride=1,
                          padding=(1,1)),
                nn.ReLu(inplace=True),
                )
        self.maxpool = nn.MaxPool2d(kernel_size=(3,3), stride=2)
        self.classifer = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=256*6*6, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features=4096, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=4096, out_features=1000),
                )

        def forward(self, x):
            x = self.features(x)
            x = self.maxpool(x)
            logits = self.classifer(x)
            return logits

if __name__ == "__main__":
    PATH = "../../data/ImageNet/"

    out = _get_output_size((224,224), (11,11), padding=2, stride=4)
    print(f"Conv1: {out} x {out}")
    out = _get_output_size((out,out), (3,3), padding=0, stride=2)
    print(f"MaxPool1: {out} x {out}")

    out = _get_output_size((out,out), (5,5), padding=2, stride=1)
    print(f"Conv2: {out} x {out}")
    out = _get_output_size((out,out), (3,3), padding=0, stride=2)
    print(f"MaxPool2: {out} x {out}")

    out = _get_output_size((out,out), (3,3), padding=1, stride=1)
    print(f"Conv3: {out} x {out}")

    out = _get_output_size((out,out), (3,3), padding=1, stride=1)
    print(f"Conv4: {out} x {out}")

    out = _get_output_size((out,out), (3,3), padding=1, stride=1)
    print(f"Conv5: {out} x {out}")

    out = _get_output_size((out,out), (3,3), padding=0, stride=2)
    print(f"MaxPool3: {out} x {out}")
