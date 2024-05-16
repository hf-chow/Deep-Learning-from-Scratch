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
# - Linear 2048
# - Dropout 0.5
# - Linear 2048
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

def _get_output_size(input_size, kernel_size, padding, stride):
    # Helper function to calculate the out_channels for Conv2D
    # This function assumes padding and stride are both single dimensional
    # and the input image is square
    output_size = math.ceil((input_size[0] - kernel_size[0]+2*padding)/stride) + 1
    return output_size

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = nn.Sequential(
                nn.Conv2d(in_channels=3, 
                          out_channels=64,      # Original paper does not yield the correct out_channels due to its implementation for dual GPUs, this follows the "one weird trick papers" implementation
                          kernel_size=(11, 11), 
                          stride=4,
                          )
                )




if __name__ == "__main__":
    PATH = "../../data/ImageNet/"

    out = _get_output_size((224,224), (11,11), 2, 4)
    print(out)
    
