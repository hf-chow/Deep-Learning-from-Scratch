import torch
import torch.nn as nn
import numpy as np


# TODO need to implement affine transform
class BatchNorm(nn.Module):
    def __init__(self, channels, eps = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps


    def forward(self, x):
        # We are usin the notation of the original paper
        batch_size = x.shape[0]
        original_shape = x.shape
        print(x_shape)
        x = x.reshape(batch_size, self.channels, -1)
        mb_mean = x.mean(dim=[0, 2]).view(1, -1 , 1)
        mb_var = ((x - mb_mean)**2).mean(dim=[0, 2])
        x_norm = x - mb_mean / ((mb_var + self.eps).sqrt().view(1, -1, 1))

        x = x_norm.reshape(original_shape)

        return x 

def test():
    x = torch.zeros([2, 3, 64, 64])
    BN = BatchNorm(3)
    x = BN(x)
    print(x.shape)


if __name__ == "__main__":
    test()
