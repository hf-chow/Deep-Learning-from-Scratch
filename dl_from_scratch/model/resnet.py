from dl_from_scratch.module.residual_block import BasicBlock
from dl_from_scratch.utils.dataset import Dataset
from dl_from_scratch.utils.normalizer import normalize_2d
from dl_from_scratch.utils.plot import acc_loss_plot

from collections import deque

import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = deque(layers)
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=64,
                          kernel_size=(7,7),
                          stride=2,
                          padding=3),
                nn.MaxPool2d(kernel_size=(3, 3))
                )
        self.resnet_layers = self.get_layers(self.layers)
        self.classifer =  nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(512, 1000)
                ) 
    def get_layers(self, downsampling=False):
        resnet_layers = nn.Sequential()

        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        while len(self.layers) > 1:
            current_layer = self.layers.popleft()
            next_layer = self.layers[0]
            if current_layer == next_layer:
                resnet_layers.append(BasicBlock(in_channels=current_layer,
                                                out_channels=next_layer))
                resnet_layers.append(maxpool)
            else:
                resnet_layers.append(
                        BasicBlock(in_channels=current_layer,
                                   out_channels=next_layer,
                                   downsample=nn.Conv2d(in_channels=current_layer,
                                                        out_channels=next_layer,
                                                        kernel_size=(1, 1))))
                resnet_layers.append(maxpool)
        return resnet_layers

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet_layers(x)
        x = self.classifer(x)

def test():
    layers = [64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512]
    rn = ResNet(layers)
    print(rn)
    x = torch.rand(256, 3, 224, 224)
    x = rn(x)

if __name__ == "__main__":
    test()

    
 

