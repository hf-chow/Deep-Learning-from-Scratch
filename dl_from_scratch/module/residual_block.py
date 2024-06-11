from collections import deque

import torch
import torch.nn as nn
from dl_from_scratch.module.batchnorm import BatchNorm


class BasicBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1,
                 downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn(x)
        if self.downsample is None:
            x += identity
        else:
            x += self.downsample(identity)
        x = self.relu(x)
        
        return x

class ResNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = deque(layers)
        self.resnet_layers = self.get_layers(self.layers)

    def get_layers(self, downsampling=False):
        resnet_layers = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=64,
                          kernel_size=(7, 7),
                          stride=2,
                          padding=3))
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
        x = self.resnet_layers(x)


def test():
    layers = [64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512]
    rn = ResNet(layers)
    print(rn)
    x = torch.rand(256, 3, 224, 224)
    x = rn(x)

#    conv1 = nn.Sequential(
#            nn.Conv2d(in_channels=3,
#                      out_channels=64,
#                      kernel_size=(7,7),
#                      stride=2,
#                      padding=3),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#            )
#    mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#    
#    ds1 = nn.Conv2d(64, 128, kernel_size=1)
#    ds2 = nn.Conv2d(128, 256, kernel_size=1)
#    ds3 = nn.Conv2d(256, 512, kernel_size=1)
#
#    bb1 = BasicBlock(in_channels=64, out_channels=64)
#    bb2 = BasicBlock(in_channels=64, out_channels=128, downsample=ds1)
#    bb3 = BasicBlock(in_channels=128, out_channels=256, downsample=ds2)
#    bb4 = BasicBlock(in_channels=256, out_channels=512, downsample=ds3)
#
#
#
#    x = torch.rand(256, 3, 224, 224)
#    x = conv1(x)
#    x = bb1(x)
#    x = mp(x)
#    x = bb2(x)
#    x = mp(x)
#    x = bb3(x)
#    x = mp(x)
#    x = bb4(x)

if __name__ == "__main__":
    test()

    
 

