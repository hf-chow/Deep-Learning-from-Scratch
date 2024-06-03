import torch
import torch.nn as nn

from dl_from_scratch.module.batchnorm import BatchNorm

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
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

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn(x)
        x += identity
        x = self.relu(x)
        
        return x

def test():
    conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=(7,7),
                      stride=2,
                      padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
    bb = BasicBlock(in_channels=64, out_channels=64)

    x = torch.rand(256, 3, 224, 224)
    x = conv1(x)
    x = bb(x)

if __name__ == "__main__":
    test()

    
 

