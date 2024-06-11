from dl_from_scratch.module.residual_block import BasicBlock
from dl_from_scratch.utils.dataset import Dataset
from dl_from_scratch.utils.normalizer import normalize_2d
from dl_from_scratch.utils.plot import acc_loss_plot

from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

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
        return x


def dataloader(train, test, batch_size):
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

def train(model, dataloader, loss_func, optim):
    model.train()
    
    loss, acc = 0, 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_func(pred, y)

        loss.backward()
        optim.step()
        optim.zero_grad()

        loss = loss.item()
        acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        train_size = len(dataloader.dataset)
        if batch % 10 == 0:
            current_prog = (batch + 1)*len(x)
            print(f"Current loss: {loss} || Progress {current_prog}/{train_size}")

    loss /= len(dataloader)
    acc /= len(dataloader.dataset)
    return loss, acc

def test(model, dataloader, loss_func):
    model.eval()
    loss, acc = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            acc += (pred.argmax(1)==y).type(torch.float).sum().item()
            loss += loss_func(pred, y).item()

    loss /= len(dataloader)
    acc /= len(dataloader.dataset)
    print(f"Accuracy: {(acc*100):0.1f}% || Average loss: {loss:8f}\n")
    return loss, acc


def main():
#    PATH = "../../data/ImageNet/"
    PATH = "../../data/Imagenette/"
    epochs = 20
    batch_size = 64

    _train_data = datasets.Imagenette(root=PATH, 
                                      split="train", 
                                      size="320px", 
                                      transform=transforms.Compose(
                                          [transforms.RandomCrop(227,227), 
                                           transforms.ToTensor()]))
    means, stds = normalize_2d(_train_data)


    train_trans = transforms.Compose([
#            transforms.Resize((227, 227)),
            transforms.RandomCrop((227, 227)),
#            transforms.RandomCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
            ])
    test_trans = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
        ])

    trans = (train_trans, test_trans)

    train_data, test_data = Dataset.load_imagenette(save_path=PATH, download=False, transform_func=trans)
    train_dataloader, test_dataloader = dataloader(train_data, test_data, batch_size)

    resnet_34_layers = [64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512]
    ResNet34 = ResNet(layers=resnet_34_layers)

    model = ResNet34.to(device)
    print(model)
    loss_func = nn.CrossEntropyLoss()
    # SGD
    optim = torch.optim.SGD(model.parameters(), lr=5e-3, weight_decay=5e-3, momentum=0.9)
    # Adam
#    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_loss, train_acc = train(model, train_dataloader, loss_func=loss_func, optim=optim)
        test_loss, test_acc = test(model, test_dataloader, loss_func=loss_func)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
    acc_loss_plot(train_accs, train_losses, figname="../assets/alexnet_train.png")
    acc_loss_plot(test_accs, test_losses, figname="../assets/alexnet_test.png")

if __name__ == "__main__":
    main()

    
 

