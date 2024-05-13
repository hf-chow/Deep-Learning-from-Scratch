import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


# Building a MLP with pytorch
# The architecture would be mirroring mlp_numpy.py

device = (
          "cuda" if torch.cuda.is_available() 
          else "mps" if torch.backends.mps.is_available()
          else "cpu"
          )

def load_mnist(save_path):
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
    return train_data, test_data

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
                nn.Linear(28*28, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits

def dataloader(train, test, batch_size):
    train_dataloader = DataLoader(train, batch_size=batch_size)
    test_dataloader = DataLoader(test, batch_size=batch_size)
    return train_dataloader, test_dataloader

def train(model, dataloader, loss_func, optim):
    # Set model to training mode, instead of evaluation mode
    model.train()

    for batch, (x, y) in enumerate(dataloader):

        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_func(pred, y)

        loss.backward()
        # Perform optimization (e.g. gradient descend)
        optim.step()
        # Resetting grads for the next epoch
        optim.zero_grad()

        # Print status every 100 epochs
        train_size = len(dataloader.dataset)
        if batch % 100 == 0:
            loss = loss.item() # .item returns the values of the tensor as a Python float
            current_prog = (batch + 1)*len(x)

            print(f"Current loss: {loss} || Progress {current_prog}/{train_size}")

def test(model, dataloader, loss_func):
    # Set model to evaluation mode
    model.eval()
    loss , correct = 0, 0
    with torch.no_grad(): # Stop Pytorch from accumulating grads, which it does by default
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss += loss_func(pred, y).item()
    loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    print(f"Accuracy: {(correct*100):0.1f}% || Average loss: {loss:8f}\n")

def main():
    PATH = "../../data/MNIST/"
    batch_size = 128
    epochs = 20 

    print(f"Using {device} for trainig")
    model = MLP().to(device)
    print(model)
    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_data, test_data = load_mnist(save_path=PATH)
    train_dataloader, test_dataloader = dataloader(train_data, test_data, batch_size)

    for i in range(epochs):
        print(f"Epoch {i+1}")
        train(model, train_dataloader, loss_func=loss_func, optim=optim)
        test(model, test_dataloader, loss_func=loss_func)

#    load_mnist()

if __name__ ==  "__main__":
    main()
