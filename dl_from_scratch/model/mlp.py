from dl_from_scratch.utils.dataset import load_mnist
from dl_from_scratch.utils.plot import acc_loss_plot

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


device = (
          "cuda" if torch.cuda.is_available() 
          else "mps" if torch.backends.mps.is_available()
          else "cpu"
          )

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

    loss, acc = 0, 0
    for batch, (x, y) in enumerate(dataloader):

        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_func(pred, y)

        loss.backward()
        # Perform optimization (e.g. gradient descend)
        optim.step()
        # Resetting grads for the next epoch
        optim.zero_grad()

        loss = loss.item() # .item returns the values of the tensor as a Python float
        acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Print status every 100 epochs
        train_size = len(dataloader.dataset)
        if batch % 100 == 0:
            current_prog = (batch + 1)*len(x)
            print(f"Current loss: {loss} || Progress {current_prog}/{train_size}")

    loss /= len(dataloader)
    acc /= len(dataloader.dataset)
    return acc, loss

def test(model, dataloader, loss_func):
    # Set model to evaluation mode
    model.eval()
    loss , acc = 0, 0
    with torch.no_grad(): # Stop Pytorch from accumulating grads, which it does by default
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            acc += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss += loss_func(pred, y).item()

    loss /= len(dataloader)
    acc /= len(dataloader.dataset)
    print(f"Accuracy: {(acc*100):0.1f}% || Average loss: {loss:8f}\n")
    return acc, loss

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

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_acc, train_loss = train(model, train_dataloader, loss_func=loss_func, optim=optim)
        test_acc, test_loss = test(model, test_dataloader, loss_func=loss_func)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
    acc_loss_plot(train_accs, train_losses, figname="../assets/torch_mlp_train.png")
    acc_loss_plot(test_accs, test_losses, figname="../assets/torch_mlp_test.png")

#    load_mnist()

if __name__ ==  "__main__":
    main()
