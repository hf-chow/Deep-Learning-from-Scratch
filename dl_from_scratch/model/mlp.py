from dl_from_scratch.utils.dataloader import load_hdf5

from math import log

from typing import Callable 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def acc_loss_plot(accuracies, losses):
    x = list(range(1,len(accuracies)+1))

    fig, ax1 = plt.subplots()
    ax1.plot(x, accuracies,label="Accuracy")

    ax2 = ax1.twinx()
    ax2.plot(x, losses,label="Loss", color="orange")

    plt.savefig("plot.png")

def logsumexp(x):
    c = x.max(axis=1)
    return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def d_sigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

def relu(x):
    return np.maximum(x, 0)

def d_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def map_labels(label):
    result = np.zeros((10, 1))
    result[label] = 1
    return result

def init_layer(input_shape, output_shape, mode="gaussian") -> np.array:
    # for MNIST, the layer 1 has input of 784 nodes, layer 2 has input of 128
    # hence the weight of layer 1 is a (784, 128) np array
    # to initialize the weight, we need to randomize the number and and redistribute 
    # the magnitude such that one randomized node is not statistically more important 
    # than another at the beginning, so as to not mislead the model
    # We will use the Gaussian initialization 
    # Gaussian distribution has a standard deviation of sqrt(2/n), where n 
    # is the nubmer of samples
    if mode == "gaussian":
        std = np.sqrt(2/(input_shape*output_shape))
        weight = np.random.rand(input_shape, output_shape)
        scaled_weight = np.multiply(weight, std)
    if mode == "uniform":
        scaled_weight = np.random.uniform(-1., 1., size=(input_shape, output_shape))/np.sqrt(input_shape*output_shape)
    return scaled_weight

def softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex)

def d_softmax(x):
    ex = np.exp(x-x.max())
    return ex/np.sum(ex, axis=0)*(1-ex/np.sum(ex, axis=0))

def cross_entropy_loss(pred, label):
    label_oh = (label[:, np.newaxis] == np.arange(10)).astype(int)
    loss_sample =(np.log(pred) * label_oh).sum(axis=1)
    loss = -np.mean(loss_sample)
    return loss

def SGD(l1, l2, d_l1, d_l2, lr=1e-3):
    l1 = l1 - lr*d_l1
    l2 = l2 - lr*d_l2
    return l1, l2

#TODO Refactor training loop to cater for different activation functions
def forward_backward(x, y, l1, l2):
    # Forward  
    label = np.zeros((len(y), 10))
    label[range(label.shape[0]), y] = 1
    x_l1 = x.dot(l1)

    out = np.zeros((len(y), 10), np.float32)
    out[range(out.shape[0]), y] = 1

    # Sigmoid
#    l1_out = sigmoid(x_l1)
    # relu
#    l1_out = relu(x_l1)
    l1_out = np.maximum(x_l1, 0)

    x_l2 =  l1_out.dot(l2)
#    x_out = softmax(x_l2)
    x_lsm = x_l2 - logsumexp(x_l2).reshape((-1, 1))
    x_loss = (-out * x_lsm).mean(axis=1)
    
    # Backward
    d_out = -out/len(y)
    dx_lsm = d_out - np.exp(x_lsm)*d_out.sum(axis=1).reshape((-1, 1))
#    d_sm = d_softmax(x_l2)
#    x_out_error = 2*d_sm*(x_out-label)/x_out.shape[0]
#    d_l2 = l1_out.T.dot(x_out_error)
    d_l2 = l1_out.T.dot(dx_lsm)
    dx_relu = dx_lsm.dot(l2.T)

#    d_l2_error = ((l2).dot(x_out_error.T)).T*d_sigmoid(x_l1)
#    d_l2_error = ((l2).dot(x_out_error.T)).T*d_relu(x_l1)
    dx_l1 = (l1_out > 0).astype(np.float32)*dx_relu
    d_l1 = x.T.dot(dx_l1)

    return x_l2, x_loss, d_l1, d_l2

def train():
    PATH = "../../data/MNIST/train.hdf5"
    x_train, y_train = load_hdf5(PATH)
#    TRAIN_PATH = "../../data/MNIST/train.csv"
#    y_train = pd.read_csv(TRAIN_PATH)["label"]
#    x_train = pd.read_csv(TRAIN_PATH).drop(labels=["label"], axis=1)

#    print(x_train.head())
    
    l1 = init_layer(784, 128, mode="uniform")
    l2 = init_layer(128, 10, mode="uniform")

    epochs = 1000
    batch_size = 128

    losses = []
    accuracies = []

    for i in tqdm(range(epochs)):
        sample = np.random.randint(0, x_train.shape[0], size=(batch_size))
        x = x_train[sample].reshape((-1, 28*28))
        y = y_train[sample]
        x_l2, x_loss, d_l1, d_l2 = forward_backward(x, y, l1, l2)
        l1, l2 = SGD(l1, l2, d_l1, d_l2)

        pred = np.argmax(x_l2,axis=1)
        acc = (pred == y).mean()
        loss = x_loss.mean()

        losses.append(loss)
        accuracies.append(acc)

    acc_loss_plot(accuracies, losses)
          

if __name__ == "__main__":
    train()
