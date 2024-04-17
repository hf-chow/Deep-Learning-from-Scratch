from utils import dataloader

import numpy as np

def sigmoid(x: np.float64) -> np.float64:
    return 1 / (1+np.exp(-x))

def sigmoid_deriv(x: np.float64) -> np.float64:
    return sigmoid(x) * (1-sigmoid(x))

def forward(input: np.ndarray, weight: np.ndarray) -> np.float64:
    # Forward pass is a matrix vector multiplication between the input vector
    # and the weight matrix. For MNIST we will flatten the entire 20x20 array
    # into a 1D vector and the resultant will be reduced a single float
    results = np.matmul(input, weight)
    results = sigmoid(results)
    return  results
    
