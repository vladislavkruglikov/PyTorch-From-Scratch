import numpy as np

from torch.nn import Module


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.output = sigmoid(input)
        return self.output

    def backward(self, input, grad_output):
        grad_input = sigmoid(input) * (1 - sigmoid(input)) * grad_output
        return grad_input
