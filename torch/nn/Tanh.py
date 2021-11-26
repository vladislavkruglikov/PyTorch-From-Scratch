import numpy as np

from torch.nn import Module


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input):
        self.output = (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
        return self.output

    def backward(self, input, grad):
        return (1 - self.forward(input) ** 2) * grad
