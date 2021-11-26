import numpy as np

from torch.nn import Module


class SoftMax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        self.output = np.exp(self.output) / np.sum(np.exp(self.output), axis=1, keepdims=True)
        return self.output

    def backward(self, input, grad_output):
        return grad_output
