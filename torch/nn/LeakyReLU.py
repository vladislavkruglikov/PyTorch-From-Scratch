import numpy as np

from torch.nn import Module


class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        super().__init__()

        self.slope = slope

    def forward(self, input):
        self.output = np.where(input <= 0, input, self.slope * input)
        return self.output

    def backward(self, input, grad_output):
        grad_input = np.choose(input <= 0, [np.sign(input) * self.slope, 1]) * grad_output
        return grad_input
