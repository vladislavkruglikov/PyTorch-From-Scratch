import numpy as np

from torch.nn import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, input > 0)
        return grad_input


input = np.array([
    [1, 1],
    [-1, 5],
    [9, -1],
    [-1, -1],
    [12, 4],
    [16, 2]
])


grad_output = np.array([
    [0, 0],
    [0, 4],
    [3, 0],
    [0, -5],
    [-9, -1],
    [-1, -1]
])

activation = ReLU()

assert np.array_equal(activation.forward(input), np.array([
    [1, 1],
    [0, 5],
    [9, 0],
    [0, 0],
    [12, 4],
    [16, 2]
]))

assert np.array_equal(activation.backward(input, grad_output), np.array([
    [0, 0],
    [0, 4],
    [3, 0],
    [0, 0],
    [-9, -1],
    [-1, -1]
]))
