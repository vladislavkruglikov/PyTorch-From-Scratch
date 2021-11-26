import numpy as np


from torch.nn.Module import Module


class Dropout(Module):
    def __init__(self, pKeep=0.9):
        super(Dropout, self).__init__()
        self.pKeep = pKeep

    def forward(self, input):
        if self._train:
            binary_value = np.random.rand(input.shape[0], input.shape[1]) < self.pKeep
            res = np.multiply(input, binary_value)
            res /= self.pKeep
            self.output = res
        else:
            self.output = input

        return self.output

    def backward(self, input, gradient):
        return gradient
