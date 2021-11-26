import numpy as np


from torch.nn import Module


class Linear(Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        stdv = 1. / np.sqrt(dim_in)
        # self.W = np.random.uniform(-stdv, stdv, size=(dim_in, dim_out))
        # self.b = np.random.uniform(-stdv, stdv, size=dim_out)

        self.W = np.random.randn(dim_in, dim_out)
        self.b = np.random.randn(1, dim_out)

    def forward(self, input):
        self.output = (np.dot(input, self.W) + self.b)
        return self.output

    def backward(self, input, grad_output):
        self.grad_b = np.mean(grad_output, axis=0)
        self.grad_W = np.dot(input.T, grad_output)

        self.grad_W /= input.shape[0]

        grad_input = np.dot(grad_output, self.W.T)

        return grad_input

    def parameters(self):
        return [self.W, self.b]

    def grad_parameters(self):
        return [self.grad_W, self.grad_b]
