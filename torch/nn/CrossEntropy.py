import numpy as np


from torch.nn import Criterion


def softmax(xs):
    xs = np.subtract(xs, xs.max(axis=1, keepdims=True))
    xs = np.exp(xs) / np.sum(np.exp(xs), axis=1, keepdims=True)
    return xs


class CrossEntropy(Criterion):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        eps = 1e-9
        predictions = np.clip(input, eps, 1. - eps)
        N = predictions.shape[0]
        ce = -np.sum(target * np.log(predictions))
        return ce / N


    def backward(self, input, target):
        eps = 1e-9
        input_clamp = np.clip(input, eps, 1 - eps)
        return softmax(input_clamp) - target