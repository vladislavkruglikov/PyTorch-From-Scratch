from torch.nn import Module


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)

        self.output = input
        return self.output

    def backward(self, input, grad_output):
        for i in range(len(self.layers) - 1, 0, -1):
            grad_output = self.layers[i].backward(self.layers[i-1].output, grad_output)

        grad_output = self.layers[0].backward(input, grad_output)

        return grad_output

    def parameters(self):
        res = []
        for l in self.layers:
            res += l.parameters()
        return res

    def grad_parameters(self):
        res = []
        for l in self.layers:
            res += l.grad_parameters()
        return res

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()
