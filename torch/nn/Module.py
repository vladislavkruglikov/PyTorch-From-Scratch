class Module:
    def __init__(self):
        self._train = True

    def forward(self, input):
        raise NotImplementedError

    def backward(self, input, grad_output):
        raise NotImplementedError

    def parameters(self):
        """Возвращает список собственных параметров."""
        return []

    def grad_parameters(self):
        """Возвращает список тензоров-градиентов для своих параметров."""
        return []

    def train(self):
        self._train = True

    def eval(self):
        self._train = False
