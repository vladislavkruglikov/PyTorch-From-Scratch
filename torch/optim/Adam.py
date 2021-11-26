import numpy as np


class Adam:
    def __init__(self, model):
        self.prev_m = None
        self.prev_v = None
        self.model = model
        self.t = 1

    def step(self, lr, beta1, beta2):
        prev_m_tmp = []
        prev_v_tmp = []
        eps = 1e-7

        for i, (weights, gradient) in enumerate(zip(self.model.parameters(), self.model.grad_parameters())):
            if self.prev_m and self.prev_v:
                m = beta1 * self.prev_m[i] + (1 - beta1) * gradient
                v = beta2 * self.prev_v[i] + (1 - beta2) * gradient ** 2
                m_hat = m / (1 - beta1 ** self.t)
                v_hat = v / (1 - beta2 ** self.t)
            else:
                m = beta1 * 0 + (1 - beta1) * gradient
                v = beta2 * 0 + (1 - beta2) * gradient ** 2
                m_hat = m / (1 - beta1 ** self.t)
                v_hat = v / (1 - beta2 ** self.t)

            weights -= lr * m_hat / (np.sqrt(v_hat) + eps)

            prev_m_tmp.append(m)
            prev_v_tmp.append(v)

        self.prev_m = prev_m_tmp
        self.prev_v = prev_v_tmp

        self.t += 1
