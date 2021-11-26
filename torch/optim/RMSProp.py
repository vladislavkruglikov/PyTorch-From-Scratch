import numpy as np


prev_E = []
prev_E_tmp = []


def RMSProp(model, lr, beta):
    global prev_E
    global prev_E_tmp

    for i, (weights, gradient) in enumerate(zip(model.parameters(), model.grad_parameters())):
        eps = 1e-7
        if prev_E:
            E = beta * prev_E[i] + (1 - beta) * (gradient ** 2)
        else:
            E = (1 - beta) * (gradient ** 2)
        weights -= (lr * gradient) / (np.sqrt(E + eps))
        prev_E_tmp.append(E)

    prev_E = prev_E_tmp
