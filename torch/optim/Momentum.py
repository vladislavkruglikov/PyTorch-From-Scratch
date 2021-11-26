import numpy as np


def Momentum(model, lr, beta):
    prev_v = None
    prev_v_tmp = []
    for i, (weights, gradient) in enumerate(zip(model.parameters(), model.grad_parameters())):
        if prev_v:
            v = beta * prev_v[i] - lr * gradient
        else:
            v = - lr * gradient

        weights += v
        prev_v_tmp.append(v)

    prev_v = prev_v_tmp
