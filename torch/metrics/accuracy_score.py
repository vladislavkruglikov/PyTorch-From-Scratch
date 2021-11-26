import numpy as np


def accuracy_score(y_true, y_pred):
    a = np.argmax(y_true, axis=1)
    b = np.argmax(y_pred, axis=1)
    return np.count_nonzero(a == b) / y_true.shape[0]
