import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from torch.metrics import accuracy_score
from torch.optim import RMSProp, Momentum, Adam, SGD
from sklearn.model_selection import train_test_split

np.random.seed(54)

# Unpack zip before loading
df = pd.read_csv('mnist.csv').to_numpy()

Y = np.array(pd.get_dummies(df[:, -1]))
X = np.array(df[:, :-1])
X = (X / 255).astype('float32')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = nn.Sequential(
    nn.Linear(784, 200),
    nn.Sigmoid(),
    nn.Dropout(pKeep=0.8),

    nn.Linear(200, 80),
    nn.Sigmoid(),

    nn.Linear(80, 10),
    nn.SoftMax(),
)

epochs = 300
eval_every = 1
batch_size = 1024
criterion = nn.CrossEntropy()
optimizer = Adam(model)

for epoch in range(epochs):
    for x, y in DataLoader(X_train, Y_train, batch_size=batch_size):
        model.train()

        y_pred = model.forward(x)
        grad = criterion.backward(y_pred, y)
        model.backward(x, grad)

        optimizer.step(lr=0.003, beta1=0.9, beta2=0.999)

    if (epoch + 1) % eval_every == 0:
        model.eval()
        y_train_pred = model.forward(X_train)
        y_test_pred = model.forward(X_test)
        loss_train = criterion.forward(y_train_pred, Y_train)
        loss_test = criterion.forward(y_test_pred, Y_test)
        print(f'Epoch: {epoch + 1}/{epochs}')
        print(f'Train Loss: {loss_train} Train Accuracy: {accuracy_score(Y_train, y_train_pred)}')
        print(f'Test Loss: {loss_test} Test Accuracy: {accuracy_score(Y_test, y_test_pred)} \n')
