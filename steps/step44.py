if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
from dezero import Function
from dezero.utils import plot_dot_graph
import dezero.functions as F
import dezero.layers as L

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

l1 = L.Linear(10) # 出力サイズを指定
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid_simple(y)
    y = l2(x)
    return y

lr = 0.2
iters = 10000

# ③ニューラルネットワークの学習
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            print(p)
            p.data -= lr * p.grad.data
    if i % 1000 == 0: # 1000回ごとに出力
        print(loss)
        

x_P = np.linspace(0, 1, 100).reshape(100,1)
y_P = predict(x_P)

plt.scatter(x, y)
plt.plot(x_P, y_P.data)
plt.show()