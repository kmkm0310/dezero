if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math
from dezero import Variable
from dezero import Function
from dezero.utils import plot_dot_graph

def f(x):
    y = x **4 - 2 * x **2
    return y

def gx2(x):
    return 12 * x ** 2 - 4

x = Variable(np.array(2.0))
lr = 0.001 # 学習率
iters = 10 # 繰り返す回数

for i in range(iters):
    print(i, x.data)

    y = f(x)
    print(y.data)
    
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data
