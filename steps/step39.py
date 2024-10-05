if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
from dezero import Function
from dezero.utils import plot_dot_graph
import dezero.functions as F


x = Variable(np.array([[1, 2, 3, 4, 5, 6]]))
# x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# y = F.reshape(x, (6))
y = F.sum(x)
y.backward(retain_grad=True)
print(x.grad)
print(y.data)

x = Variable(np.random.randn(2, 3, 4, 5))
y = x.sum(keepdims=True)
print(y.shape)
print(y.data)