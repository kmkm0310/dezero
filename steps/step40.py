if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
from dezero import Function
from dezero.utils import plot_dot_graph
import dezero.functions as F


x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
# y = F.reshape(x, (6))
y = x0 + x1
print(y.data)
y.backward(retain_grad=True)
print(x0.grad)
print(x1.grad)