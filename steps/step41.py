if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
from dezero import Function
from dezero.utils import plot_dot_graph
import dezero.functions as F


x = Variable(np.random.randn(2, 3))
W = Variable(np.random.randn(3, 4))
y = F.matmul(x, W)
y.backward(retain_grad=True)

print(x.grad.shape)
print(W.grad.shape)