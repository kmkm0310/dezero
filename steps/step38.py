if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
from dezero import Function
from dezero.utils import plot_dot_graph
import dezero.functions as F


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# y = F.reshape(x, (6))
y = F.transpose(x)
y.backward(retain_grad=True)
print(x.grad)
print(y.data)
