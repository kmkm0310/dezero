if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math
from dezero import Variable
from dezero import Function
from dezero.utils import plot_dot_graph

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx
    
def sin(x):
    return Sin()(x)

def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

x0 = Variable(np.array(np.pi/4))
y0 = sin(x0)
y0.backward()

x1 = Variable(np.array(np.pi/4))
y1 = my_sin(x1)
y1.backward()

print(y0.data)
print(x0.grad)

print(y1.data)
print(x1.grad)
