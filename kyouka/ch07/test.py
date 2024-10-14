if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x_np = np.array(5.0)
x = Variable(x_np)

y = 3 * x ** 2
print(y)
y.backward()
print(x.grad)