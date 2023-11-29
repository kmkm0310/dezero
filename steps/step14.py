import numpy as np
import unittest

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
        
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
        
    def test_gradient_check(self):
        x = Variable(np.random.rand(1)) # ランダムな入力値を生成
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

class Variable:
    def __init__(self, data):
        # if data is not None:
        #     if not isinstance(data, np.ndarray):
        #         raise TypeError('{} is not supported'.format(type(data)))
            
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 関数を取得
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
                
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
            
                if x.creator is not None:
                    funcs.append(x.creator) # 1つ前の関数をリストに追加
    
    def cleargrad(self):
        self.grad = None

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 具体的な計算はforwardメソッドで行う
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self) # 出力変数に生みの親を覚えさせる
        self.inputs = inputs #入力された変数を覚える
        self.outputs = outputs # 出力も覚える
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def add(x0, x1):
    return Add()(x0, x1)

# 1回目の計算
x = Variable(np.array(3.0))
y = add(x, x)
y.backward()
print(y.data)
print(x.grad)

# 2回目の計算（同じxを使って、別の計算を行う）
x.cleargrad()
y = add(add(x, x), x)
y.backward()
print(y.data)
print(x.grad)
