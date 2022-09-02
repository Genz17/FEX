import torch
import torch.nn as nn

class UnaryOperation(nn.Module):
    def __init__(self, operator, isLeave):
        super(UnaryOperation, self).__init__()
        self.operator = operator
        self.isLeave  = isLeave
        if not isLeave:
            self.a = nn.Parameter(torch.tensor(1.0))
            self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputData):
        if self.isLeave:
            return self.operator(inputData)
        else:
            return self.a*self.operator(inputData)+self.b

class BinaryOperation(nn.Module):
    def __init__(self, operator):
        super(BinaryOperation, self).__init__()
        self.operator = operator

    def forward(self, x, y):
        return self.operator(x, y)




unary_functions = [lambda x: 0*x,
                   lambda x: 1+0*x,
                   lambda x: x,
                   lambda x: x**2,
                   lambda x: x**3,
                   lambda x: x**4,
                   torch.exp,
                   torch.sin,
                   torch.cos,]

binary_functions = [lambda x,y: x+y,
                    lambda x,y: x*y,
                    lambda x,y: x-y]


unary_functions_str = ['lambda x: 0*x**2',
                   'lambda x: 1+0*x**2',
                   'lambda x: x+0*x**2',
                   'lambda x: x**2',
                   'lambda x: x**3',
                   'lambda x: x**4',
                   'torch.exp',
                   'torch.sin',
                   'torch.cos',]

binary_functions_str = ['lambda x,y: x+y',
                    'lambda x,y: x*y',
                    'lambda x,y: x-y']


