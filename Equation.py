import numpy as np
import torch
from torch import sin, cos, exp
import math

def LHS_pde(u, x, dim_set):

    v = torch.ones(u.shape)
    bs = x.size(0)
    ux = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True)[0]
    uxx = torch.zeros(bs, dim_set)
    for i in range(dim_set):
        ux_tem = ux[:, i:i+1]
        uxx_tem = torch.autograd.grad(ux_tem, x, grad_outputs=v, create_graph=True)[0]
        uxx[:, i] = uxx_tem[:, i]
    LHS = -torch.sum(uxx, dim=1, keepdim=True)
    return LHS

def RHS_pde(x):
    bs = x.size(0)
    dim = x.size(1)
    return -dim*torch.ones(bs, 1)

def true_solution(x):
    return 0.5*torch.sum(x**2, dim=1, keepdim=True)#1 / (2 * x[:, 0:1] + x[:, 1:2]-5)
