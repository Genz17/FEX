import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import BinaryTree
from Equation import LHS_pde,RHS_pde,true_solution
from DataGen import DataGen
from Controller import Controller
from GetReward import GetReward
from OperationBuffer import Buffer
from Candidate import Candidate

def train(model, dim, max_iter):
    optimizer = torch.optim.Adam(model.NN.parameters())
    buffer = Buffer(10)
    databd0 = DataGen(1000, dim, 0, 1, True).cuda()
    data0 = DataGen(1000, dim, 0, 1).cuda()
    solubd0 = true_solution(databd0)

    for step in range(max_iter):
        optimizer.zero_grad()
        print('-----------step:{}---------------'.format(step))
        actions = model.sample()
        rewards = GetReward(model, actions, 0, 1, dim)

        print(min(rewards))
        action = actions[rewards.index(min(rewards))]

        model.tree.PlaceOP(action)

        ansTrainTree = model.tree(data0)
        ansbdTree = model.tree(databd0)
        loss = torch.nn.functional.mse_loss(LHS_pde(ansTrainTree, data0, dim), RHS_pde(data0)) + torch.nn.functional.mse_loss(ansbdTree, solubd0)

        buffer.refresh(Candidate(action.detach().cpu().tolist(), loss.item()))
        loss.backward(retain_graph=True)
        optimizer.step()
        print(len(buffer.bufferzone))
        print(buffer.bufferzone[0].error)



if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    tree = BinaryTree.TrainableTree(1).cuda()
    model = Controller(tree).cuda()
    train(model, 1, 300)
