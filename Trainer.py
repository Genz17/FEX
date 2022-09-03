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

    for step in range(max_iter):
        print('-----------step:{}---------------'.format(step))

        actions = model.sample()
        treeBuffer = GetReward(model, actions, -1, 1, dim)

        databd0 = DataGen(1000, dim, -1, 1, True).cuda()
        data0 = DataGen(1000, dim, -1, 1).cuda()
        solubd0 = true_solution(databd0)

        errList = torch.zeros(model.batchSize)
        for batch in range(model.batchSize):

            ansTrainTree = treeBuffer[batch](data0)
            ansbdTree = treeBuffer[batch](databd0)
            loss = torch.nn.functional.mse_loss(LHS_pde(ansTrainTree, data0, dim), RHS_pde(data0)) + torch.nn.functional.mse_loss(ansbdTree, solubd0)
            errList[batch] = loss

        errinx = torch.argmin(errList)
        err = torch.min(errList)
        buffer.refresh(Candidate(treeBuffer[errinx], actions[errinx], err.item()))
        optimizer.zero_grad()
        err.backward()
        optimizer.step()
        for i in range(len(buffer.bufferzone)):
            print(buffer.bufferzone[i].action)

        with torch.no_grad():
            x = torch.linspace(-1,1,1000).view(1000,1).cuda()
            z = (0.5*x**2).view(1000)
            y = buffer.bufferzone[0].tree(x)
            print('relerr: {}'.format(torch.norm(y.view(1000)-z)/torch.norm(z)))
            x = x.view(1000).cpu().detach().numpy()
            y = y.view(1000).cpu().detach().numpy()
            #fig = plt.figure()
            #plt.plot(x,y)
            #plt.show()


if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    tree = BinaryTree.TrainableTree(1).cuda()
    model = Controller(tree).cuda()
    train(model, 1, 10)
