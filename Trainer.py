import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import BinaryTree
from Controller import Controller
from GetReward import GetReward

def train(model, dim, max_iter):
    optimizer = torch.optim.Adam(model.parameters())
    pList = [0 for i in range(max_iter)]

    for step in range(max_iter):
        print('-----------step:{}---------------'.format(step))
        optimizer.zero_grad()
        actions = model.sample()
        rewards = GetReward(model.batchSize, actions, 0, 1, dim, model.tree)
        #print(rewards)
        #err = min(rewards)
        #print(err)
        err = rewards
        err.backward(retain_graph=True)
        optimizer.step()
        print(err)
        pList[step] = err.item()
    fig = plt.figure()
    plt.semilogy(range(max_iter), pList)
    plt.show()




if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    tree = BinaryTree.TrainableTree(1)
    model = Controller(tree)
    model.train
    train(model, 1, 30)
