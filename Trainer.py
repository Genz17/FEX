import torch
import torch.nn as nn
import BinaryTree
from Controller import Controller
from GetReward import GetReward

def train(model, dim, max_iter):
    optimizer = torch.optim.Adam(model.parameters())

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



if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    tree = BinaryTree.TrainableTree(100)
    model = Controller(tree)
    model.train
    train(model, 100, 10)
