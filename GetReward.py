import torch
from DataGen import DataGen
from Equation import LHS_pde,RHS_pde,true_solution

def GetReward(batchSize, batchOperations, domainLeft, domainRight, dim, trainableTree):

    data = DataGen(batchSize, dim, domainLeft, domainRight)
    errList = []

    for batch in range(batchSize):
        batchOperation = batchOperations[batch]
        optimizer = torch.optim.Adam(trainableTree.parameters())

        for _ in range(20):
            optimizer.zero_grad()
            databd = DataGen(batchSize, dim, domainLeft, domainRight, True)
            solubd = true_solution(databd)
            ansTrainTree = trainableTree(data, batchOperation)
            ansbdTree = trainableTree(databd, batchOperation)
            loss = torch.nn.functional.mse_loss(LHS_pde(ansTrainTree, data, dim), RHS_pde(data)) + 100*torch.nn.functional.mse_loss(ansbdTree, solubd)
            loss.backward(retain_graph=True)
            optimizer.step()


        optimizer = torch.optim.LBFGS(trainableTree.parameters(), lr=1, max_iter = 20)
        medList = []

        def closure():
            optimizer.zero_grad()
            databd = DataGen(batchSize, dim, domainLeft, domainRight, True)
            solubd = true_solution(databd)
            ansTrainTree = trainableTree(data, batchOperation)
            ansbdTree = trainableTree(databd, batchOperation)
            loss = torch.nn.functional.mse_loss(LHS_pde(ansTrainTree, data, dim), RHS_pde(data)) + 100*torch.nn.functional.mse_loss(ansbdTree, solubd)
            #medList.append(loss)
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)

        databd = DataGen(batchSize, dim, domainLeft, domainRight, True)
        solubd = true_solution(databd)
        ansTrainTree = trainableTree(data, batchOperation)
        ansbdTree = trainableTree(databd, batchOperation)
        loss = torch.nn.functional.mse_loss(LHS_pde(ansTrainTree, data, dim), RHS_pde(data)) + 100*torch.nn.functional.mse_loss(ansbdTree, solubd)
        #medList.append(loss)
        #errList.append(min(medList))

    return loss


