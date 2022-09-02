import torch
from DataGen import DataGen
from Equation import LHS_pde,RHS_pde,true_solution

def GetReward(model, batchOperations, domainLeft, domainRight, dim):

    batchSize = model.batchSize
    errList = []

    for batch in range(batchSize):
        batchOperation = batchOperations[batch]
        model.tree.PlaceOP(batchOperation)
        optimizer = torch.optim.Adam(model.tree.parameters())

        for _ in range(20):
            data = DataGen(1000, dim, domainLeft, domainRight).cuda()
            optimizer.zero_grad()
            databd = DataGen(1000, dim, domainLeft, domainRight, True).cuda()
            solubd = true_solution(databd)
            ansTrainTree = model.tree(data)
            ansbdTree = model.tree(databd)
            loss = torch.nn.functional.mse_loss(LHS_pde(ansTrainTree, data, dim), RHS_pde(data)) + torch.nn.functional.mse_loss(ansbdTree, solubd)
            loss.backward()
            optimizer.step()


        medList = []
        optimizer = torch.optim.LBFGS(model.tree.parameters(), lr=1, max_iter = 20)

        def closure():
            data = DataGen(1000, dim, domainLeft, domainRight).cuda()
            optimizer.zero_grad()
            databd = DataGen(1000, dim, domainLeft, domainRight, True).cuda()
            solubd = true_solution(databd)
            ansTrainTree = model.tree(data)
            ansbdTree = model.tree(databd)
            loss = torch.nn.functional.mse_loss(LHS_pde(ansTrainTree, data, dim), RHS_pde(data)) + torch.nn.functional.mse_loss(ansbdTree, solubd)
            medList.append(loss)
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)

        errList.append(min(medList))

    return errList
