import torch
import torch.nn as nn
from Operations import unary_functions, binary_functions, unary_functions_str, binary_functions_str
from Operations import UnaryOperation, BinaryOperation

class BinaryTreeNode(object):
    def __init__(self, is_unary, count=0, isLeave=True):
        self.operation      = lambda x:0
        self.is_unary       = is_unary
        self.count          = count
        self.isLeave        = isLeave
        self.leftchild      = None
        self.rightchild     = None

    def insertLeft(self, is_unary):
        self.isLeave = False
        if self.leftchild == None:
            self.leftchild = BinaryTreeNode(is_unary)
        else:
            pass

    def insertRight(self, is_unary):
        self.isLeave = False
        if self.rightchild == None:
            self.rightchild = BinaryTreeNode(is_unary)
        else:
            pass


def OperationPlace(tree, operationInxList, operationDict):
    if tree.leftchild == None and tree.rightchild == None:
        tree.operation = operationDict[str((tree.count, operationInxList[tree.count-1].item()))]

    if tree.leftchild != None and tree.rightchild == None:
        tree.operation = operationDict[str((tree.count, operationInxList[tree.count-1].item()))]
        OperationPlace(tree.leftchild, operationInxList, operationDict)

    if tree.leftchild == None and tree.rightchild != None:
        tree.operation = operationDict[str((tree.count, operationInxList[tree.count-1].item()))]
        OperationPlace(tree.rightchild, operationInxList, operationDict)

    if tree.leftchild != None and tree.rightchild != None:
        tree.operation = operationDict[str((tree.count, operationInxList[tree.count-1].item()))]
        OperationPlace(tree.leftchild, operationInxList, operationDict)
        OperationPlace(tree.rightchild, operationInxList, operationDict)


def BasicTreeGen():
    tree = BinaryTreeNode(False)
    tree.insertLeft(False)
    tree.insertRight(False)
    tree.leftchild.insertLeft(False)
    tree.leftchild.insertRight(False)
    tree.rightchild.insertLeft(False)
    tree.rightchild.insertRight(False)
    tree.leftchild.leftchild.insertLeft(True)
    tree.leftchild.leftchild.insertRight(True)
    tree.leftchild.rightchild.insertLeft(True)
    tree.leftchild.rightchild.insertRight(True)
    tree.rightchild.leftchild.insertLeft(True)
    tree.rightchild.leftchild.insertRight(True)
    tree.rightchild.rightchild.insertLeft(True)
    tree.rightchild.rightchild.insertRight(True)
    NodeNumCompute(tree)
    return tree

def ShowTree(tree, cnt, ans=None):
    if tree.count == cnt:
        return tree
    else:
        childList = [tree.leftchild, tree.rightchild]
        for tree in childList:
            if tree != None:
               ans = ShowTree(tree, cnt)
               if ans != None:
                   break
    if ans != None:
        return ans

def ComputeThroughTree(treeNode, linearTransform, inputData):
    if treeNode.leftchild == None and treeNode.rightchild == None:
        ans = treeNode.operation(inputData)
        ans = linearTransform[str(treeNode.count)](ans)

    if treeNode.leftchild != None and treeNode.rightchild == None:
        ans = treeNode.operation(ComputeThroughTree(treeNode.leftchild,linearTransform,inputData))

    if treeNode.leftchild == None and treeNode.rightchild != None:
        ans = treeNode.operation(ComputeThroughTree(treeNode.rightchild,linearTransform,inputData))

    if treeNode.leftchild != None and treeNode.rightchild != None:
        ans = treeNode.operation(ComputeThroughTree(treeNode.leftchild,linearTransform,inputData),ComputeThroughTree(treeNode.rightchild,linearTransform,inputData))
    return ans

def NodeNumCompute(tree, num=0):
    if tree.leftchild == None and tree.rightchild == None:
        num = num + 1
        tree.count = num

    if tree.leftchild != None and tree.rightchild == None:
        num = NodeNumCompute(tree.leftchild, num) + 1
        tree.count = num

    if tree.leftchild == None and tree.rightchild != None:
        num = NodeNumCompute(tree.rightchild, num) + 1
        tree.count = num

    if tree.leftchild != None and tree.rightchild != None:
        num = NodeNumCompute(tree.leftchild, num)
        num = NodeNumCompute(tree.rightchild, num)
        num += 1
        tree.count = num
    return num

def TotalopNumCompute(tree):
    opList = [0 for i in range(tree.count)]
    for i in range(1,tree.count+1):
        treeNode = ShowTree(tree, i)
        if treeNode.is_unary:
            opList[treeNode.count-1] = len(unary_functions)
        else:
            opList[treeNode.count-1] = len(binary_functions)
    return opList


def LeaveNumCompute(tree):
    leaveList = []
    for i in range(1,tree.count+1):
        leave = ShowTree(tree, i)
        if leave.leftchild == None and leave.rightchild == None:
            leaveList.append(leave.count)
    return leaveList

class TrainableTree(nn.Module):
    def __init__(self, dim):
        super(TrainableTree, self).__init__()
        self.dim                = dim
        self.tree               = BasicTreeGen()
        self.operators          = {}
        self.linearTransform    = {str(i):nn.Sequential(nn.Linear(dim, dim),nn.ReLU(),nn.Linear(dim, 1)) for i in LeaveNumCompute(self.tree)}
        self.linearTransform    = nn.ModuleDict(self.linearTransform)
        self.OperatorsGen(self.tree)
        self.operators          = nn.ModuleDict(self.operators)

    def forward(self, inputData):
        return ComputeThroughTree(self.tree, self.linearTransform, inputData)

    def PlaceOP(self, operationList):
        OperationPlace(self.tree, operationList, self.operators)
        self.operationList = operationList

    def LinearGen(self):
        for key in self.linearTransform:
            for layer in self.linearTransform[key].modules():
                try:
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
                except BaseException:
                    pass

    def OperatorsGen(self, tree):
        if tree.leftchild == None and tree.rightchild == None:
            for i in range(len(unary_functions)):
                self.operators.update({str((tree.count, i)):UnaryOperation(unary_functions[i], True)})

        if tree.leftchild != None and tree.rightchild == None:
            for i in range(len(unary_functions)):
                self.operators.update({str((tree.count, i)):UnaryOperation(unary_functions[i], True)})
            self.OperatorsGen(tree.leftchild)

        if tree.leftchild == None and tree.rightchild != None:
            for i in range(len(unary_functions)):
                self.operators.update({str((tree.count, i)):UnaryOperation(unary_functions[i], True)})
            self.OperatorsGen(tree.rightchild)

        if tree.leftchild != None and tree.rightchild != None:
            for i in range(len(binary_functions)):
                self.operators.update({str((tree.count, i)):BinaryOperation(binary_functions[i])})
            self.OperatorsGen(tree.leftchild)
            self.OperatorsGen(tree.rightchild)


