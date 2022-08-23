import torch

def DataGen(batchSize, dim, domainLeft, domainRight, boundary=False):
    data = torch.rand(batchSize, dim, requires_grad=True)*(domainRight-domainLeft) + domainLeft
    if boundary:
        # Make half of the data with left bd, other half right bd.
        N1 = batchSize//2

        data[:N1,:] = domainLeft
        data[N1:,:] = domainRight

    return data
