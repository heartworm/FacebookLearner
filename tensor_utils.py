import torch

def onehot_to_index(oh):
    _, indexes = torch.max(oh, 1)
    index = indexes[0][0]
    return index

def index_to_onehot(ind, size):
    tensor = torch.zeros(1, size)
    tensor[0][ind] = 1
    return tensor