import torch

def onehot_to_index(oh):
    _, indexes = torch.max(oh, 1)
    index = indexes[0][0]
    return index

def index_to_onehot(ind, size):
    tensor = torch.zeros(1, size)
    tensor[0][ind] = 1
    return tensor

def index_sequence_to_onehot_sequence(sequence, size):
    out_tensor = None
    for ind in sequence:
        tens = torch.unsqueeze(index_to_onehot(ind, size), 0)
        if out_tensor is None:
            out_tensor = tens
        else:
            out_tensor = torch.cat((out_tensor, tens), 0)
    return out_tensor