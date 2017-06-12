import torch
import torch.nn as nn
from torch.autograd import Variable

# Neural net

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, use_cuda=False):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_cuda = use_cuda

        self.rnn1 = nn.RNNCell(input_size, hidden_size)
        self.rnn2 = nn.RNNCell(hidden_size, input_size)
        self.output = nn.LogSoftmax()

    def forward(self, input, states):
        state1, state2 = states

        if self.use_cuda:
            input = input.cuda()
            state1 = state1.cuda()
            state2 = state2.cuda()

        state1 = self.rnn1(input, state1)
        state2 = self.rnn2(state1, state2)
        output = self.output(state2)
        return output, (state1, state2)

    def init_state(self):
        return Variable(torch.zeros(1, self.hidden_size)), Variable(torch.zeros(1, self.input_size))