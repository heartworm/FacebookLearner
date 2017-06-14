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

        self.rnn1 = nn.LSTMCell(input_size, hidden_size)
        self.rnn2 = nn.LSTMCell(hidden_size, hidden_size)
        self.pre_output = nn.Linear(hidden_size, input_size)
        self.output = nn.LogSoftmax()

    def forward(self, input, states):
        state1, state2 = states

        if self.use_cuda:
            input = input.cuda()
            state1 = (state1[0].cuda(), state1[1].cuda())
            state2 = (state2[0].cuda(), state2[1].cuda())

        state1 = self.rnn1(input, state1)
        state2 = self.rnn2(state1[0], state2)

        pre_output = self.pre_output(state2[0])
        output = self.output(pre_output)
        return output, (state1, state2)

    def init_state(self, batch_size):
        out = ((Variable(torch.zeros(batch_size, self.hidden_size)), Variable(torch.zeros(batch_size, self.hidden_size))),
              (Variable(torch.zeros(batch_size, self.hidden_size)), Variable(torch.zeros(batch_size, self.hidden_size))))
        if self.use_cuda:
            for hidden in out:
                for item in hidden:
                    item.cuda()
        return out