from MessageReader import MessageReader
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensor_utils import *
from NetModule import NetModule
import numpy as np
from pathlib import Path
import time
import math


# Neural net

class NetModule(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, use_cuda=True):
        super(NetModule, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_cuda = use_cuda
        self.n_layers = n_layers
        self.rnn_layers = [nn.LSTMCell(input_size, hidden_size)]

        for layer in range(n_layers - 1):
            self.rnn_layers.append(nn.LSTMCell(hidden_size, hidden_size))

        self.rnn_layers = nn.ModuleList(self.rnn_layers)

        self.linear_output = nn.Linear(hidden_size, input_size)
        self.softmax_output = nn.LogSoftmax()

    def forward(self, input, states):

        if self.use_cuda:
            input = input.cuda()

        for layer in range(self.n_layers):
            rnn_layer = self.rnn_layers[layer]
            states[layer] = rnn_layer(input, states[layer])
            input = states[layer][0]

        last_hidden = states[-1][0]

        linear_output = self.linear_output(last_hidden)
        softmax_output = self.softmax_output(linear_output)
        return softmax_output, states

    def init_state(self, batch_size):
        tensor_constructor = torch.FloatTensor
        state = torch.zeros(batch_size, self.hidden_size)

        if self.use_cuda:
            state = state.cuda()
            tensor_constructor = torch.cuda.FloatTensor

        out = []

        for layer in range(self.n_layers):
            hidden_cell_pair = [Variable(tensor_constructor(state)), Variable(tensor_constructor(state))]
            out.append(hidden_cell_pair)

        return out