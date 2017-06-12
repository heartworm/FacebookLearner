# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from MessageReader import MessageReader
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensor_utils import *
from Net import Net
import numpy as np
from pathlib import Path
mr = MessageReader("messages.json")

######################################################################
# Training
# =========
# Preparing for Training
# ----------------------
#
# First of all, helper functions to get random pairs of (category, line):
#


def message_sequence_to_humanreadable(msgseq):
    out_str = ""
    for i in range(msgseq.size()[0]):
        out_str += mr.index_to_humanreadable(onehot_to_index(msgseq[i]))
    return out_str


######################################################################
# Training the Network
# --------------------
#
# In contrast to classification, where only the last output is used, we
# are making a prediction at every step, so we are calculating loss at
# every step.
#
# The magic of autograd allows you to simply sum these losses at each step
# and call backward at the end.
#

n_message_batch = 100 #messages per training epoch
hidden_nodes = 500
rnn = Net(mr.n_input_vec, hidden_nodes)
optimizer = optim.Adam(params=rnn.parameters())

if Path("net.dat").is_file() and Path("optimizer.dat").is_file():
    print("Found saved state files. Loading.")
    rnn.load_state_dict(torch.load("net.dat"))
    optimizer.load_state_dict(torch.load("optimizer.dat"))

def save_state():
    torch.save(rnn.state_dict(), "net.dat")
    torch.save(optimizer.state_dict(), "optimizer.dat")

def train(message_sequence_tensor):
    criterion = nn.NLLLoss()
    message_sequence = message_sequence_tensor
    hidden = rnn.init_state()

    rnn.zero_grad()

    loss = 0

    for i in range(message_sequence.size()[0] - 1):
        current_char = Variable(message_sequence[i])

        next_char = torch.LongTensor(1)
        next_char[0] = onehot_to_index(message_sequence[i+1])
        next_char = Variable(next_char)

        prediction, hidden = rnn(current_char, hidden)
        loss += criterion(prediction, next_char)

    loss.backward()
    optimizer.step()

    # for p in rnn.parameters():
    #     p.data.add_(-learning_rate, p.grad.data)

    return loss.data[0] / message_sequence_tensor.size()[0]


######################################################################
# To keep track of how long training takes I am adding a
# ``timeSince(timestamp)`` function which returns a human readable string:
#

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


######################################################################
# Training is business as usual - call train a bunch of times and wait a
# few minutes, printing the current time and loss every ``print_every``
# examples, and keeping store of an average loss per ``plot_every`` examples
# in ``all_losses`` for plotting later.
#

n_iters = 100
print_every = 10
plot_every = 5000
save_every = 100
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
    loss = train(mr.random_message_sequence_tensor(n_message_batch))
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

    # if iter % save_every == 0:
    #     print("Saved state.")
    #     save_state()


import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)

def sample(author, length):
    input = mr.author_tensor(author)
    hidden = rnn.init_state()

    output_message = mr.index_to_humanreadable(onehot_to_index(input))

    for i in range(length):
        output, hidden = rnn(Variable(input), hidden)

        output_exped = torch.exp(output.data).view(-1).numpy()
        chanced_letter_index = np.random.choice(mr.n_input_vec, p=output_exped)
        chanced_letter = index_to_onehot(chanced_letter_index, mr.n_input_vec)

        output_message += mr.index_to_humanreadable(chanced_letter_index)
        input = chanced_letter

    return output_message