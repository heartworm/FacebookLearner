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

use_cuda = torch.cuda.is_available()

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

n_message_batch = 1000 #messages per training epoch
n_parallel_batches = 50 #how many msg sequences of length n_message_batch to compute in parallel
hidden_nodes = 500
rnn = Net(mr.n_input_vec, hidden_nodes, use_cuda)
optimizer = optim.Adam(params=rnn.parameters())
criterion = nn.NLLLoss()

if use_cuda:
    rnn.cuda()

if Path("net.dat").is_file() and Path("optimizer.dat").is_file():
    print("Found saved state files. Loading.")
    rnn.load_state_dict(torch.load("net.dat"))
    optimizer.load_state_dict(torch.load("optimizer.dat"))

def save_state():
    torch.save(rnn.state_dict(), "net.dat")
    torch.save(optimizer.state_dict(), "optimizer.dat")

def train(message_sequence):
    tensor_size = message_sequence.size()
    sequence_length = tensor_size[0]
    batches = tensor_size[1]

    hidden = rnn.init_state(batches)

    rnn.zero_grad()

    loss = 0

    for i in range(sequence_length - 1):
        current_char_batch = Variable(message_sequence[i])

        _, next_char = torch.max(message_sequence[i+1], 1)
        next_char = torch.squeeze(next_char)

        if use_cuda:
            next_char = next_char.cuda()

        next_char = Variable(next_char)

        prediction, hidden = rnn(current_char_batch, hidden)
        loss += criterion(prediction, next_char)

    loss.backward()
    optimizer.step()

    return loss.data[0] / message_sequence.size()[0]


import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def sample(author, length, probability_based = True, message_text=None):
    hidden = rnn.init_state(1)
    output_message = ""
    if message_text is not None:
        message_tensor = mr.message_tensor({"author":author, "message":message_text})
        sequence_length = message_tensor.size()[0]
        for n in range(sequence_length - 1):
            _, hidden = rnn(Variable(message_tensor[n]), hidden)
            output_message += mr.index_to_humanreadable(onehot_to_index(message_tensor[n]))
        input = message_tensor[sequence_length - 1]
    else:
        input = mr.author_tensor(author)

    output_message += mr.index_to_humanreadable(onehot_to_index(input))

    for i in range(length):
        output, hidden = rnn(Variable(input), hidden)

        output_exped = torch.exp(output.data)
        output_exped_np = output_exped.cpu().view(-1).numpy()
        chanced_letter_index = np.random.choice(mr.n_input_vec, p=output_exped_np)

        _, max_letter_index = torch.max(output_exped, 1)
        max_letter_index = max_letter_index[0][0]

        letter_index = chanced_letter_index if probability_based else max_letter_index

        output_message += mr.index_to_humanreadable(letter_index)

        input = index_to_onehot(letter_index, mr.n_input_vec)

    return output_message

if __name__ == "__main__":
    n_iters = 20000 * 12
    print_every = 100
    save_every = 100
    all_losses = []
    total_loss = 0 # Reset every plot_every iters

    start = time.time()

    for iter in range(1, n_iters + 1):
        loss = train(mr.random_message_sequence_batch(n_message_batch,n_parallel_batches))
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
            print(sample(np.random.choice(mr.all_authors), 50))

        if iter % save_every == 0:
            print("Saved state.")
            save_state()


