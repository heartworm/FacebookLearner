# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from MessageReader import MessageReader
from MessageNet import MessageNet
import numpy as np
import time
import math

mr = MessageReader("messages.json")
net = MessageNet(mr)

if net.load_state():
    print("Found saved state :)")
else:
    print("Didn't find saved state :(")

n_sequence_length = 100 #characters per message sequence
n_batches = 50 #how many msg sequences to train on in parallel

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == "__main__":
    n_iters = 20000 * 12
    print_every = 100
    save_every = 100

    start = time.time()

    for iter in range(1, n_iters + 1):
        loss = net.train(n_sequence_length, n_batches)

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
            random_email = np.random.choice(mr.all_emails)
            index_sequence, _ = net.sample_author(random_email, 50, continuous=True)
            print("".join( [mr.index_to_humanreadable(ind) for ind in index_sequence] ))

        if iter % save_every == 0:
            print("Saving state...")
            net.save_state()


