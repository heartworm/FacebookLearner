from NetModule import NetModule
from pathlib import Path
from tensor_utils import *
from torch.autograd import Variable
import numpy as np
import torch

class MessageNet:
    def __init__(self, mr, net_path="net.dat", optimizer_path="optimizer.dat"):
        self.net_path = net_path
        self.optimizer_path = optimizer_path
        self.mr = mr

        hidden_nodes = 500
        hidden_layers = 3

        self.use_cuda = torch.cuda.is_available()
        self.net = NetModule(mr.n_input_vec, hidden_nodes, n_layers=hidden_layers, use_cuda=self.use_cuda)
        self.optimizer = torch.optim.Adam(params=self.net.parameters())
        self.criterion = torch.nn.NLLLoss()

    def load_state(self):
        if Path(self.net_path).is_file() and Path(self.optimizer_path).is_file():
            self.net.load_state_dict(torch.load(self.net_path))
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
            return True
        return False

    def save_state(self):
        torch.save(self.net.state_dict(), self.net_path)
        torch.save(self.optimizer.state_dict(), self.optimizer_path)

    def train(self, sequence_length, batches):
        message_sequence = self.mr.random_message_sequence_batch(sequence_length, batches)
        hidden = self.net.init_state(batches)
        self.net.zero_grad()

        loss = 0

        for i in range(sequence_length - 1):
            current_char_batch = Variable(message_sequence[i])

            _, next_char = torch.max(message_sequence[i + 1], 1)
            next_char = torch.squeeze(next_char)

            if self.use_cuda:
                next_char = next_char.cuda()

            next_char = Variable(next_char)

            prediction, hidden = self.net(current_char_batch, hidden)
            loss += self.criterion(prediction, next_char)

        loss.backward()
        self.optimizer.step()

        return loss.data[0] / message_sequence.size()[0]

    def sample(self, author, length, probability_based=True, message_text=None, newlines=True):
        hidden = self.net.init_state(1)
        output_message = ""
        if message_text is not None:
            message_tensor = self.mr.message_tensor({"author": author, "message": message_text})
            sequence_length = message_tensor.size()[0]
            for n in range(sequence_length - 1):
                _, hidden = self.net(Variable(message_tensor[n]), hidden)
                output_message += self.mr.index_to_humanreadable(onehot_to_index(message_tensor[n]), newlines=newlines)
            input = message_tensor[sequence_length - 1]
        else:
            input = self.mr.author_tensor(author)

        output_message += self.mr.index_to_humanreadable(onehot_to_index(input), newlines=True)

        for i in range(length):
            output, hidden = self.net(Variable(input), hidden)

            output_exped = torch.exp(output.data)
            output_exped_np = output_exped.cpu().view(-1).numpy()
            chanced_letter_index = np.random.choice(self.mr.n_input_vec, p=output_exped_np)

            _, max_letter_index = torch.max(output_exped, 1)
            max_letter_index = max_letter_index[0][0]

            letter_index = chanced_letter_index if probability_based else max_letter_index

            output_message += self.mr.index_to_humanreadable(letter_index, newlines=True)

            input = index_to_onehot(letter_index, self.mr.n_input_vec)

        return output_message
