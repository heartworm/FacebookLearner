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

    def sample(self, message, length, probability_based=True, continuous=False, hidden=None):

        # For continuous chats context is persistent and will be passed in to the function
        if hidden is None:
            hidden = self.net.init_state(1)

        index_list = []

        if message["message"] is not None:
            message_tensor = self.mr.message_sequence(message)
            sequence_length = message_tensor.size()[0]
            for n in range(sequence_length - 1):
                _, hidden = self.net(Variable(message_tensor[n]), hidden)
            input = message_tensor[sequence_length - 1]
        else:
            author_ind = self.mr.author_index(message["email"])
            input = index_to_onehot(author_ind, self.mr.n_input_vec)
            index_list.append(author_ind)



        for i in range(length):
            output, hidden = self.net(Variable(input), hidden)

            output_exped = torch.exp(output.data)
            output_exped_np = output_exped.cpu().view(-1).numpy()

            # Using output_exped as probability distribution, choose a random letter index
            chanced_index = np.random.choice(self.mr.n_input_vec, p=output_exped_np)

            # Select a letter index using argmax, this can result in screwed up things like endless "hahahahahahaha...."
            _, max_index = torch.max(output_exped, 1)
            max_index = max_index[0][0]

            letter_index = chanced_index if probability_based else max_index
            index_list.append(letter_index)

            input = index_to_onehot(letter_index, self.mr.n_input_vec)

            if not continuous and letter_index == self.mr.end_index:
                break

        return index_list, hidden

