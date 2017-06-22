import json
from tensor_utils import *
import random

class MessageReader:
    def __init__(self, file_name):

        with open(file_name, 'r') as json_file:
            messages_obj = json.load(json_file)

        self.messages = messages_obj["messages"]
        self.all_authors = messages_obj["authors"]
        self.all_letters = messages_obj["letters"]
        self.n_messages = len(self.messages)
        self.n_letters = len(self.all_letters)
        self.n_authors = len(self.all_authors)
        self.n_input_vec = self.n_letters + self.n_authors + 1 #end of message character

        random.seed(12345678)

    def author_tensor(self, author):
        ind = self.all_authors.index(author)
        return index_to_onehot(ind, self.n_input_vec)

    def character_tensor(self, char):
        try:
            ind = self.all_letters.index(char)
        except ValueError as e:
            print("ValueError when trying to look for character ", char)
            raise e
        return index_to_onehot(self.n_authors + ind, self.n_input_vec)

    def message_tensor(self, message):
        author = self.author_tensor(message["author"])
        message = message["message"]
        eom_tensor = torch.unsqueeze(index_to_onehot(self.n_input_vec - 1, self.n_input_vec), 0)

        out_tensor = torch.unsqueeze(author, 0)
        for char in message:
            char_unsqueezed = torch.unsqueeze(self.character_tensor(char), 0)
            out_tensor = torch.cat((out_tensor, char_unsqueezed), 0)
        out_tensor = torch.cat((out_tensor, eom_tensor), 0)
        return out_tensor

    def index_to_humanreadable(self, ind, newlines=False):
        if ind == (self.n_input_vec - 1):
            return "<END>\n" if newlines else "<END>"
        elif ind < self.n_authors:
            return "<" + self.all_authors[ind] + ">"
        else:
            return self.all_letters[ind - self.n_authors]

    def random_message_sequence_tensor(self, length):
        ind = random.randint(0, self.n_messages - 100)
        out_tensor = None
        cur_len = 0
        while cur_len < length:
            msg_tensor = self.message_tensor(self.messages[ind])
            cur_len += msg_tensor.size()[0]
            if out_tensor is None:
                out_tensor = msg_tensor
            else:
                out_tensor = torch.cat([out_tensor, msg_tensor], 0)
            ind = ind + 1

        if out_tensor is None:
            return torch.zeros(length, 1, self.n_input_vec)
        else:
            return out_tensor[0:length]

    def random_message_sequence_batch(self, length, batches):
        out_tensor = None
        for batch in range(batches):
            new_tensor = self.random_message_sequence_tensor(length)
            if out_tensor is None:
                out_tensor = new_tensor
            else:
                out_tensor = torch.cat([out_tensor, new_tensor], 1)
        return out_tensor

    def message_sequence_to_humanreadable(self, msgseq):
        out_str = ""
        for i in range(msgseq.size()[0]):
            out_str += self.index_to_humanreadable(onehot_to_index(msgseq[i]))
        return out_str