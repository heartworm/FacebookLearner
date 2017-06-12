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

    def index_to_humanreadable(self, ind):
        if ind == (self.n_input_vec - 1):
            return "<END>"
        elif ind < self.n_authors:
            return "<" + self.all_authors[ind] + ">"
        else:
            return self.all_letters[ - self.n_authors]

    def random_message_sequence_tensor(self, length):
        start_ind = random.randint(0, self.n_messages - length)
        out_tensor = None
        for ind in range(start_ind, start_ind + length):
            msg = self.message_tensor(self.messages[ind])
            if out_tensor is None:
                out_tensor = msg
            else:
                out_tensor = torch.cat((out_tensor, msg), 0)
        return out_tensor
