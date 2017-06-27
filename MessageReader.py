import json
from tensor_utils import *
import random

class MessageReader:
    def __init__(self, file_name):

        with open(file_name, 'r') as json_file:
            messages_obj = json.load(json_file)

        self.messages = messages_obj["messages"]
        self.author_name_list = messages_obj["authors"]
        self.all_emails = [author["email"] for author in self.author_name_list]
        self.all_names = [author["name"] for author in self.author_name_list]
        self.all_letters = messages_obj["letters"]
        self.n_messages = len(self.messages)
        self.n_letters = len(self.all_letters)
        self.n_authors = len(self.all_emails)
        self.n_input_vec = self.n_letters + self.n_authors + 1 #end of message character
        self.end_index = self.n_input_vec - 1

        random.seed(12345678)

    def author_index(self, email):
        return self.all_emails.index(email)

    def character_index(self, char):
        try:
            ind = self.all_letters.index(char)
        except ValueError as e:
            print("ValueError when trying to look for character ", char)
            raise e
        return self.n_authors + ind

    def message_index_sequence(self, message):
        sequence = [self.author_index(message["email"])]
        sequence += [self.character_index(char) for char in message["message"]]
        sequence.append(self.end_index)
        return sequence

    def index_sequence_to_messages(self, index_sequence):
        in_message = False
        current_message = ""
        email = ""
        messages = []

        for ind in index_sequence:
            if ind < self.n_authors:
                in_message = True
                current_message = ""
                email = self.all_emails[ind]
            elif in_message:
                if ind == self.end_index:
                    messages.append({
                        "message": current_message,
                        "email": email
                    })
                    in_message = False
                else:
                    current_message += self.index_to_humanreadable(ind)

        if in_message:
            messages.append({
                "message": current_message + "<TRUNCATED>",
                "email": email
            })

        return messages

    def index_to_humanreadable(self, ind, newlines=False):
        if ind == (self.n_input_vec - 1):
            return "<END>\n" if newlines else "<END>"
        elif ind < self.n_authors:
            return "<" + self.author_name_list[ind]["name"] + ">"
        else:
            return self.all_letters[ind - self.n_authors]

    def random_message_sequence(self, length):
        ind = random.randint(0, self.n_messages - 100)
        out_tensor = None
        cur_len = 0
        while cur_len < length:
            msg_tensor = index_sequence_to_onehot_sequence(self.message_index_sequence(self.messages[ind]))
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
            new_tensor = self.random_message_sequence(length)
            if out_tensor is None:
                out_tensor = new_tensor
            else:
                out_tensor = torch.cat([out_tensor, new_tensor], 1)
        return out_tensor

    def message_sequence_to_humanreadable(self, sequence):
        out_str = ""
        for i in range(sequence.size()[0]):
            out_str += self.index_to_humanreadable(onehot_to_index(sequence[i]))
        return out_str