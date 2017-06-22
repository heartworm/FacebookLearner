from NetModule import NetModule
from pathlib import Path
import torch

class MessageNet:
    def __init__(self, mr, net_path, optimizer_path):
        self.net_path = net_path
        self.optimizer_path = optimizer_path
        self.mr = mr



        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:



    def load_state(self):
        pass

    def save_state(self):
        pass

    def train(self):
        pass

    def sample(self):
        pass
