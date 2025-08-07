import torch
import torch.nn as nn
import torch.nn.functional as F

class MTPModule(nn.Module):

    def __init__(self, num_heads):
        """
        Initializing metrics for multi token predicition
        """
        self.num+heads = num_heads

    def forward(self, x):
        """
        Passing hidden state x to mtp module
        """
        pass