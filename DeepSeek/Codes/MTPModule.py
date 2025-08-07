import torch
import torch.nn as nn
import torch.nn.functional as F

from RMSNorm import RMSNorm

class MTPModule(nn.Module):

    def __init__(self, num_heads, d_model, vocab_size):
        """
        Initializing metrics for multi token predicition
        """
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Root-Mean-Square Normalization, Project layers and Transformer block
        self.rms = RMSNorm(self.d_model)
        self.proj = nn.Linear(2*d_model, d_model)
        self.tfmr_blk = [
            nn.TransformerDecoderLayer(d_model, num_heads) for _ in range(num_heads)
        ]

        # Unembeddings metric
        self.unemb = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Passing hidden state x to mtp module
        """
        pass