import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
  """
  Expert networkA simple MLP with a linear layer followed by a ReLU activation for each experts.
  """

  def __init__(self, embed_dim, dropout=0.1):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(embed_dim, 4*embed_dim),
        nn.ReLU(),
        nn.Linear(4*embed_dim, embed_dim),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)