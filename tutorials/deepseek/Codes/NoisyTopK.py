import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyTopK(nn.Module):
  """
  This is the class for the routing matrix and getting the top k experts.
  """
  def __init__(self, n_embed, n_experts, top_k):
    super().__init__()
    # This is the routing matrix which goes from embedding dim to number of experts and topk
    self.linear = nn.Linear(n_embed, n_experts)
    self.top_k = top_k

    # A bit of noise
    self.noise = nn.Linear(n_embed, n_experts)

  def forward(self, x):
    # Getting the expert selector matrix and then getting topk results from each dimensions
    logits = self.linear(x)
    noise_logits = self.noise(x)
    noisy_logits = logits + noise_logits

    topk_logits, topk_indices = torch.topk(noisy_logits, k=self.top_k, dim=2)

    # we create a same shaped matrix with all being -inf and then wherever the indices are for topk, we leave that and make others -inf
    zeroes = torch.full_like(noisy_logits, float('-inf'))
    sparse_logits = zeroes.scatter(-1, topk_indices, topk_logits)
    router_output = F.softmax(sparse_logits, dim=-1)

    return router_output, topk_indices