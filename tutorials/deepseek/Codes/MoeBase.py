import torch
import torch.nn as nn
import torch.nn.functional as F

import NoisyTopK
import Expert

class MoeBase(nn.Module):

  def __init__(self, embed_dim, n_experts, top_k):
    super().__init__()
    self.router = NoisyTopK(embed_dim, n_experts, top_k)
    self.experts = nn.ModuleList([Expert(embed_dim) for _ in range(n_experts)])
    self.topk = top_k

  def forward(self, x):
    gating_output , indices = self.router(x)
    final_output = torch.zeros_like(x)

    # Reshaping for batch processing
    flat_x = x.view(-1, x.size(-1)) # [batch, seq, emb] -> [batch*seq, emb]
    flat_gatting_output = gating_output.view(-1, gating_output.size(-1)) # [batch, seq, n_experts] -> [batch*seq, n_experts]

    # Processing each expert in parellel
    for i, expert in enumerate(self.experts):
      # Creating a mask where each token is routed to expert i
        # For example, expert_mask = [True, False, False, True, ...]
          # Shape: [batch, seq_len] — one True/False per token
      expert_mask = (indices == i).any(dim=-1) # [batch, seq_len]

      # Flattened to [batch * seq_len] so it matches flat_x
      flat_mask = expert_mask.view(-1) # [batch * seq_len]

      if flat_mask.any():
        # WHERVER we have TRUE in flat_mask, we take those tokens from flat_x, we pass them through expert and we save those tokens
          # At the exact places in final_output where we have true in corespondance to flat_mask
        expert_input = flat_x[flat_mask]
        expert_output = expert(expert_input)

        # Extracting and applying gating scores
        gating_scores = flat_gatting_output[flat_mask, i].unsqueeze(1)
        weighted_expert_output = gating_scores * expert_output

        # putting in weighted expert outputs to the final output matrix
        final_output[expert_mask] += weighted_expert_output.squeeze(1)

    return final_output