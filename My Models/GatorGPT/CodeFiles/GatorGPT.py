import torch
import torch.nn as nn

from Block import Block

class GatorGPT(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            d_model: int = 384,
            n_heads: int = 8,
            gqa_groups: int = 2,
            max_len: int = 1024,
            d_ff: int = 960,
            eps: float = 1e-5,
            dropout_p: float = 0.0,
            blocks: int = 10,
        ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.unembed = nn.Linear(d_model, vocab_size)

        self.final_rms = nn.modules.normalization.RMSNorm(d_model, eps)

        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    gqa_groups=gqa_groups,
                    max_len=max_len,
                    d_ff=d_ff,
                    eps=eps,
                    dropout_p=dropout_p
                ) for _ in range(blocks)
            ]
        )

    def forward(self, x):
        """
        Forward method that takes in the tokens
        """
        return self.unembed(self.final_rms(self.blocks(self.embed(x))))