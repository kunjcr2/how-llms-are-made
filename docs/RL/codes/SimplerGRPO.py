"""
Load three models. Live X (trains), snapshot Y (frozen copy), reference Z (frozen forever).

Generate 8 completions from Y. Save log π_old per token -> [8, 500].

Score them. Rewards -> group-normalized advantages -> [8].

Forward pass completions through X, grads ON  -> log π_θ  [8, 500].
Forward pass completions through Z, no_grad    -> log π_ref [8, 500].

ratio = exp(log π_θ - log π_old)     # how far X moved from Y
KL    = divergence(X || Z)           # how far X moved from Z

Objective per token = clipped(ratio × advantage) - β × KL
  advantage: push good tokens up, bad ones down
  clip:      cap the step size vs Y
  KL:        stay anchored to Z

loss = -mean over all tokens, all generations
backward, step. Then Y <- X. Z never changes.
"""

import torch
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    generations: int = 8

class GRPOTrainer:
    
    def __init__(self, config: Config):
        self.config = config

    def ppo_loss(self):
        pass

    def advantage(self) -> List[torch.tensor]:
        """based on generations, calculates the advantage."""

        generations = self.generate()
        mean = torch.mean(generations)
        std = torch.std(generations) + 1e-8

        advantages = (generations - mean)/std

        return [generations, advantages]

    def generate(self) -> torch.tensor:
        """Generates random rewards between 0 and 5 based on generations."""
        # replace with actual generations and reward model integrations to get the reward for the actual response
        return torch.randint(low=0, high=6, size=(self.config.generations, ), dtype=torch.float64)

    def get_weights(self):
        """Genrates just random weights. Normal and reference model (frozen). same as of now."""
        return torch.rand(size=(10, 10), dtype=torch.float64)

if __name__=="__main__":
    tr = GRPOTrainer(Config())

    adv = tr.advantage()
    print(f"Generations:\n{adv[0]}\n\nAdvantages:\n{adv[1]}")