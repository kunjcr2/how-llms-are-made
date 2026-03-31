"""
mamba_train.py — Train a Mamba sequence model from scratch (character-level).

This script implements the core Mamba architecture (Gu & Dao, 2023) in pure
PyTorch and trains it on a tiny text dataset so you can see the full
pipeline: tokenization → Mamba blocks → cross-entropy loss → generation.

Requirements:
    pip install torch

Usage:
    python mamba_train.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ─
# Config
# ─

@dataclass
class MambaConfig:
    vocab_size: int = 256          # character-level (extended ASCII)
    d_model: int = 128             # model dimension
    n_layers: int = 4              # number of Mamba blocks
    d_state: int = 16              # SSM state dimension (N)
    d_conv: int = 4                # causal conv1d kernel size
    expand: int = 2                # expansion factor (E)
    dt_rank: str | int = "auto"    # rank of Δ projection ("auto" = d_model // 16)
    bias: bool = False             # linear layer bias
    seq_len: int = 128             # training sequence length
    batch_size: int = 32
    lr: float = 3e-4
    n_steps: int = 2000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.d_inner = self.d_model * self.expand
        if self.dt_rank == "auto":
            self.dt_rank = max(self.d_model // 16, 1) # 8


# ─
# Selective SSM (core of Mamba)
# ─

class SelectiveSSM(nn.Module):
    """
    The selective scan: input-dependent B, C, Δ applied to a
    discretized state space recurrence.

    For each channel d in [0, d_inner):
        h_t[d] = A_bar[d] * h_{t-1}[d] + B_bar[t] * x_t[d]
        y_t[d] = C[t] · h_t[d]

    Where A_bar, B_bar come from zero-order-hold discretization
    using the input-dependent step size Δ.
    """

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.d_inner = cfg.d_inner
        self.d_state = cfg.d_state
        self.dt_rank = cfg.dt_rank

        #  Low-rank Δ expansion -> adds some static knowledge to delta which is not based on input
        self.dt_proj = nn.Linear(cfg.dt_rank, cfg.d_inner, bias=True)

        #  Projects x → (Δ_rank, B, C) jointly
        # You take the input and put it up to delta+B+C dimension and then break it into 3 parts later
        self.x_proj = nn.Linear(cfg.d_inner, cfg.dt_rank + 2 * cfg.d_state, bias=False)

        #  A: diagonal HiPPO-style init, stored in log space 
        A = torch.arange(1, cfg.d_state + 1, dtype=torch.float32).unsqueeze(0) # (1, N)
        A = A.expand(cfg.d_inner, -1)  # (d_inner, N)
        self.A_log = nn.Parameter(torch.log(A)) # <- This needs to be positive, so we take log

        #  D: skip connection 
        self.D = nn.Parameter(torch.ones(cfg.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_inner)
        returns: (B, L, d_inner)
        """
        B, L, D = x.shape

        #  Project to get Δ (low-rank), B_input, C_input 
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        delta, B_input, C_input = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        #  Expand Δ from low-rank and apply softplus 
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)

        #  Recover A (always negative for stability) 
        A = -torch.exp(self.A_log)  # (d_inner, N) <- and this becomes ALWAYS NEGATIVE

        #  Discretize: zero-order hold 
        # A_bar = exp(Δ * A)  →  (B, L, d_inner, N)
        A_bar = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )
        # B_bar = Δ * B  →  (B, L, d_inner, N)
        B_bar = delta.unsqueeze(-1) * B_input.unsqueeze(2)

        #  Sequential scan 
        # (For educational clarity; production Mamba uses a parallel scan.)
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            #  h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            #  y_t = C_t · h_t   (dot product over state dim)
            y_t = (C_input[:, t].unsqueeze(1) * h).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        #  Skip connection 
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y


# ─
# Mamba Block
# ─

class MambaBlock(nn.Module):
    """
    A single Mamba block:
        x → [in_proj → split → conv1d+SiLU → SSM] ⊙ [SiLU gate] → out_proj
    with a residual connection and RMSNorm.
    """

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        d_inner = cfg.d_inner

        #  Normalization 
        self.norm = RMSNorm(cfg.d_model)

        #  Input projection (both SSM path and gate) 
        self.in_proj = nn.Linear(cfg.d_model, 2 * d_inner, bias=cfg.bias)

        #  Causal depthwise conv1d 
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=cfg.d_conv,
            padding=cfg.d_conv - 1,     # causal padding (we truncate below)
            groups=d_inner,              # depthwise
            bias=True,
        )

        #  Selective SSM 
        self.ssm = SelectiveSSM(cfg)

        #  Output projection 
        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        returns: (B, L, D)
        """
        residual = x
        x = self.norm(x)

        #  Dual projection 
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        #  Causal Conv1D 
        # Conv1d expects (B, C, L) but x is (B, L, C), so swap the last two dims
        x_ssm = x_ssm.transpose(1, 2)
        # Pad d_conv-1 zeros on both sides (symmetric), then slice [:L] to keep only causal outputs
        # This ensures y_t depends only on x_t and earlier — future tokens are discarded
        x_ssm = self.conv1d(x_ssm)[:, :, :x.shape[1]]
        # Swap back to (B, L, d_inner) for the SSM
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = F.silu(x_ssm)       

        #  Selective SSM 
        y = self.ssm(x_ssm) # attention for mamba

        #  Gated output 
        y = y * F.silu(z) # element wise multiplication
        out = self.out_proj(y)

        return out + residual


# ─
# RMSNorm
# ─

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ─
# Full Mamba Language Model
# ─

class MambaLM(nn.Module):
    """
    Stacks N MambaBlocks with an embedding layer and an LM head.
    """

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([MambaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)
        print(f"MambaLM — {sum(p.numel() for p in self.parameters()):,} parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, L) — token indices
        returns: logits (B, L, vocab_size)
        """
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, 
                    prompt_ids: torch.Tensor, 
                    max_new_tokens: int = 200,
                    temperature: float = 0.8, 
                    top_k: int = 40) -> torch.Tensor:
        """
        Autoregressive generation from a prompt.
        prompt_ids: (1, T)
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Take last seq_len tokens if context is too long
            context = ids[:, -self.cfg.seq_len:]
            logits = self(context) # i didnt know we can do that
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                # not sure what this does, assuming it makes all tokens except the top k = -inf
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)

        return ids


# ─
# Dataset: Character-Level Text
# ─

class CharDataset(torch.utils.data.Dataset):
    """
    Wraps a raw text string into overlapping (input, target) chunks
    for next-character prediction training.
    """

    def __init__(self, text: str, seq_len: int):
        self.data = torch.tensor([ord(c) for c in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(len(self.data) - self.seq_len - 1, 0)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:] # (First seq_len-1), (Last seq_len-1)


# ─
# Training Corpus
# ─

CORPUS = """
The Mamba architecture represents a paradigm shift in sequence modeling. Unlike Transformers 
that compute pairwise attention between every token pair at quadratic cost, Mamba processes 
sequences through a selective state space model that runs in linear time.

At its core, Mamba maintains a hidden state — a fixed-size vector that compresses the entire 
history of the sequence. Each new token updates this state through learned, input-dependent 
gates. The model decides what to remember and what to forget based on the content itself, 
not just the position.

The key innovation is the selective mechanism. In classical state space models, the matrices 
A, B, and C are constant — every token is processed identically. Mamba makes B, C, and the 
discretization step Delta functions of the current input. This gives the model the ability 
to selectively focus on relevant tokens, similar to attention, but without the quadratic cost.

Training is made efficient through a hardware-aware parallel scan algorithm that exploits 
the associative property of the recurrence. Instead of computing states sequentially, Mamba 
uses a tree-reduction approach that achieves O(log L) depth on modern GPUs.

During inference, the model reverts to its recurrent form. Each new token requires only a 
constant-time state update — the hidden state is always the same size regardless of how long 
the conversation has been. This makes Mamba ideal for streaming applications and scenarios 
with very long contexts.

Recent experiments show that Mamba matches Transformer performance on language modeling 
benchmarks up to the billion-parameter scale, while being significantly faster at inference. 
Hybrid architectures that combine Mamba layers with sparse attention layers — such as Jamba 
by AI21 — push performance even further by leveraging the strengths of both paradigms.

The state space model framework also extends naturally to other modalities. Mamba has shown 
strong results in audio modeling, DNA sequence analysis, and time series forecasting, where 
the ability to handle very long sequences with constant memory is particularly valuable.

Perhaps the most exciting aspect of Mamba is what it proves: attention is not all you need. 
Sequence modeling can be done without computing pairwise token interactions, as long as the 
model has a sufficiently expressive mechanism for selectively compressing information into 
its running state.

In summary, Mamba achieves the selectivity of attention mechanisms through input-dependent 
state space parameters, while maintaining the efficiency of recurrent models. It combines 
the best of both worlds: content-aware processing with linear-time complexity, making it a 
compelling alternative for next-generation sequence models.
""".strip()


# ─
# Training Loop
# ─

def train():
    cfg = MambaConfig()
    print(f"Device: {cfg.device}")
    print(f"Corpus length: {len(CORPUS):,} characters\n")

    #  Dataset & Loader 
    dataset = CharDataset(CORPUS, cfg.seq_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    #  Model & Optimizer 
    model = MambaLM(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_steps)

    #  Training 
    model.train()
    step = 0
    running_loss = 0.0

    print("=" * 60)
    print("Training started")
    print("=" * 60)

    while step < cfg.n_steps:
        for inputs, targets in loader:
            if step >= cfg.n_steps:
                break

            inputs = inputs.to(cfg.device)
            targets = targets.to(cfg.device)

            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step += 1

            if step % 100 == 0:
                avg = running_loss / 100
                lr = scheduler.get_last_lr()[0]
                print(f"  step {step:5d} | loss {avg:.4f} | lr {lr:.2e}")
                running_loss = 0.0

    print("=" * 60)
    print("Training complete!\n")

    #  Generation 
    prompts = ["The Mamba", "During in", "At its co"]
    for prompt in prompts:
        prompt_ids = torch.tensor(
            [[ord(c) for c in prompt]], dtype=torch.long, device=cfg.device
        )
        generated_ids = model.generate(prompt_ids, max_new_tokens=200, temperature=0.7)
        text = "".join(chr(min(c, 127)) for c in generated_ids[0].tolist())
        print(f"Prompt: '{prompt}'")
        print(f"Generated:\n{text}\n")
        print("-" * 60)


if __name__ == "__main__":
    train()
