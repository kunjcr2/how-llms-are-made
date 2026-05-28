"""
QWEN3.5 ARCHITECTURE
Source: NVIDIA blog (Mar 2026), HuggingFace model card
No technical report yet.
Early Fusion: No separate vision encoder. Hybrid GDN + GatedAttention + MoE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)


class SwiGLU_MLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim=None):
        super().__init__()
        self.intermediate_dim = intermediate_dim or int(hidden_dim * 8 / 3)
        self.W_gate = nn.Linear(hidden_dim, self.intermediate_dim, bias=False)
        self.W_up   = nn.Linear(hidden_dim, self.intermediate_dim, bias=False)
        self.W_down = nn.Linear(self.intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.W_gate(x))
        up   = self.W_up(x)
        return self.W_down(gate * up)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, hidden_dim=7168):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, image):
        # image: (B, 3, H, W)
        x = self.proj(image)               # (B, hidden_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)   # (B, num_patches, hidden_dim)
        return x


class GatedDeltaNet(nn.Module):
    def __init__(self, hidden_dim, d_k=128, d_v=128, num_heads=56):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_q    = nn.Linear(hidden_dim, num_heads * d_k, bias=False)
        self.W_k    = nn.Linear(hidden_dim, num_heads * d_k, bias=False)
        self.W_v    = nn.Linear(hidden_dim, num_heads * d_v, bias=False)
        self.W_gate = nn.Linear(hidden_dim, num_heads * d_k, bias=True)
        self.W_o    = nn.Linear(num_heads * d_v, hidden_dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        H, dk, dv = self.num_heads, self.d_k, self.d_v

        Q = self.W_q(x).view(B, N, H, dk)                     # (B, N, H, dk)
        K = F.normalize(self.W_k(x).view(B, N, H, dk), dim=-1)
        V = self.W_v(x).view(B, N, H, dv)                     # (B, N, H, dv)
        G = torch.sigmoid(self.W_gate(x).view(B, N, H, dk))   # (B, N, H, dk)

        # Per-head recurrent state: (B, H, dk, dv)
        state = torch.zeros(B, H, dk, dv, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(N):
            q_t = Q[:, t]                                      # (B, H, dk)
            k_t = K[:, t]                                      # (B, H, dk)
            v_t = V[:, t]                                      # (B, H, dv)
            g_t = G[:, t].unsqueeze(-1)                        # (B, H, dk, 1)

            # Delta rule update
            pred = torch.einsum('bhk,bhkv->bhv', k_t, state)  # (B, H, dv)
            delta = torch.einsum('bhk,bhv->bhkv', k_t, v_t - pred)
            state = g_t * state + delta

            # Read
            out_t = torch.einsum('bhk,bhkv->bhv', q_t, state) # (B, H, dv)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1)                      # (B, N, H, dv)
        out = out.reshape(B, N, H * dv)
        return self.W_o(out)


class GatedFullAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.W_q    = nn.Linear(hidden_dim, hidden_dim, bias=True)   # QKV bias
        self.W_k    = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_v    = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_o    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_gate = nn.Linear(hidden_dim, hidden_dim, bias=True)   # output gate

    def forward(self, x):
        B, N, C = x.shape
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        # MRoPE would be applied here

        scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn   = F.softmax(scores, dim=-1)
        out    = (attn @ V).transpose(1, 2).reshape(B, N, C)
        out    = self.W_o(out)

        gate = torch.sigmoid(self.W_gate(x))
        return gate * out


class MoELayer(nn.Module):
    def __init__(self, hidden_dim, num_experts=512, top_k=10, num_shared_experts=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.router      = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts     = nn.ModuleList([SwiGLU_MLP(hidden_dim) for _ in range(num_experts)])
        self.shared      = nn.ModuleList([SwiGLU_MLP(hidden_dim) for _ in range(num_shared_experts)])

    def forward(self, x):
        B, N, C = x.shape
        x_flat  = x.view(-1, C)
        logits  = self.router(x_flat)
        scores  = F.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        routed_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (topk_indices[:, k] == e)
                if mask.any():
                    routed_out[mask] += topk_weights[mask, k:k+1] * self.experts[e](x_flat[mask])

        shared_out = sum(s(x_flat) for s in self.shared)
        return (routed_out + shared_out).view(B, N, C)


class HybridTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, layer_idx,
                 full_attn_every_n=5, num_experts=512, top_k=10, num_shared_experts=1):
        super().__init__()
        self.norm_1 = RMSNorm(hidden_dim)
        self.norm_2 = RMSNorm(hidden_dim)
        use_full_attn = ((layer_idx + 1) % full_attn_every_n == 0)
        self.attn = (GatedFullAttention(hidden_dim, num_heads)
                     if use_full_attn
                     else GatedDeltaNet(hidden_dim, d_k=hidden_dim // num_heads,
                                        d_v=hidden_dim // num_heads, num_heads=num_heads))
        self.ffn = MoELayer(hidden_dim, num_experts, top_k, num_shared_experts)

    def forward(self, x):
        x = x + self.attn(self.norm_1(x))
        x = x + self.ffn(self.norm_2(x))
        return x


class MultiTokenPredictionHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size=248320, num_future_tokens=4):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, vocab_size, bias=False)
                                    for _ in range(num_future_tokens)])

    def forward(self, hidden_states):
        return [head(hidden_states) for head in self.heads]


class Qwen35(nn.Module):
    def __init__(self, hidden_dim=7168, vocab_size=248320, num_layers=60,
                 num_heads=56, num_experts=512, top_k=10, num_shared_experts=1):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size=16, hidden_dim=hidden_dim)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            HybridTransformerBlock(
                hidden_dim=hidden_dim, num_heads=num_heads, layer_idx=i,
                full_attn_every_n=5, num_experts=num_experts,
                top_k=top_k, num_shared_experts=num_shared_experts,
            )
            for i in range(num_layers)
        ])
        self.norm    = RMSNorm(hidden_dim)
        self.lm_head = MultiTokenPredictionHead(hidden_dim, vocab_size, num_future_tokens=4)

    def forward(self, text_tokens, images=None):
        x = self.token_embed(text_tokens)             # (B, N_text, hidden_dim)

        if images is not None:
            visual_tokens = self.patch_embed(images)   # (B, N_patches, hidden_dim)
            x = torch.cat([visual_tokens, x], dim=1)  # early fusion: prepend patches

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.lm_head(x)


MODEL_SPECS = {
    "total_params":     "397B",
    "active_params":    "17B (4.28%)",
    "num_experts":      512,
    "active_experts":   "10 routed + 1 shared = 11",
    "num_layers":       60,
    "hidden_dim":       7168,
    "vocab_size":       248320,
    "context":          "256K (extensible to 1M)",
    "languages":        201,
    "attention":        "Hybrid GatedDeltaNet + GatedFullAttention",
    "training":         "Early fusion from scratch on interleaved multimodal tokens",
    "precision":        "FP8",
    "multi_token_pred": 4,
}