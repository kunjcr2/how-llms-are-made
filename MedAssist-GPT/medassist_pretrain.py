"""
MedAssist-GPT: Complete Medical LLM Pretraining Script
=====================================================
Modern architecture with RoPE, GQA, SwiGLU, RMSNorm
Optimized for A100 GPU with Flash Attention
Automatic checkpointing and HuggingFace uploads
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.optim.lr_scheduler import OneCycleLR

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import tiktoken
import wandb
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login, create_repo, upload_folder, HfApi
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    "vocab_size": 50257,
    "d_model": 512,
    "n_heads": 8,
    "gqa_groups": 2,        # 4 KV heads (8 query heads / 2 groups)
    "max_len": 1024,
    "d_ff": 2048,           # 4x hidden dimension
    "eps": 1e-5,
    "dropout_p": 0.0,       # No dropout during pretraining
    "blocks": 12,           # ~500M parameters
}

TRAINING_CONFIG = {
    "batch_size": 64,
    "max_length": 1024,
    "stride": 1024,
    "gradient_accumulation_steps": 2,  # Effective batch size: 128
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "warmup_steps": 500,
    "max_steps": 15000,      # ~5B tokens in 7 hours on A100
    "eval_freq": 500,
    "eval_iter": 100,
    "save_freq": 500,
    "grad_clip": 1.0,
    "num_workers": 4,
    "seed": 42,
}

DATA_CONFIG = {
    "dataset_name": "pubmed",  # or "scientific_papers"
    "train_split": 0.95,
    "max_train_samples": 100_000,  # Adjust based on time/compute
    "streaming": False,  # Set True for very large datasets
}

WANDB_CONFIG = {
    "project": "MedAssist-GPT-Pretraining",
    "entity": kunjcr2,  # Your wandb username
    "name": "medassist-500M-run1",
}

HF_CONFIG = {
    "repo_id": "kunjcr2/MedAssist-GPT-500M",  # Change this!
    "upload_checkpoints": True,
    "upload_frequency": 1000,  # Upload every N steps
}

# ============================================================================
# ARCHITECTURE COMPONENTS
# ============================================================================

class RoPE(nn.Module):
    """Rotary Position Embeddings (RoPE)"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Position indices - tensor (0,1,2,...,max_len) of size (max_len, 1)
        self.register_buffer('position_ids', torch.arange(max_len).unsqueeze(1))

        # Frequency terms
        self.register_buffer(
            'div_term',
            torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
            # e^(2i*(-log(10000))/d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Get positions
        position_ids = self.position_ids[:seq_len] # (seq_len, 1)
        
        # Calculate angles
        angles = position_ids * self.div_term # (seq_len, d_model/2)
        cos_vals = torch.cos(angles) 
        sin_vals = torch.sin(angles) 
        
        # Reshape for rotation
        x_pairs = x.view(batch_size, seq_len, d_model // 2, 2) # (b, s, d//2, 2)
        x_even = x_pairs[..., 0] # (b, s, d//2)
        x_odd = x_pairs[..., 1] # (b, s, d//2)
        
        # Apply rotation
        rotated_even = x_even * cos_vals - x_odd * sin_vals
        rotated_odd = x_even * sin_vals + x_odd * cos_vals
        
        # Reconstruct
        rotated_pairs = torch.stack([rotated_even, rotated_odd], dim=-1) # (b, s, d//2, 2)
        rotated_x = rotated_pairs.view(batch_size, seq_len, d_model) # (b, s, d)
        
        return rotated_x


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) with RoPE"""
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        gqa_groups: int = 2,
        max_len: int = 1024,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % gqa_groups == 0, "n_heads must be divisible by gqa_groups"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.gqa_groups = gqa_groups
        self.head_dim = d_model // n_heads
        self.n_kv_heads = n_heads // gqa_groups
        self.max_len = max_len
        
        # Projections (bias-free)
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        
        # RoPE for Q and K
        self.rope_q = RoPE(d_model=n_heads * self.head_dim, max_len=max_len)
        self.rope_k = RoPE(d_model=self.n_kv_heads * self.head_dim, max_len=max_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)  # (B, T, H*D)
        k = self.k_proj(x)  # (B, T, H_kv*D)
        v = self.v_proj(x)  # (B, T, H_kv*D)
        
        # Apply RoPE
        q = self.rope_q(q)
        k = self.rope_k(k)
        
        # Reshape to heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, H_kv, T, D)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, H_kv, T, D)
        
        # Expand K and V for GQA
        expand_factor = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(expand_factor, dim=1)  # (B, H, T, D)
        v = v.repeat_interleave(expand_factor, dim=1)  # (B, H, T, D)
        
        # Scaled dot-product attention with Flash Attention if available
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )
        
        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        
        # Output projection
        out = self.o_proj(out)
        
        return out


class SwiGLU_MLP(nn.Module):
    """SwiGLU Feed-Forward Network"""
    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super().__init__()
        # Fused up + gate projection
        self.w1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        # Down projection
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up, gate = self.w1(x).chunk(2, dim=-1) # breaks it into 2 parts - (b, s, d_ff)
        x = up * F.silu(gate)  # SwiGLU activation - (b,s,d_ff) * (b,s,d_ff) = (b,s,d_ff)
        x = self.w2(x)  # (b,s,d_model)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and residual connections"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.rms1 = nn.RMSNorm(config["d_model"], eps=config["eps"])
        self.rms2 = nn.RMSNorm(config["d_model"], eps=config["eps"])
        
        self.attn = GroupedQueryAttention(
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            gqa_groups=config["gqa_groups"],
            max_len=config["max_len"]
        )
        
        self.mlp = SwiGLU_MLP(
            d_model=config["d_model"],
            d_ff=config["d_ff"]
        )
        
        self.dropout = nn.Dropout(config["dropout_p"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.dropout(self.attn(self.rms1(x)))
        # Pre-norm MLP
        x = x + self.dropout(self.mlp(self.rms2(x)))
        return x


class MedAssistGPT(nn.Module):
    """Main model class"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed = nn.Embedding(config["vocab_size"], config["d_model"])
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config["blocks"])
        ])
        
        # Final RMSNorm
        self.final_rms = nn.RMSNorm(config["d_model"], eps=config["eps"])
        
        # Language model head (weight-tied with embeddings)
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        self.lm_head.weight = self.embed.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        h = self.embed(input_ids)  # (batch, seq_len, d_model)
        
        # Pass through transformer blocks
        for block in self.blocks:
            h = block(h)
        
        # Final normalization
        h = self.final_rms(h)
        
        # Language model head
        logits = self.lm_head(h)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# DATA LOADING
# ============================================================================

class MedicalDataset(Dataset):
    """Fast dataset with pre-computed sliding windows"""
    def __init__(self, tokens: np.ndarray, max_length: int = 1024, stride: int = 1024):
        self.tokens = np.array(tokens, dtype=np.int32)
        self.max_length = max_length
        
        # Pre-compute all valid start positions
        self.starts = np.arange(0, len(tokens) - max_length, stride)
    
    def __len__(self) -> int:
        return len(self.starts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        end = start + self.max_length
        
        input_ids = torch.from_numpy(self.tokens[start:end].copy()).long()
        target_ids = torch.from_numpy(self.tokens[start+1:end+1].copy()).long()
        
        return input_ids, target_ids


def tokenize_batch(text_batch: List[str], tokenizer) -> List[int]:
    """Tokenize a batch of texts"""
    joined = "\n\n".join(text_batch)
    return tokenizer.encode(joined, allowed_special={"<|endoftext|>"})


def prepare_medical_data(config: Dict[str, Any], tokenizer):
    """Load and tokenize medical dataset"""
    print("🔥 Loading medical dataset...")
    
    # Load dataset
    if config["streaming"]:
        dataset = load_dataset(
            config["dataset_name"],
            split="train",
            streaming=True
        )
        # Take first N samples for streaming
        dataset = dataset.take(config["max_train_samples"])
        data = pd.DataFrame(list(dataset))
    else:
        dataset = load_dataset(config["dataset_name"], split="train")
        # Limit samples
        max_samples = min(config["max_train_samples"], len(dataset))
        data = pd.DataFrame(dataset[:max_samples])
    
    print(f"📊 Loaded {len(data)} documents")
    
    # Extract text column (adjust based on dataset)
    if 'article' in data.columns:
        texts = data['article'].tolist()
    elif 'MedlineCitation' in data.columns:  # For PubMed
        texts = data['MedlineCitation'].apply(
            lambda x: x.get('Article', {}).get('Abstract', {}).get('AbstractText', [''])[0]
        ).tolist()
    else:
        raise ValueError(f"Unknown text column in dataset. Available: {data.columns.tolist()}")
    
    # Split train/val
    split_idx = int(config["train_split"] * len(texts))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    print(f"🔥 Tokenizing {len(train_texts)} training documents...")
    
    # Tokenize in parallel
    def process_split(text_list: List[str], chunk_size: int = 1000):
        chunks = [text_list[i:i+chunk_size] for i in range(0, len(text_list), chunk_size)]
        
        all_tokens = []
        # Multiple cores used
        with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), 8)) as executor:
            futures = [executor.submit(tokenize_batch, chunk, tokenizer) for chunk in chunks]
            for future in tqdm(futures, desc="Tokenizing"):
                all_tokens.extend(future.result())
        
        return all_tokens
    
    train_tokens = process_split(train_texts)
    print(f"🔥 Tokenizing {len(val_texts)} validation documents...")
    val_tokens = process_split(val_texts)
    
    print(f"✅ Training tokens: {len(train_tokens):,}")
    print(f"✅ Validation tokens: {len(val_tokens):,}")
    
    return train_tokens, val_tokens


def create_dataloader(
    tokens: List[int],
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create optimized dataloader"""
    dataset = MedicalDataset(tokens, max_length, stride)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device
) -> torch.Tensor:
    """Calculate loss for a single batch"""
    input_batch = input_batch.to(device, non_blocking=True)
    target_batch = target_batch.to(device, non_blocking=True)
    
    with autocast("cuda", torch.bfloat16):
        logits = model(input_batch)
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            target_batch.flatten()
        )
    
    return loss


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_batches: int = 100
) -> float:
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    num_batches = min(num_batches, len(data_loader))
    
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
    
    model.train()
    return total_loss / num_batches


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    loss: float,
    save_dir: Path,
    config: Dict[str, Any]
):
    """Save model checkpoint"""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config,
    }
    
    save_path = save_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved: {save_path}")
    
    return save_path


def upload_to_huggingface(
    model: nn.Module,
    save_dir: Path,
    repo_id: str,
    config: Dict[str, Any],
    step: int
):
    """Upload model to HuggingFace Hub"""
    try:
        # Save model weights
        torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
        
        # Save config
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Upload to HF
        api = HfApi()
        api.upload_folder(
            folder_path=str(save_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Training checkpoint at step {step}"
        )
        
        print(f"☁️  Uploaded to HuggingFace: {repo_id}")
    except Exception as e:
        print(f"⚠️  Failed to upload to HuggingFace: {e}")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    config: Dict[str, Any],
    save_dir: Path,
    hf_repo_id: str = None
):
    """Main training loop with all optimizations"""
    
    print("=" * 80)
    print("🚀 STARTING MEDICAL LLM PRETRAINING")
    print("=" * 80)
    print(f"📊 Model: {model.count_parameters():,} parameters")
    print(f"📊 Training batches: {len(train_loader):,}")
    print(f"📊 Max steps: {config['max_steps']:,}")
    print(f"📊 Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"📊 Device: {device}")
    print("=" * 80)
    
    model.train()
    global_step = 0
    tokens_seen = 0
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []
    
    grad_accum = config["gradient_accumulation_steps"]
    
    try:
        for epoch in range(100):  # Virtually unlimited epochs
            epoch_loss = 0.0
            epoch_steps = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, (input_batch, target_batch) in enumerate(progress_bar):
                # Forward pass
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accum
                loss.backward()
                
                # Accumulate
                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=config["grad_clip"]
                    )
                    
                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Update counters
                    global_step += 1
                    tokens_seen += input_batch.numel() * grad_accum
                    
                    # Log training loss
                    train_losses.append(loss.item() * grad_accum)
                    epoch_loss += loss.item() * grad_accum
                    epoch_steps += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * grad_accum:.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                        'step': global_step
                    })
                    
                    # Evaluation
                    if global_step % config["eval_freq"] == 0:
                        val_loss = evaluate_model(
                            model, val_loader, device, config["eval_iter"]
                        )
                        val_losses.append(val_loss)
                        
                        # Log to wandb
                        wandb.log({
                            "train_loss": train_losses[-1],
                            "val_loss": val_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "tokens_seen": tokens_seen,
                            "step": global_step,
                        })
                        
                        # Check for improvement
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            print(f"\n✨ New best validation loss: {val_loss:.4f}")
                    
                    # Save checkpoint
                    if global_step % config["save_freq"] == 0:
                        save_checkpoint(
                            model, optimizer, scheduler,
                            global_step, train_losses[-1],
                            save_dir, MODEL_CONFIG
                        )
                        
                        # Upload to HuggingFace
                        if hf_repo_id and config.get("upload_checkpoints", False):
                            if global_step % config.get("upload_frequency", 1000) == 0:
                                upload_to_huggingface(
                                    model, save_dir / "hf_upload",
                                    hf_repo_id, MODEL_CONFIG, global_step
                                )
                    
                    # Check if max steps reached
                    if global_step >= config["max_steps"]:
                        print(f"\n🎉 Reached max steps ({config['max_steps']})")
                        raise StopIteration
            
            # End of epoch summary
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else float('inf')
            print(f"\nEpoch {epoch+1} complete - Avg loss: {avg_epoch_loss:.4f}")
    
    except (KeyboardInterrupt, StopIteration):
        print("\n⚠️  Training stopped")
    
    # Final checkpoint
    print("\n💾 Saving final checkpoint...")
    save_checkpoint(
        model, optimizer, scheduler,
        global_step, train_losses[-1] if train_losses else 0,
        save_dir, MODEL_CONFIG
    )
    
    print(f"\n🎉 Training complete!")
    print(f"📊 Total steps: {global_step:,}")
    print(f"📊 Total tokens: {tokens_seen:,}")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Set random seeds
    torch.manual_seed(TRAINING_CONFIG["seed"])
    np.random.seed(TRAINING_CONFIG["seed"])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Create save directory
    save_dir = Path("./checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    # Initialize tokenizer
    print("🔧 Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("p50k_base")
    
    # Load and prepare data
    train_tokens, val_tokens = prepare_medical_data(DATA_CONFIG, tokenizer)
    
    # Create dataloaders
    print("🔧 Creating dataloaders...")
    train_loader = create_dataloader(
        train_tokens,
        batch_size=TRAINING_CONFIG["batch_size"],
        max_length=TRAINING_CONFIG["max_length"],
        stride=TRAINING_CONFIG["stride"],
        shuffle=True,
        num_workers=TRAINING_CONFIG["num_workers"]
    )
    
    val_loader = create_dataloader(
        val_tokens,
        batch_size=TRAINING_CONFIG["batch_size"],
        max_length=TRAINING_CONFIG["max_length"],
        stride=TRAINING_CONFIG["stride"],
        shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"]
    )
    
    # Initialize model
    print("🔧 Initializing model...")
    model = MedAssistGPT(MODEL_CONFIG)
    model = model.to(device)
    
    # Compile model (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        print("🔧 Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")
    
    print(f"✅ Model has {model.count_parameters():,} parameters")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        betas=(TRAINING_CONFIG["beta1"], TRAINING_CONFIG["beta2"]),
        eps=TRAINING_CONFIG["eps"]
    )
    
    # Initialize scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=TRAINING_CONFIG["learning_rate"],
        total_steps=TRAINING_CONFIG["max_steps"],
        pct_start=TRAINING_CONFIG["warmup_steps"] / TRAINING_CONFIG["max_steps"],
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )
    
    # Initialize wandb
    wandb.init(
        project=WANDB_CONFIG["project"],
        entity=WANDB_CONFIG["entity"],
        name=WANDB_CONFIG["name"],
        config={**MODEL_CONFIG, **TRAINING_CONFIG, **DATA_CONFIG}
    )
    
    # Login to HuggingFace (if uploading)
    if HF_CONFIG.get("upload_checkpoints", False):
        try:
            login()
            create_repo(HF_CONFIG["repo_id"], repo_type="model", exist_ok=True)
            print(f"✅ HuggingFace repo ready: {HF_CONFIG['repo_id']}")
        except Exception as e:
            print(f"⚠️  HuggingFace setup failed: {e}")
            HF_CONFIG["upload_checkpoints"] = False
    
    # Train!
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=TRAINING_CONFIG,
        save_dir=save_dir,
        hf_repo_id=HF_CONFIG["repo_id"] if HF_CONFIG.get("upload_checkpoints") else None
    )
    
    wandb.finish()
    print("\n✅ All done!")


if __name__ == "__main__":
    main()