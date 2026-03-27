"""
RLHF Pipeline: SFT → PPO with a Pre-built Reward Model
=======================================================
Stages
  1. SFT  – supervised fine-tune the base LLM on instruction data
  2. PPO  – align the SFT model using RL, with rewards coming from a
            *pre-built* reward model (no custom RM training required)

Pre-built reward model used
  OpenAssistant/reward-model-deberta-v3-large-v2
  • ~400 MB DeBERTa-v3 encoder fine-tuned on human preference data
  • Input format: "Question: <prompt>\n\nAnswer: <response>"
  • Output: scalar logit (higher = better response)

Install
  pip install -U "transformers>=4.44" "trl>=0.12" datasets accelerate peft bitsandbytes
"""

import torch
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import (
    SFTTrainer, SFTConfig,
    PPOTrainer, PPOConfig,
    AutoModelForCausalLMWithValueHead,   # value head is REQUIRED for PPO
)
from peft import LoraConfig

# ── Global settings ──────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BASE_ID    = "Qwen/Qwen2.5-0.5B-Instruct"   # small model with a chat template
RM_ID      = "OpenAssistant/reward-model-deberta-v3-large-v2"
MAXLEN_SFT = 1024
MAXLEN_RM  = 512

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
if tok.pad_token is None:          # DeBERTa uses eos as pad
    tok.pad_token = tok.eos_token

# =============================================================================
# 1) SFT — Supervised Fine-Tuning (policy initialisation)
# =============================================================================
print("=" * 60)
print("Stage 1 / 2 : SFT")
print("=" * 60)

# Load 6 000 instruction-response pairs
chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:6000]")

def sft_text(example):
    """
    ultrachat_200k schema:
        'messages' : list of {"role": "user"/"assistant", "content": "..."}
        'prompt'   : string (the final user turn, used separately in PPO)

    We feed the full 'messages' list directly into apply_chat_template —
    no manual field extraction needed and works for multi-turn conversations.
    """
    return {"text": tok.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )}

chat = chat.map(sft_text, remove_columns=chat.column_names)

# Load base model with 4-bit quantisation + LoRA adapters
policy_sft = AutoModelForCausalLM.from_pretrained(
    BASE_ID, device_map="auto", load_in_4bit=True
)
peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

sft_cfg = SFTConfig(
    output_dir="ckpt_sft",
    dataset_text_field="text",         # preferred over deprecated formatting_func
    max_seq_length=MAXLEN_SFT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=50,
    report_to=["wandb"],
)

sft_tr = SFTTrainer(
    model=policy_sft,
    tokenizer=tok,
    train_dataset=chat,
    args=sft_cfg,
    peft_config=peft_cfg,
)
sft_tr.train()
sft_tr.model.save_pretrained("ckpt_sft")
tok.save_pretrained("ckpt_sft")
print("SFT checkpoint saved → ckpt_sft")


# =============================================================================
# 2) Load the Pre-built Reward Model (no training needed)
# =============================================================================
print("=" * 60)
print("Loading pre-built reward model :", RM_ID)
print("=" * 60)

# DeBERTa is small enough to run on CPU if no GPU is available
rm_device = DEVICE
rm_tok = AutoTokenizer.from_pretrained(RM_ID)
rm_model = AutoModelForSequenceClassification.from_pretrained(RM_ID).to(rm_device)
rm_model.eval()


@torch.no_grad()
def reward_score(prompt: str, response: str) -> float:
    """
    Score a single prompt-response pair with the pre-built reward model.

    The DeBERTa model expects the concatenation format:
        "Question: <prompt>\n\nAnswer: <response>"
    and returns a single scalar logit (higher is better).
    """
    text = f"Question: {prompt}\n\nAnswer: {response}"
    enc  = rm_tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAXLEN_RM,
    ).to(rm_device)
    return float(rm_model(**enc).logits[0])


# =============================================================================
# 3) PPO — Policy Optimisation
# =============================================================================
print("=" * 60)
print("Stage 2 / 2 : PPO")
print("=" * 60)

# Load the SFT checkpoint with a value head attached (required for PPO)
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "ckpt_sft", device_map="auto", load_in_4bit=True
)

ppo_cfg = PPOConfig(
    model_name=None,
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=8,
    ppo_epochs=2,
    target_kl=0.1,
    # use_score_scaling  : normalise rewards to zero-mean/unit-variance per batch
    # whiten_rewards     : additionally whiten advantages — both reduce reward
    #                      hacking and training instability
    use_score_scaling=True,
    whiten_rewards=True,
    remove_unused_columns=False,
)

ppo_tr = PPOTrainer(config=ppo_cfg, model=ppo_model, tokenizer=tok)

# 400 prompts from the same chat dataset; the model will generate responses
raw_prompts = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:400]")["prompt"]


def to_gen_prompt(user_prompt: str) -> str:
    """Wrap a user message in the model's chat template (no assistant turn yet)."""
    return tok.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


ppo_prompts = [to_gen_prompt(p) for p in raw_prompts]

gen_kwargs = dict(
    max_new_tokens=128,
    do_sample=True,
    top_p=0.95,
    temperature=0.8,
    pad_token_id=tok.eos_token_id,
)

# ── PPO training loop ─────────────────────────────────────────────────────────
for step, i in enumerate(range(0, len(ppo_prompts), ppo_cfg.batch_size)):
    """
    Each iteration of the loop:
      1. Tokenise a batch of prompts.
      2. Generate a response for every prompt.
      3. Decode the response-only tokens (strip the input prefix).
      4. Score each (prompt, response) pair with the reward model.
      5. Apply a tiny length penalty to discourage padding/verbosity.
      6. Convert everything to lists of 1-D tensors — required by PPOTrainer.
      7. Call ppo_tr.step() to compute PPO loss and update the policy.
    """
    batch_prompts = ppo_prompts[i : i + ppo_cfg.batch_size]
    if not batch_prompts:
        break

    # 1. Tokenise queries
    q_enc     = tok(batch_prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512)
    q_tensors = q_enc.input_ids.to(DEVICE)           # shape: [B, L_q]

    # 2. Generate full sequences (prompt + response)
    full      = ppo_tr.generate(q_tensors, **gen_kwargs)   # [B, L_q + L_r]
    resp_only = full[:, q_tensors.shape[1]:]               # [B, L_r]

    # 3. Decode responses
    responses = tok.batch_decode(resp_only, skip_special_tokens=True)

    # 4+5. Score with the pre-built reward model + tiny length penalty
    rewards = []
    for j, resp in enumerate(responses):
        score   = reward_score(raw_prompts[i + j], resp)
        n_toks  = resp_only[j].shape[0]
        # penalise very long outputs (discourage padding inflation)
        score  -= 0.002 * n_toks
        rewards.append(score)

    # 6. Convert to lists of 1-D tensors — PPOTrainer requires this format
    query_tensors    = [q_tensors[j]  for j in range(len(batch_prompts))]
    response_tensors = [resp_only[j]  for j in range(len(batch_prompts))]
    reward_tensors   = [torch.tensor(r, dtype=torch.float) for r in rewards]

    # 7. PPO update step
    stats = ppo_tr.step(query_tensors, response_tensors, reward_tensors)

    mean_r = np.mean(rewards)
    kl     = stats.get("ppo/mean_non_score_reward", "n/a")   # KL proxy in TRL
    print(f"PPO step {step:03d} | mean_reward={mean_r:.3f} | KL_proxy={kl}")

# ── Save final aligned policy ─────────────────────────────────────────────────
ppo_tr.model.save_pretrained("ckpt_ppo")
tok.save_pretrained("ckpt_ppo")
print("Saved aligned policy → ckpt_ppo")
