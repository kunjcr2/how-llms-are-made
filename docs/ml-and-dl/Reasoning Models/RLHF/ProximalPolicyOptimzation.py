# pip install -U "transformers>=4.44" "trl>=0.12" datasets accelerate peft bitsandbytes

import torch, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import (
    SFTTrainer, SFTConfig,
    RewardTrainer, RewardConfig,
    PPOTrainer, PPOConfig,
    AutoModelForCausalLMWithValueHead
)
from peft import LoraConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # has chat template
MAXLEN_SFT, MAXLEN_RM = 1024, 512

tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

# ---------- 1) SFT (policy init) ----------
chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:6000]")
def sft_text(p, a):
    msgs = [{"role":"user","content":p},{"role":"assistant","content":a}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
chat = chat.map(lambda ex: {"text": sft_text(ex["prompt"], ex["response"])},
                remove_columns=chat.column_names)

policy_sft = AutoModelForCausalLMWithValueHead.from_pretrained(  # <-- VALUE HEAD
    BASE_ID, device_map="auto", load_in_4bit=True
)
peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
sft_cfg = SFTConfig(output_dir="ckpt_sft_qwen05b_vhead",
                    max_seq_length=MAXLEN_SFT,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=16,
                    num_train_epochs=1,
                    learning_rate=5e-5,
                    logging_steps=50, report_to=[])
sft_tr = SFTTrainer(model=policy_sft, tokenizer=tok, train_dataset=chat,
                    args=sft_cfg, peft_config=peft_cfg, formatting_func=lambda b: b["text"])
sft_tr.train()
sft_tr.model.save_pretrained("ckpt_sft_qwen05b_vhead"); tok.save_pretrained("ckpt_sft_qwen05b_vhead")

# ---------- 2) Reward Model (pairwise) ----------
rm_base = "microsoft/deberta-v3-small"
rm_tok  = AutoTokenizer.from_pretrained(rm_base, use_fast=True)
rm_model = AutoModelForSequenceClassification.from_pretrained(
    rm_base, num_labels=1, problem_type="regression"
).to(DEVICE)

prefs_tr = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:5000]")
prefs_ev = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs[:1000]")

def pack_rm(prompt, answer):
    msgs = [{"role":"user","content":prompt},{"role":"assistant","content":answer}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

def prep(ex):
    ex["chosen_text"]   = pack_rm(ex["prompt"], ex["chosen"])
    ex["rejected_text"] = pack_rm(ex["prompt"], ex["rejected"])
    return ex
prefs_tr = prefs_tr.map(prep); prefs_ev = prefs_ev.map(prep)

def rm_tok_fn(b):
    ch = rm_tok(b["chosen_text"],   truncation=True, padding=True, max_length=MAXLEN_RM)
    rj = rm_tok(b["rejected_text"], truncation=True, padding=True, max_length=MAXLEN_RM)
    return {"chosen_input_ids":ch["input_ids"], "chosen_attention_mask":ch["attention_mask"],
            "rejected_input_ids":rj["input_ids"], "rejected_attention_mask":rj["attention_mask"]}
prefs_tr = prefs_tr.map(rm_tok_fn, batched=True, remove_columns=prefs_tr.column_names)
prefs_ev  = prefs_ev.map(rm_tok_fn, batched=True, remove_columns=prefs_ev.column_names)

rm_cfg = RewardConfig(output_dir="ckpt_rm_debv3",
                      per_device_train_batch_size=8, gradient_accumulation_steps=2,
                      per_device_eval_batch_size=8, num_train_epochs=1, learning_rate=2e-5,
                      warmup_ratio=0.05, fp16=torch.cuda.is_available(), logging_steps=50,
                      eval_strategy="steps", eval_steps=200, save_steps=400, report_to=[])
rm_tr = RewardTrainer(model=rm_model, args=rm_cfg, train_dataset=prefs_tr,
                      eval_dataset=prefs_ev, tokenizer=rm_tok)
rm_tr.train(); rm_tr.save_model("ckpt_rm_debv3")

import numpy as np, torch.nn.functional as F
@torch.no_grad()
def rm_score_strs(texts):
    enc = rm_tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=MAXLEN_RM).to(DEVICE)
    return rm_model(**enc).logits.squeeze(-1).cpu().numpy()

# ---------- 3) PPO (policy has value head) ----------
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(  # <-- VALUE HEAD
    "ckpt_sft_qwen05b_vhead", device_map="auto", load_in_4bit=True
)
ppo_cfg = PPOConfig(model_name=None, learning_rate=1e-5,
                    batch_size=16, mini_batch_size=8, ppo_epochs=2,
                    target_kl=0.1, remove_unused_columns=False)
ppo_tr = PPOTrainer(config=ppo_cfg, model=ppo_model, tokenizer=tok)

raw_prompts = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:400]")["prompt"]
def to_gen_prompt(user_p):
    return tok.apply_chat_template([{"role":"user","content":user_p}],
                                   tokenize=False, add_generation_prompt=True)
ppo_prompts = [to_gen_prompt(p) for p in raw_prompts]

gen_kwargs = dict(max_new_tokens=128, do_sample=True, top_p=0.95, temperature=0.8, pad_token_id=tok.eos_token_id)

for i in range(0, len(ppo_prompts), ppo_cfg.batch_size):
    batch_prompts = ppo_prompts[i:i+ppo_cfg.batch_size]
    if not batch_prompts: break
    q = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(DEVICE)
    resp = ppo_tr.generate(q, **gen_kwargs)
    gen_only = resp[:, q.shape[1]:]
    responses = tok.batch_decode(gen_only, skip_special_tokens=True)

    # reward from RM on full chat
    rm_inputs = [pack_rm(raw_prompts[i+j], responses[j]) for j in range(len(responses))]
    rewards = rm_score_strs(rm_inputs).tolist()
    # tiny length penalty (optional)
    lengths = [len(tok(r).input_ids) for r in responses]
    rewards = [float(r - 0.002 * L) for r, L in zip(rewards, lengths)]

    stats = ppo_tr.step(list(q), list(resp), rewards)
    print(f"PPO step {i//ppo_cfg.batch_size:03d} | mean_reward={np.mean(rewards):.3f} | KL={stats.get('ppo/kl')}")

ppo_tr.model.save_pretrained("ckpt_ppo_qwen05b_vhead"); tok.save_pretrained("ckpt_ppo_qwen05b_vhead")
print("Saved PPO policy → ckpt_ppo_qwen05b_vhead")
