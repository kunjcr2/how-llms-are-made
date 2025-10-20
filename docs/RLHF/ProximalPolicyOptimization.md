# 📘 PPO in RLHF — Beginner Friendly Guide

Reinforcement Learning with Human Feedback (**RLHF**) often uses **Proximal Policy Optimization (PPO)** to fine-tune large language models.
This guide explains what PPO does in RLHF, why we use it, and how to implement it with **PyTorch** and **Hugging Face TRL**.

---

## 🔑 Core Idea of PPO in RLHF

1. **Prompt pool** → pick a batch of prompts.
2. **Policy model** → generate answers (sampled).
3. **Reward model** → score each answer.
4. **PPO update** → adjust the policy so:

   - High-reward answers become **more likely**.
   - Low-reward answers become **less likely**.
   - Changes are **clipped**, so the new model doesn’t drift too far from the old one.

👉 PPO replaces the “plain policy gradient step” in RLHF with a **safe update**.

---

## ⚖️ PPO Objective in Plain English

- **Vanilla PG update**:

  $$
  \nabla J(\theta) \approx R \cdot \nabla \log \pi_\theta(y|x)
  $$

  _Push up/down based only on reward._

- **PPO update**:

  $$
  L^{\text{PPO}}(\theta) = \min\Big( r(\theta) \cdot A,\; \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \cdot A \Big)
  $$

Where:

- $r(\theta)$ = new prob / old prob.
- $A$ = advantage (reward - baseline).
- $\epsilon$ = small number (e.g. 0.2).

👉 PPO says: _“Improve the policy, but don’t let it jump too far.”_

---

## 🧩 PPO in RLHF — Workflow Diagram

```
[Prompts] → [Policy Model] → [Generated Answers]
                          ↓
                [Reward Model scores]
                          ↓
          [PPO objective: reward + clip]
                          ↓
                 [Optimizer updates policy]
```

---

## 📝 PyTorch-Style Pseudocode

```python
for batch in prompt_loader:
    # 1. Sample answers from policy
    answers, logprobs_old = policy.sample(batch)

    # 2. Reward model scores
    rewards = reward_model.score(batch, answers)

    # 3. Compute advantage (reward - baseline)
    advantages = rewards - rewards.mean()

    # 4. Policy ratio: new / old
    logprobs_new = policy.logprobs(batch, answers)
    ratio = (logprobs_new - logprobs_old).exp()

    # 5. PPO objective
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1-eps, 1+eps) * advantages
    loss = -torch.min(unclipped, clipped).mean()

    # 6. Backprop + update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 🚀 PPO with Hugging Face TRL

Hugging Face’s [**trl**](https://huggingface.co/docs/trl/index) library makes this loop easier.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig

# 1. Load models
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. PPO config
config = PPOConfig(
    model_name="gpt2",
    learning_rate=1e-5,
    batch_size=16,
    ppo_epochs=4
)

ppo_trainer = PPOTrainer(config, model, tokenizer)

# 3. RLHF loop
for batch in prompts:
    # generate
    responses = ppo_trainer.generate(batch, max_new_tokens=50)

    # reward model (toy: length as reward)
    rewards = [len(r) for r in responses]

    # PPO step
    stats = ppo_trainer.step(batch, responses, rewards)
```

👉 In practice, replace the toy reward with a **trained reward model** or preference data.

---

## 📚 Resources

- [Hugging Face TRL docs](https://huggingface.co/docs/trl/index)
- [OpenAI PPO paper (2017)](https://arxiv.org/abs/1707.06347)
- [Illustrated PPO blog](https://huggingface.co/blog/ppo)
- [RLHF with TRL course](https://huggingface.co/learn/rl-course/unit2/ppo)

---

## 🎯 Key Takeaways

- PPO = policy gradient with a **clipping rule** for safe updates.
- In RLHF, PPO ensures the model **aligns with preferences** without drifting too far from its base.
- Hugging Face TRL gives you **ready-to-use PPOTrainer** for LLM fine-tuning.
