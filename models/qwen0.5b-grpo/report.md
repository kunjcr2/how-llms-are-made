# Qwen2.5-0.5B GSM8K Post-Training Experiments

## Overview

Comparison of **SFT, weighted SFT, rejection-sampling fine-tuning (RFT), GRPO, and Dr. GRPO** on `Qwen/Qwen2.5-0.5B-Instruct`.

**Evaluation:** full GSM8K test split (`1,319` problems), greedy decoding, exact numeric-answer accuracy.

## Methods

### Standard SFT

LoRA fine-tuning on GSM8K gold solutions.

* LoRA rank: `16`
* LR: `2e-4`
* Epochs: `2`
* Completion-only cross-entropy
* `####` format rate: ~`99.85%`

### Weighted SFT

Modified token-level objective:

* Normal CE weight: `1×`
* Final-answer/correctness weight: `5×`

### RFT

Rejection-sampling fine-tuning:

1. Sample `8` solutions per training problem.
2. Keep correct, unique trajectories.
3. LoRA-SFT on the retained generations.

### GRPO

Online RL using fresh sampled completions and binary answer-correctness reward.

* `num_generations = 8`
* `beta = 0`
* Group-relative advantage optimization

### Dr. GRPO

GRPO variant designed to reduce normalization-induced bias.

* `loss_type = "dr_grpo"`
* `scale_rewards = False`
* No per-group reward standard-deviation normalization
* Constant completion-length normalization

## Results

| Method                                  |         GSM8K Accuracy |
| --------------------------------------- | ---------------------: |
| Base                                    |  **42.08%** (555/1319) |
| Standard Gold SFT                       |  **32.30%** (426/1319) |
| Weighted SFT (`1× CE + 5× correctness`) |             **<24.x%** |
| Full-model SFT                          |               **<30%** |
| RFT                                     |             **~37.4%** |
| Base → GRPO                             |  **45.03%** (594/1319) |
| RFT → GRPO @ 300                        |  **36.54%** (482/1319) |
| RFT → GRPO @ 500                        |  **34.95%** (461/1319) |
| Base → Dr. GRPO @ 200                   |  **46.02%** (607/1319) |
| Base → Dr. GRPO @ 250                   | **44.66%** (~589/1319) |

## Findings

* **Dr. GRPO @ 200 is the best observed checkpoint:** `42.08% → 46.02%` (`+3.94 pp`).
* Standard GRPO also improved the base model to `45.03%`.
* SFT, weighted SFT, and RFT all reduced held-out accuracy.
* RFT → GRPO degraded progressively: `~37.4% → 36.54% → 34.95%`.
* Dr. GRPO peaked early: `46.02% @ 200 → 44.66% @ 250`, showing that training reward and held-out greedy accuracy can diverge.
* Standard SFT achieved near-perfect formatting while substantially reducing mathematical accuracy.

## Conclusion

For Qwen2.5-0.5B, **RL directly from the base model outperformed supervised post-training**, with Dr. GRPO producing the best observed result at **46.02%**. Intermediate checkpoint evaluation was critical, since both RFT→GRPO and later Dr. GRPO training showed that improving training objectives did not guarantee better held-out greedy accuracy.
