# Qwen2.5-0.5B GSM8K Experiments

## Overview

Comparison of **SFT, weighted SFT, rejection-sampling fine-tuning (RFT), and GRPO** on `Qwen/Qwen2.5-0.5B-Instruct` using GSM8K.

**Evaluation:** 1,319 GSM8K test problems, greedy decoding, exact numeric-answer accuracy.

## Methods

### Standard SFT

LoRA fine-tuning on GSM8K gold solutions using standard completion-token cross-entropy.

- LoRA rank: `16`
- LR: `2e-4`
- Epochs: `2`
- Completion-only loss
- `####` format rate: ~`99.85%`

### Weighted SFT

Modified SFT objective emphasizing answer correctness:

- Standard CE weight: `1×`
- Correctness/final-answer weight: `5×`

Despite stronger emphasis on the final answer, performance degraded further.

### RFT

For each training problem:

1. Sample `8` solutions from the base model.
2. Retain correct, unique trajectories.
3. LoRA-SFT on the filtered trajectories.

### GRPO

Online RL with fresh model generations and binary final-answer reward.

- Generations/prompt: `8`
- Temperature: `0.8`
- Top-p: `0.95`
- `beta = 0`

## Results

| Method                                  |        GSM8K Accuracy |
| --------------------------------------- | --------------------: |
| Base                                    | **42.08%** (555/1319) |
| Standard Gold SFT                       | **32.30%** (426/1319) |
| Weighted SFT (`1× CE + 5× correctness`) |            **<24.x%** |
| Full-model SFT                          |              **<30%** |
| RFT                                     |            **~37.4%** |
| Base → GRPO                             | **45.03%** (594/1319) |
| RFT → GRPO @ 300                        | **36.54%** (482/1319) |
| RFT → GRPO @ 500                        | **34.95%** (461/1319) |

## Findings

- **Base → GRPO performed best:** `42.08% → 45.03%` (`+2.95 pp`).
- Standard SFT learned formatting almost perfectly but reduced accuracy by ~`10 pp`.
- Increasing answer/correctness weighting to `5×` made SFT substantially worse (`<24.x%`).
- RFT also degraded the base model: `42.08% → ~37.4%`.
- GRPO from the RFT initialization degraded further: `~37.4% → 36.54% → 34.95%`.
- RFT→GRPO training reward increased while greedy test accuracy decreased, showing a mismatch between **on-policy reward optimization and held-out greedy generalization**.

## Conclusion

For this 0.5B model, **direct GRPO was the only training strategy that improved GSM8K accuracy**. Both standard and answer-weighted SFT degraded reasoning performance, while RFT initialization made subsequent GRPO optimization progressively worse.
