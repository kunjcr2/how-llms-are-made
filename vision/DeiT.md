# DeiT (Data-efficient Image Transformers)

## Problem and Motivation
The original Vision Transformer (ViT) by Dosovitskiy et al. (2020) demonstrated that pure transformers could replace CNNs for vision. However, it had a major catch: **it lacked the inductive biases** of CNNs (translation invariance, locality).

To compensate, ViT required **massive datasets** (like JFT-300M) for pre-training. When trained only on ImageNet-1K (1.3M images) from scratch, it underperformed ResNet counterparts significantly. DeiT (Data-efficient Image Transformers) was introduced to solve this: enabling high-performance ViT training on standard hardware with standard datasets.

## Core Ideas and Intuition
DeiT introduces two main pillars to make ViT efficient:
1.  **Strong Regularization & Augmentation**: Replacing the missing inductive biases with aggressive data augmentation and training heuristics.
2.  **Hard-Label Distillation**: Using a "teacher" model (e.g., a ConvNet like RegNet) to guide the student Transformer.

A key innovation is the **Distillation Token**. Unlike standard distillation where the student mimics the teacher's output distribution on the class token, DeiT adds a special token specifically responsible for interacting with the teacher.

## Formalism / Objective
Let $Z_s$ be the logits of the student model and $Z_t$ be the logits of the teacher model.
Let $y$ be the true ground-truth label.
Let $\psi$ be the Softmax function.
Let $\text{CE}$ be Cross-Entropy and $\text{KL}$ be Kullback-Leibler divergence.

Standard Soft Distillation minimizes:
$$
\mathcal{L} = (1 - \lambda) \text{CE}(\psi(Z_s), y) + \lambda \tau^2 \text{KL}(\psi(Z_s / \tau), \psi(Z_t / \tau))
$$

DeiT often prefers **Hard Distillation**, where the student simply predicts the teacher's *hard* decision (i.e., the argmax class).
$$
\mathcal{L}_{\text{hardDistill}} = \frac{1}{2} \text{CE}(\psi(Z_s), y) + \frac{1}{2} \text{CE}(\psi(Z_{\text{distill}}), y_{\text{teacher}})
$$
where $y_{\text{teacher}} = \text{argmax}(Z_t)$.

The student has two heads: one for the Classification Token (`[CLS]`) trained on ground truth, and one for the Distillation Token (`[DIST]`) trained on the teacher's prediction. The final inference prediction is often the average of both heads.

## Architecture / Design
The architecture is identical to a standard ViT, except for the input sequence:
1.  **Patches**: Image is split into fixed-size patches (e.g., $16 \times 16$).
2.  **Tokens**: Flattened patches + Position Embeddings.
3.  **Special Tokens**:
    *   `[CLS]`: Standard class token.
    *   `[DIST]`: **New distillation token**.
4.  **Interaction**: Self-attention layers allow `[CLS]` and `[DIST]` to interact with image patches and each other.

Crucially, the `[DIST]` token learns to mimic what the CNN teacher "sees", effectively injecting CNN-like inductive biases (locality) into the Transformer through the loss function.

## Training Procedure
The "DeiT recipe" is as famous as the architecture itself because ViTs are prone to overfitting on small data.
*   **Augmentations**:
    *   RandAugment suitable for Transformers.
    *   Mixup and CutMix (essential).
    *   Random Erasing.
*   **Regularization**:
    *   **Stochastic Depth**: Randomly dropping layers during training (row pruning in the residual branches).
    *   **Weight Decay**: carefully tuned (e.g., 0.05).
*   **Optimizer**: AdamW. SGD generally works poorly for ViTs compared to CNNs.

## Practical Considerations and Pitfalls
*   **Teacher Choice**: A ConvNet teacher typically works better than a Transformer teacher for a Transformer student, likely because the ConvNet teaches the "missing" inductive biases.
*   **Hyperparameters**: The recipe is brittle. Dropping Mixup or Stochastic Depth usually causes performance to collapse by huge margins (5-10% accuracy).
*   **Resolution**: Fine-tuning at higher resolution (e.g., train at 224, test at 384) requires interpolating positional embeddings.

## References
*   **(Touvron et al., 2021) Training data-efficient image transformers & distillation through attention**: The core DeiT paper.
