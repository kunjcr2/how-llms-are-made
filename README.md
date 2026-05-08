# LLMs from Scratch

A comprehensive repository covering Large Language Models from fundamentals to advanced architectures. Includes theory, hands-on notebooks, from-scratch implementations, fine-tuning pipelines, and trained artifacts. Built entirely in 2025.

---

## Highlight: MedAssistGPT

A 401M parameter medical domain LLM, pretrained from scratch on 2 million PubMed abstracts.

**Architecture:**
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA) with 4 KV groups
- SwiGLU activation in feed-forward layers
- RMSNorm for layer normalization
- 24 transformer blocks, 1024 hidden dimension, 16 attention heads

**Training Features:**
- Flash Attention optimization for A100 GPU
- Memory-mapped datasets for zero RAM overhead
- Parallel data processing with multiprocessing
- Gradient accumulation (effective batch size: 128)
- Automatic checkpointing and HuggingFace uploads
- Weights and Biases integration

**Links:** [HuggingFace Model](https://huggingface.co/kunjcr2/MedAssistGPT)

See `models/MedAssistGPT/` for full implementation.

---

## Repository Structure

```
llms-from-scratch/
├── llms/               # Tutorials and implementations
├── models/             # Trained model artifacts
├── docs/               # Notes, theory, and deep dives
├── vision/             # Computer vision models
├── basics/             # ML math and PyTorch fundamentals
├── papers.md           # Curated paper list with links
└── ToRead.md           # Active reading list
```

---

### llms/

#### DeepSeek (`llms/deepseek/`)

Complete implementation and documentation of DeepSeek architecture:

**Notes (`llms/deepseek/notes/`):**

| File | Topic |
|------|-------|
| `MultiHeadLatentAttention.md` | KV cache compression via latent matrices, absorbed queries |
| `MixtureOfExperts.md` | Sparse expert routing, auxiliary loss, load balancing |
| `RotaryPositionalEncoding.md` | RoPE derivation, complex number formulation, MLA integration |
| `SinusoidalPositionalEncoding.md` | Integer, binary, sinusoidal positional encoding theory |
| `KVCache.md` | KV cache mechanics, memory layout, eviction strategies |
| `CompressedSparseAttention.md` | Sparse attention patterns and compression |
| `MTP_technical.md` | Multi-Token Prediction technical implementation |
| `MTP_theoritical.md` | Multi-Token Prediction theory |
| `ManifoldConstraintHyperConnection.md` | Manifold constraints and hyper-connections |

**Code (`llms/deepseek/codes/`):** Expert routing, MoE base, noisy top-k, RMSNorm, RoPE, complete DeepSeek implementation notebook.

---

#### GPT (`llms/gpt/`)

End-to-end tutorials for building GPT from scratch:

| Step | Folder | Content |
|------|--------|---------|
| 1 | `1_tokenizer/` | tiktoken implementation |
| 2 | `2_attention/` | Self-attention and multi-head attention |
| 3 | `3_architecture/` | Full GPT model construction |
| 4 | `4_training/` | Training loops with gradient checkpointing |
| 5 | `5_post_training/` | Techniques after pretraining |
| 6 | `6_finetune/` | LoRA and full fine-tuning |

---

#### Mamba (`llms/mamba/`)

State Space Models as transformer alternatives:

| File | Topic |
|------|-------|
| `StateSpaceModels.md` | Linear RNNs with discretization, A/B/C/Delta matrices, GPU-parallelizable convolutions |
| `SelectiveStateSpaceModels.md` | Input-dependent parameters, parallel associative scan, SRAM optimization |
| `Mamba.md` | Full Mamba architecture deep dive |
| `mamba.py` | From-scratch implementation |
| `mamba.ipynb` | Interactive notebook |

---

#### Mixture of Depths (`llms/mod/`)

Google DeepMind's dynamic compute allocation:

| File | Topic |
|------|-------|
| `what_is_mod.md` | Top-k routing, per-token layer skipping, static computation graphs |
| `how_mod_work.md` | Per-block routing, residual paths, MoDE (combined with MoE) |

Up to 50% FLOPs reduction while maintaining performance.

---

### models/

| Model | Description |
|-------|-------------|
| **MedAssistGPT** | 401M medical LLM pretrained on PubMed — RoPE, GQA, SwiGLU |
| **GatorGPT** | Modern transformer with GQA, RoPE, SwiGLU, vLLM ready |
| **gpt155m** | GPT from scratch, 155M parameters |
| **gpt211m** | GPT from scratch, 211M parameters |
| **qwen2.5-0.5b-sft-dpo** | Qwen 2.5 fine-tuned with SFT then DPO |
| **flan-t5-finetuned** | Fine-tuned Flan-T5 |
| **gemma2-9b** | Gemma 2 9B experiments |
| **llama3-3b-lora-openhermes** | LoRA adapters for LLaMA 3-3B on OpenHermes |
| **llm-firewall** | LLM safety classifier / adversarial defense project |
| **AdaptRoute** | Adaptive routing experiments |
| **GAN** | Generative Adversarial Network implementation |
| **SVD** | SVD-based model compression |
| **SimpleML** | Foundational ML models |
| **neural-sort** | Neural network sorting experiments |

---

### docs/

#### Reinforcement Learning (`docs/RL/`)

A complete 20-lecture RL series from fundamentals to LLM alignment:

| Lecture | Topic |
|---------|-------|
| 01 | RL Fundamentals |
| 02 | Markov Decision Process |
| 03 | Epsilon-Greedy Exploration |
| 04 | Value Functions |
| 05 | Dynamic Programming |
| 06 | Monte Carlo Methods |
| 07 | Temporal Difference Learning |
| 08 | TD Control (SARSA, Q-Learning) |
| 09 | Value Function Approximation |
| 10 | Policy Gradient Theorem |
| 11 | Deep Q-Network (DQN) |
| 12 | REINFORCE |
| 13 | Advantage Function |
| 14 | KL Divergence |
| 15 | TRPO |
| 16 | PPO |
| 17 | RL for LLMs |
| 18 | Reward Model |
| 19 | DPO |
| 20 | GRPO |

Also includes `rl_mindmap.html` — interactive visual map of the full RL landscape.

---

#### Reasoning Models (`docs/Reasoning Models/`)

Notes across 4 lectures on LLM reasoning capabilities and implementation, with code notebook.

---

#### RAG (`docs/rag/`)

Retrieval-Augmented Generation — 2 lecture notes covering basics to implementation.

---

#### ML and DL Fundamentals (`docs/ml-and-dl/`)

| File | Topic |
|------|-------|
| `FrontierLLMTraining.md` | Full-stack frontier LLM training: memory, parallelism, MoE, precision, fault tolerance |
| `FlashAttention.md` | SRAM tiling, online softmax, memory complexity |
| `SlidingWindowAttention.md` | Local attention patterns |
| `AliBi.md` | Attention with Linear Biases |
| `GatedLinearUnit.md` | SwiGLU, GeGLU activations |
| `EncoderOnly.md` | BERT-style encoder-only architectures |
| `FourierTransform.md` | Fourier Transform theory for signal processing |
| `BackPropogation.md` | Backpropagation theory |
| `BackProp.ipynb` | Interactive backpropagation notebook |
| `LightningModule.md` | Training with PyTorch Lightning |
| `SupervisedFinetuning-DataPreprocess.md` | SFT data preprocessing pipelines |
| `Prompt.md` | Prompt engineering techniques |
| `AK_GPT.md` | GPT from scratch notes (Karpathy-style) |
| `AK_BetterModel.md` | Improving GPT — modern techniques |

---

#### Evaluation (`docs/eval/`)

| File | Topic |
|------|-------|
| `bleu/` | BLEU score implementation and documentation |
| `MLEval/MLEvalCla.md` | ML classification evaluation metrics |
| `MLEval/MLEvalReg.md` | ML regression evaluation metrics |
| `MLEval/MLEvalCla.py` | Classification evaluation implementation |
| `MLEval/MLEvalReg.py` | Regression evaluation implementation |

---

#### Optimization (`docs/optimization/`)

- Introduction to Optimization
- SGD Optimizer
- Momentum Gradient Descent
- RMSProp
- Adam Optimizer

---

#### Real Problems (`docs/RealProblems.md`)

Notes on real-world production challenges in deploying LLMs.

---

### vision/

#### Notes (`vision/notes/`)

| File | Topic |
|------|-------|
| `CNN.md` | Convolutional Neural Networks |
| `DeiT.md` | Data-efficient Image Transformers |
| `SwinTransformers.md` | Swin Transformer architecture |
| `DetectionTransformer.md` | DETR — object detection with transformers |
| `ContrastiveLearning.md` | Contrastive learning theory |
| `CLIP.md` | CLIP — vision-language contrastive pretraining |
| `VisionLanguageModels.md` | Vision-language model architectures |
| `Flamingo.md` | Flamingo multimodal LLM |
| `LLaVA.md` | LLaVA visual instruction tuning |
| `SegmentAnythingModel.md` | SAM architecture and zero-shot segmentation |
| `TimeSformer.md` | Transformer for video understanding |
| `DDPM.md` | Denoising Diffusion Probabilistic Models |

#### Code (`vision/code/`)

| File | Description |
|------|-------------|
| `SwinTransformers.py` | Swin Transformer implementation |
| `DetectionTransformer.py` | DETR implementation |
| `TinyViT.py` | Compact ViT implementation |
| `ContrastiveLearning.py` | Contrastive learning implementation |
| `CLIP.py` | CLIP implementation |
| `VisionLanguageModels.py` | VLM implementation |
| `Flamingo.py` | Flamingo implementation |
| `SegmentAnythingModel.py` | SAM implementation |
| `TimeSformer.py` | TimeSformer implementation |
| `DDPM.py` | Diffusion model implementation |
| `ViT_demo.ipynb` | Vision Transformer demo notebook |

---

### basics/

Foundational material organized into three areas:

| Folder | Content |
|--------|---------|
| `architectures/` | Core neural network architectures |
| `ml_math/` | Mathematical foundations |
| `quicky_pytorch/` | Quick-reference PyTorch patterns |

---

## Papers and Reading

- **`papers.md`** — Curated list of all key papers with links, grouped by topic (Architecture, MoE, Efficiency, Alignment, Safety, Fine-tuning)
- **`ToRead.md`** — Active weekly reading list tracking paper progress

---

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- transformers
- tiktoken
- vLLM (for deployment)

### Clone
```bash
git clone https://github.com/kunjcr2/llms-from-scratch.git
cd llms-from-scratch
```

### Recommended Learning Path

1. **Start with GPT:** `llms/gpt/` — Build a transformer from scratch
2. **Explore DeepSeek:** `llms/deepseek/` — Modern architecture innovations
3. **Understand Mamba:** `llms/mamba/` — Alternative to attention
4. **Study MoD:** `llms/mod/` — Dynamic compute allocation
5. **Learn RL and alignment:** `docs/RL/` — 20-lecture series from RL basics to GRPO
6. **See trained models:** `models/` — Working implementations
7. **Dive into vision:** `vision/` — ViT, Swin, CLIP, SAM, diffusion

---

## Notebooks Index

| Topic | Path |
|-------|------|
| Tokenizer | `llms/gpt/1_tokenizer/LLM_tokenizer.ipynb` |
| Attention | `llms/gpt/2_attention/LLM_attention.ipynb` |
| Architecture | `llms/gpt/3_architecture/LLM_GPT_arch.ipynb` |
| Training | `llms/gpt/4_training/LLM_training.ipynb` |
| Post-training | `llms/gpt/5_post_training/LLM_post_training.ipynb` |
| LoRA Fine-tuning | `llms/gpt/6_finetune/LLM_LoRA_finetune.ipynb` |
| Full Fine-tuning | `llms/gpt/6_finetune/LLM_full_finetune.ipynb` |
| DeepSeek Complete | `llms/deepseek/codes/deepseek_complete.ipynb` |
| Mamba | `llms/mamba/mamba.ipynb` |
| Backpropagation | `docs/ml-and-dl/BackProp.ipynb` |
| ViT Demo | `vision/code/ViT_demo.ipynb` |
| Reasoning Models | `docs/Reasoning Models/Lec3_code.ipynb` |
| MedAssistGPT | `models/MedAssistGPT/MedAssistGPT.ipynb` |
| GPT-155M | `models/gpt155m/LLM_155M.ipynb` |
| GPT-211M | `models/gpt211m/LLM_211M.ipynb` |
| Qwen 2.5 DPO | `models/qwen2.5-0.5b-sft-dpo/` |
| LLM Firewall | `models/llm-firewall/` |

---

## License

None

---

Created and maintained by Kunj Shah
