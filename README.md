# How LLMs Are Made

This repository is a comprehensive guide and implementation walkthrough of Large Language Models (LLMs), from fundamentals to advanced architectures. It includes theory, hands-on notebooks, implementations, and trained artifacts.

## 📁 New Repository Layout

- **tutorials/**: Step-by-step, topic-focused walkthroughs
  - **tutorials/gpt/**: Tokenizer, attention, architecture, training, post-training, fine-tuning
  - **tutorials/deepseek/**: MoE, RoPE, latent attention, MTP; with code and assets
- **docs/**: Conceptual docs and notes
  - **docs/ml-and-dl/**: Core ML/DL concepts, attention variants, backprop, GLU, vLLM, RAG
  - **docs/mod/**: Mixture of Dendrites (Kimi) notes
- **models/**: Your trained models and implementation projects
  - Includes `GatorGPT`, LoRA adapters, checkpoints, config, inference helpers
- **research/**
  - **research/papers/**: PDFs and references used in the repo
- **deployments/**: Deployment-related artifacts (e.g., Dockerfiles, guides)

Note: Original folders are preserved inside the new sections to honor the no-delete rule.

## 🧭 Quick Start Navigation

- Learn GPT end-to-end: `tutorials/gpt/`
- Explore DeepSeek internals: `tutorials/deepseek/`
- Read core theory: `docs/ml-and-dl/`
- MoD (Kimi) notes: `docs/mod/`
- See working models: `models/` (start with `models/GatorGPT/README.md`)
- Skim references: `research/papers/`
- Deployment bits: `deployments/`

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- transformers
- tiktoken (for tokenization)
- vLLM (for deployment)

### Clone

```bash
git clone https://github.com/kunjcr2/how-llms-are-made.git
cd how-llms-are-made
```

## 📓 Notebooks Index

- Tokenizer: `tutorials/gpt/1_tokenizer/LLM_tokenizer.ipynb`
- Attention: `tutorials/gpt/2_attention/LLM_attention.ipynb`
- Architecture: `tutorials/gpt/3_architecture/LLM_GPT_arch.ipynb`
- Training: `tutorials/gpt/4_training/LLM_training.ipynb`
- Post-training: `tutorials/gpt/5_post_training/LLM_post_training.ipynb`
- Fine-tuning: `tutorials/gpt/6_finetune/`

DeepSeek examples and writeups: `tutorials/deepseek/`

## 🚀 Highlight: GatorGPT

- Lightweight transformer
- Modern choices: Grouped Query Attention, RoPE, SwiGLU
- Training and data pipeline included
- Deployment with vLLM; see `models/GatorGPT/README.md`

## 📖 Documentation

See `docs/` for theory, diagrams, and implementation notes across topics.

## 🤝 Contributing

Contributions are welcome! Please open an issue or PR.

## 📝 License

None

---

Created and maintained by Kunj Shah.
