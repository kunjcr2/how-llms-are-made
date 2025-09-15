# How LLMs Are Made

This repository is a comprehensive guide and implementation walkthrough of Large Language Models (LLMs), from basic concepts to advanced architectures. It includes theoretical explanations, practical implementations, and real-world examples of various LLM architectures.

## 📚 Repository Structure

### 1. GPT (Generative Pre-trained Transformer)

The GPT section covers the fundamental architecture of modern LLMs:

- **1_tokenizer**: Implementation of tokenization techniques with example texts
- **2_attention**: Detailed explanation of attention mechanisms with visualizations
- **3_architecture**: Complete GPT architecture implementation and explanation
- **4_training**: Training methodology and practices
- **5_post_training**: Post-training optimization techniques
- **6_finetune**: Full fine-tuning and LoRA adaptation methods

### 2. DeepSeek

Implementation and explanation of DeepSeek's advanced architectures:

- **Mixture of Experts (MoE)**: Implementation of expert-based models
- **Multi-head Latent Attention**: Advanced attention mechanisms
- **Position Embeddings**: RoPE (Rotary Position Embedding) implementation
- Technical and theoretical documentation available in markdown files

### 3. MLandDL

Core Machine Learning and Deep Learning concepts:

- BackPropagation implementation (based on Andrej Karpathy's work)
- General backpropagation theory
- Prompting techniques and strategies

### 4. Kimi

Documentation about Mixture of Dendrites (MoD):

- Theoretical explanations
- Implementation details
- Working principles

### 5. My Models

Custom implementations including:

- **GatorGPT**: A lightweight transformer model with:
  - Grouped Query Attention
  - RoPE embeddings
  - SwiGLU activation
  - vLLM deployment setup

## 🛠️ Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- transformers
- tiktoken (for tokenization)
- vLLM (for deployment)

### Installation

```bash
git clone https://github.com/kunjcr2/how-llms-are-made.git
cd how-llms-are-made
```

## 📓 Notebooks

The repository contains multiple Jupyter notebooks that demonstrate:

1. Tokenizer implementation and usage
2. Attention mechanism visualization
3. Complete architecture implementation
4. Training and fine-tuning procedures
5. Model deployment strategies

## 🚀 Models

### GatorGPT

- Lightweight transformer model
- Trained on TinyStories dataset
- Uses modern architecture choices:
  - Grouped Query Attention
  - RoPE embeddings
  - SwiGLU activation
  - Efficient vLLM deployment

## 📖 Documentation

Each section contains detailed documentation:

- Markdown files explaining theoretical concepts
- Implementation notes
- Architecture diagrams
- Training procedures
- Deployment guidelines

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

None

---

Created and maintained by Kunj Shah.
