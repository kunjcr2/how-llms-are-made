The video [“Lecture 2 - Chain of Thought Reasoning”](https://www.youtube.com/watch?v=fZNNqcN_UQM) from the _Reasoning LLMs from Scratch_ series by Vizuara explores how to enhance reasoning in large language models (LLMs) using **Inference-Time Compute Scaling**—a method that boosts reasoning capabilities without modifying the base model, simply by increasing computational effort during inference.

### 🧠 Key Concepts Covered

- **Chain of Thought (CoT) Prompting**:
  - **Few-shot CoT**: Uses examples with intermediate reasoning steps.
  - **Zero-shot CoT**: Encourages reasoning without examples, relying on model capabilities.
- **Emergent Reasoning**: The video discusses how reasoning emerges as a property of model size.
- **Datasets Introduced**:
  - Arithmetic, logical, and symbolic reasoning datasets are used to benchmark performance.

### 🧪 Hands-On Projects

Three Google Colab notebooks are provided:

1. **Model Size vs. CoT Accuracy** – Tests how scaling affects reasoning.
2. **Prompting Comparison** – Evaluates baseline vs. few-shot vs. zero-shot CoT.
3. **Zero-shot CoT on Custom LLM** – Applies techniques to a scratch-built model.

### 📚 Reference Papers

- Few-shot CoT: [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- Zero-shot CoT: [arXiv:2205.11916](https://arxiv.org/abs/2205.11916)

This lecture is a deep dive into how prompting strategies and compute scaling can unlock latent reasoning abilities in LLMs—perfect for anyone building or fine-tuning models from scratch.
