# vLLM: High-Throughput Inference Engine for LLMs

[vLLM](https://vllm.ai/) is a fast, efficient, and scalable inference and serving engine for large language models (LLMs).  
It is designed to maximize GPU utilization and serve responses at low latency, making it a great choice for production deployments.

---

## 🚀 What vLLM Uses

- **PagedAttention**: An optimized attention algorithm that reduces memory fragmentation and improves GPU memory efficiency.
- **CUDA Kernels**: Custom high-performance GPU kernels for fast inference.
- **Parallelization**: Efficient batching and request scheduling for high-throughput serving.
- **Compatibility**: Works with Hugging Face `transformers` and Hugging Face Hub models out of the box.

---

## 💡 Why vLLM is Useful

- **High Throughput**: Serve more requests per second compared to traditional Hugging Face pipelines.
- **Low Latency**: Designed for real-time applications (chatbots, copilots, RAG systems).
- **Memory Efficient**: Can handle larger batch sizes and longer context windows.
- **Production Ready**: Built-in OpenAI-compatible API server, making integration seamless.
- **Scalable**: Can be containerized (Docker/Kubernetes) and deployed on cloud GPUs.

---

## ⚙️ How to Use vLLM

### 1. Installation

```bash
pip install vllm
```
````

### 2. Running Inference

```python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Sampling configuration
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

# Run inference
outputs = llm.generate(["Hello, how are you?"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### 3. Starting an OpenAI-Compatible API Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000
```

Then you can call it just like the OpenAI API:

```python
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "EMPTY"

response = openai.ChatCompletion.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Tell me about vLLM"}]
)

print(response.choices[0].message["content"])
```

---

## 🐳 Using vLLM with Docker

Yes, you can run vLLM inside Docker for production deployments.

### Example Dockerfile

```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install Python & dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git
RUN pip3 install vllm

# Expose API port
EXPOSE 8000

# Run vLLM server
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "meta-llama/Llama-2-7b-hf", "--port", "8000"]
```

### Run with NVIDIA GPU

```bash
docker build -t vllm-server .
docker run --gpus all -p 8000:8000 vllm-server
```

---

## 📌 Key Features Recap

- ✅ Fast inference with PagedAttention
- ✅ Hugging Face & OpenAI API compatibility
- ✅ Works with Docker & Kubernetes
- ✅ Suitable for production-scale LLM serving

---

## 📚 Resources

- [vLLM Documentation](https://vllm.readthedocs.io/)
- [GitHub Repository](https://github.com/vllm-project/vllm)
- [Hugging Face Integration](https://huggingface.co/docs/transformers/main/en/vllm)

---