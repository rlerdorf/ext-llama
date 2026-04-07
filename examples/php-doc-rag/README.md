# PHP Documentation RAG

Ask questions about PHP and get answers grounded in the official PHP documentation, powered by local LLMs running entirely in PHP via ext-llama.

## How it works

1. **Index** — The PHP manual (~11,000 HTML pages) is stripped to plain text, embedded with [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF), and stored as a binary vector index.

2. **Search** — Your question is embedded with the same model, and the most relevant documentation pages are found via cosine similarity.

3. **Answer** — The retrieved docs are fed as context to [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) with a [RAG-tuned LoRA adapter](https://huggingface.co/AnjanSB/Qwen2.5-3B-Instruct-NQ-RAG-DPO-LoRA), which generates an answer with code examples.

All inference runs locally — no API keys, no cloud services.

## Setup

```bash
# 1. Download models and PHP documentation (~2.2GB total)
bash download.sh

# 2. Build the search index (~10 minutes on CPU)
php index.php

# 3. Ask questions
php search.php "How do I sort an array by value?"
php search.php --interactive
```

## Requirements

- ext-llama installed and enabled
- Python 3 with `torch` and `safetensors` (for LoRA conversion during download)
- ~4GB RAM (embedding model + generation model)
- llama.cpp source tree (for `convert_lora_to_gguf.py`)

## Models

| Model | Purpose | Size |
|-------|---------|------|
| Qwen2.5-3B-Instruct Q4_K_M | Answer generation | 2.0 GB |
| RAG DPO LoRA (AnjanSB) | Improves RAG-style answers | 15 MB |
| nomic-embed-text-v1.5 Q8_0 | Document/query embedding | 139 MB |
