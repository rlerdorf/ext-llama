#!/bin/bash
#
# Download models and PHP documentation for the RAG example.
# Run once before using the example.
#
# Downloads:
#   - Qwen2.5-3B-Instruct (Q4_K_M) — generation model (~2GB)
#   - RAG DPO LoRA for Qwen2.5-3B — improves retrieval-augmented answers (~15MB)
#   - nomic-embed-text-v1.5 (Q8_0) — embedding model for search (~139MB)
#   - PHP manual (chunked HTML) — knowledge base (~55MB compressed)
#
# Total download: ~2.2GB
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
MODEL_DIR="${SCRIPT_DIR}/models"

mkdir -p "$DATA_DIR" "$MODEL_DIR"

# --- Base model ---
echo "==> Downloading Qwen2.5-3B-Instruct (Q4_K_M)..."
if [ ! -f "$MODEL_DIR/qwen2.5-3b-instruct-q4_k_m.gguf" ]; then
    curl -L -o "$MODEL_DIR/qwen2.5-3b-instruct-q4_k_m.gguf" \
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
else
    echo "    Already exists, skipping."
fi

# --- RAG LoRA adapter ---
echo "==> Downloading RAG DPO LoRA adapter..."
if [ ! -f "$MODEL_DIR/qwen-rag-lora.gguf" ]; then
    LORA_TMP=$(mktemp -d)
    curl -sL -o "$LORA_TMP/adapter_model.safetensors" \
        "https://huggingface.co/AnjanSB/Qwen2.5-3B-Instruct-NQ-RAG-DPO-LoRA/resolve/main/adapter_model.safetensors"
    curl -sL -o "$LORA_TMP/adapter_config.json" \
        "https://huggingface.co/AnjanSB/Qwen2.5-3B-Instruct-NQ-RAG-DPO-LoRA/resolve/main/adapter_config.json"

    # The converter needs the base model's config.json
    CONFIG_TMP=$(mktemp -d)
    curl -sL -o "$CONFIG_TMP/config.json" \
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main/config.json"

    echo "    Converting LoRA to GGUF format..."
    LLAMA_CPP="${LLAMA_CPP_DIR:-$(dirname "$(which llama-cli 2>/dev/null || echo /home/rasmus/src/llama-cpp/build/bin/llama-cli)")/../..}"
    if [ -f "$LLAMA_CPP/convert_lora_to_gguf.py" ]; then
        python3 "$LLAMA_CPP/convert_lora_to_gguf.py" \
            --base "$CONFIG_TMP" \
            --outfile "$MODEL_DIR/qwen-rag-lora.gguf" \
            "$LORA_TMP" 2>&1 | grep -E "(INFO:lora|INFO:gguf|ERROR)"
    else
        echo "    WARNING: convert_lora_to_gguf.py not found."
        echo "    Set LLAMA_CPP_DIR to your llama.cpp source directory and re-run."
        echo "    Example: LLAMA_CPP_DIR=/path/to/llama.cpp $0"
    fi
    rm -rf "$LORA_TMP" "$CONFIG_TMP"
else
    echo "    Already exists, skipping."
fi

# --- Embedding model ---
echo "==> Downloading nomic-embed-text-v1.5 (Q8_0)..."
if [ ! -f "$MODEL_DIR/nomic-embed-text-v1.5-q8_0.gguf" ]; then
    curl -L -o "$MODEL_DIR/nomic-embed-text-v1.5-q8_0.gguf" \
        "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf"
else
    echo "    Already exists, skipping."
fi

# --- PHP documentation ---
echo "==> Downloading PHP manual (chunked HTML)..."
if [ ! -d "$DATA_DIR/php-manual" ]; then
    curl -sL "https://www.php.net/distributions/manual/php_manual_en.tar.gz" \
        | tar xzf - -C "$DATA_DIR"
    mv "$DATA_DIR/php-chunked-xhtml" "$DATA_DIR/php-manual"
    echo "    Extracted $(ls "$DATA_DIR/php-manual" | wc -l) pages."
else
    echo "    Already exists, skipping."
fi

echo ""
echo "Done! To build the search index, run:"
echo "  php index.php"
echo ""
echo "Then ask questions:"
echo "  php search.php \"How do I sort an array?\""
