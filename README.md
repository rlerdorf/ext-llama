# ext-llama

A PHP extension for running GGUF large language models directly in PHP using [llama.cpp](https://github.com/ggml-org/llama.cpp). No HTTP servers, no exec(), no Python. Just load a model and generate text from your PHP script.

## Why not just use llama-server?

For larger models and high-concurrency workloads, you probably should. llama.cpp ships with [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server), an HTTP server that exposes an OpenAI-compatible API. You can talk to it from PHP with any HTTP client. llama-server is the better choice when:

- **High concurrency.** llama-server holds a single copy of the model and handles parallel requests via slots. With ext-llama, each PHP-FPM worker creates its own inference context. Model *weights* are shared across workers via mmap (no duplication in system RAM), but GPU (CUDA/Metal) memory is per-process. If you're offloading a 7B model to GPU and running 4 FPM workers, that's 4x the VRAM. A dedicated llama-server avoids this entirely.
- **Large models.** For 13B+ models on GPU, the single-process architecture of llama-server is more memory-efficient.
- **Multi-language / multi-app.** If other services besides PHP need the same model, a shared server makes more sense than loading it in every process.

ext-llama is a better fit for **embedded / low-concurrency setups** where simplicity matters:

- Small to medium models (1-7B) running on CPU, or on GPU with a single or very few FPM workers where the per-worker VRAM cost is acceptable
- Dedicated appliances, IoT, edge servers, or internal tools where you want one less daemon to manage
- Use cases like RAG, structured extraction, or chat where a single PHP process handles the request end-to-end
- LoRA hot-swapping per request, allowing you to switch "personalities" in sub-millisecond time without touching a server config

| | ext-llama | llama-server + HTTP client |
|---|---|---|
| Moving parts | Just PHP | PHP + separate server process |
| Deployment | `extension=llama` in php.ini | Manage a sidecar daemon |
| Latency | Direct C calls | HTTP round-trip (~1ms loopback) |
| Model memory (CPU) | mmap shared across workers | Single process |
| Model memory (GPU) | Per-worker VRAM allocation | Single VRAM allocation |
| LoRA hot-swap | Sub-millisecond, per-request | Server restart or API call |
| Streaming | Native PHP `Iterator` | SSE parsing |
| Concurrency | Limited by FPM workers | Built-in parallel slots |

## Requirements

- PHP 8.4+
- llama.cpp built with shared libraries

## Installation

### 1. Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j$(nproc) llama ggml common
sudo make install  # installs libllama.so and headers to /usr/local
```

For CUDA (NVIDIA GPU) support, add `-DGGML_CUDA=ON` to the cmake line. Other backends like Vulkan (`-DGGML_VULKAN=ON`) and Metal (macOS, enabled by default) work the same way. The PHP extension does not need to be recompiled when switching backends. Only libllama does.

### 2. Build the extension

Point `--with-llama` at the **llama.cpp source tree** (not the install prefix). This is important because it gives the build system access to `libcommon.a` and the vendored `nlohmann/json` headers, which are needed for JSON schema constrained generation. These files are not installed by `make install`.

Via PIE:

```bash
pie install rlerdorf/ext-llama --with-llama=/path/to/llama.cpp
```

Or manually:

```bash
git clone https://github.com/rlerdorf/ext-llama
cd ext-llama
phpize
./configure --with-llama=/path/to/llama.cpp
make
sudo make install
```

If you point `--with-llama` at a system prefix like `/usr/local` instead of the source tree, the extension will still build and work, but the `json_schema` option will not be available. GBNF grammars (the `grammar` option) always work regardless. The configure output will tell you which features are enabled:

```
checking for llama.cpp common library (json-schema-to-grammar)... yes
```

### 3. Enable the extension

Add to your `php.ini`:

```ini
extension=llama
```

## Quick Start

```php
$model = new Llama\Model('/path/to/model.gguf');
$ctx = new Llama\Context($model, ['n_ctx' => 2048]);

echo $ctx->complete("The capital of France is", ['max_tokens' => 32]);
```

## API

### Llama\Model

```php
// Load a GGUF model (cached across requests in PHP-FPM)
$model = new Llama\Model('/path/to/model.gguf', [
    'n_gpu_layers' => -1,    // offload all layers to GPU (-1=all, 0=CPU only)
    'use_mmap'     => true,  // default: true
    'use_mlock'    => true,  // default: true, pin pages in RAM
]);

$model->desc();              // "llama 3B Q4_K - Medium"
$model->size();              // model file size in bytes
$model->nParams();           // parameter count
$model->nEmbd();             // embedding dimensions
$model->nLayer();            // layer count
$model->chatTemplate();      // built-in Jinja chat template, or null
$model->meta('general.name');// read GGUF metadata by key

$model->tokenize("Hello");       // [1, 15043]
$model->detokenize([1, 15043]);  // " Hello"
```

### Llama\Context

```php
$ctx = new Llama\Context($model, [
    'n_ctx'      => 2048,  // context size
    'n_batch'    => 512,   // batch size
    'n_threads'  => 4,     // CPU threads
    'embeddings' => false, // set true for embed()
    'flash_attn' => false, // flash attention
]);
```

**Text completion:**

```php
$text = $ctx->complete("Once upon a time", [
    'max_tokens'     => 256,
    'temperature'    => 0.8,
    'top_k'          => 40,
    'top_p'          => 0.95,
    'min_p'          => 0.05,
    'repeat_penalty' => 1.1,
    'seed'           => 42,
]);
```

**Chat** (applies the model's built-in chat template):

```php
$reply = $ctx->chat([
    ['role' => 'system', 'content' => 'You are a helpful assistant.'],
    ['role' => 'user',   'content' => 'What is PHP?'],
], ['max_tokens' => 256]);
```

**Streaming** (token by token):

```php
foreach ($ctx->stream("Tell me a story", ['max_tokens' => 256]) as $piece) {
    echo $piece;
    flush();
}
```

**Embeddings:**

```php
$ctx = new Llama\Context($model, ['embeddings' => true]);
$vector = $ctx->embed("Some text");  // float[]
```

**Constrained generation** with GBNF grammar or JSON schema:

```php
// Force yes/no output
$answer = $ctx->complete("Is the sky blue? ", [
    'grammar' => 'root ::= ("yes" | "no")',
]);

// Force valid JSON matching a schema
$json = $ctx->complete("Output a person as JSON:", [
    'json_schema' => json_encode([
        'type' => 'object',
        'properties' => [
            'name' => ['type' => 'string'],
            'age'  => ['type' => 'integer'],
        ],
        'required' => ['name', 'age'],
    ]),
]);
// {"name":"Alice","age":30}
```

### Llama\LoRA

```php
// Load adapters (one-time cost, ~200ms each)
$code = new Llama\LoRA($model, '/path/to/code-lora.gguf');
$chat = new Llama\LoRA($model, '/path/to/chat-lora.gguf');

// Hot-swap in sub-millisecond time
$ctx->applyLoRA($code);
$ctx->applyLoRA($chat);          // replaces previous
$ctx->applyLoRA($chat, 0.5);     // with scale

// Blend multiple LoRAs
$ctx->applyLoRA([$code, $chat], [0.6, 0.4]);

// Remove all adapters
$ctx->clearLoRA();

// Read adapter metadata
$code->meta('general.name');
```

### Llama\Exception

All errors throw `Llama\Exception` (extends `\Exception`):

```php
try {
    $model = new Llama\Model('/nonexistent.gguf');
} catch (Llama\Exception $e) {
    echo $e->getMessage(); // "Model file not found: /nonexistent.gguf"
}
```

## Memory Model

In a PHP-FPM deployment with 10 workers serving a 4GB model:

| What | Memory | Lifetime |
|------|--------|----------|
| Model weights (mmap) | 4GB shared | Process (shared across all workers) |
| Model metadata | ~KB per worker | Worker (persistent across requests) |
| KV cache | ~MB per context | Request |
| LoRA adapters | ~MB each | Worker |
| LoRA hot-swap | 0 bytes | Instant |

## License

PHP License (same as PHP itself).
