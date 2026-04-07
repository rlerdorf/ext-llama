# ext-llama

A PHP extension for running GGUF large language models directly in PHP using [llama.cpp](https://github.com/ggml-org/llama.cpp). No HTTP servers, no exec(), no Python — just load a model and generate text from your PHP script.

## Why not just use llama-server?

llama.cpp ships with [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server), an HTTP server that exposes an OpenAI-compatible API. You can talk to it from PHP with any HTTP client. That's a perfectly good approach and the right choice if:

- You want to share one model across multiple applications or languages
- You already have infrastructure for managing sidecar services
- You need the server's built-in features (parallel slots, API key auth, metrics)

This extension takes a different approach: it links directly against `libllama` and runs inference **inside the PHP process**. The tradeoffs vs. llama-server:

| | ext-llama | llama-server + HTTP client |
|---|---|---|
| Moving parts | Just PHP | PHP + separate server process |
| Deployment | `extension=llama` in php.ini | Manage a sidecar daemon |
| Latency | Direct C calls | HTTP round-trip (~1ms loopback) |
| Model memory | mmap shared across PHP-FPM workers | Dedicated server process |
| LoRA hot-swap | Sub-millisecond, per-request | Requires server restart or API call |
| Streaming | Native PHP `Iterator` | SSE parsing |
| Grammar/JSON schema | Built-in | Built-in (server supports it too) |

In short: if you want fewer moving parts and tighter integration, use the extension. If you want a standalone inference service, use llama-server.

## Requirements

- PHP 8.4+
- llama.cpp built with shared libraries

## Installation

Build llama.cpp:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install  # installs libllama.so, headers
```

For CUDA (NVIDIA GPU) support, add `-DGGML_CUDA=ON` to the cmake line. Other backends like Vulkan (`-DGGML_VULKAN=ON`) and Metal (macOS, enabled by default) work the same way. The PHP extension does not need to be recompiled when switching backends — only libllama does.

Build the extension:

```bash
git clone https://github.com/rlerdorf/ext-llama
cd ext-llama
phpize
./configure --with-llama=/path/to/llama-cpp
make
sudo make install
```

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

$model->tokenize("Hello");       // [1, 15043] — token IDs
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
