# ext-llama

PHP extension wrapping llama.cpp for running GGUF models directly in PHP processes.

## Build

Requires llama.cpp built with shared libs:

```bash
cd /home/rasmus/src/llama-cpp/build
cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF
make -j$(nproc) llama ggml common
```

Then build the extension:

```bash
phpize
./configure --with-llama=/home/rasmus/src/llama-cpp
make
```

Test: `php -dextension=modules/llama.so test.php /path/to/model.gguf`

The extension links against `libllama.so` (dynamic) and `libcommon.a` (static, for JSON schema support). If libllama is not installed system-wide, set `LD_LIBRARY_PATH` to the build/bin directory.

After `make clean`, always re-run `phpize --clean && phpize` before `./configure` since config.m4 changes require a full autoconf regeneration.

## Architecture

### Source files

- `llama.c` — Main extension: all PHP classes, methods, module lifecycle. Pure C.
- `php_llama.h` — Object structs and inline helpers.
- `json_schema_shim.cpp` — C++ bridge for two things: (1) `json_schema_to_grammar()` from libcommon (needs nlohmann/json), and (2) `llama_sampler_sample_safe()` which wraps sampling to handle grammar exhaustion (GGML_ASSERT aborts and C++ exceptions from the grammar sampler).
- `config.m4` — Build system. Uses `PHP_REQUIRE_CXX()` for mixed C/C++ compilation. Statically links libcommon.a for JSON schema conversion.

### Classes

All in the `Llama\` namespace:

- **`Model`** — Loads a GGUF file. Persistent: models are cached in a process-wide `HashTable` keyed by realpath and survive across PHP-FPM requests. The destructor skips freeing persistent models; `MSHUTDOWN` frees them all. Uses `mmap` (shared across workers via kernel page cache) and `mlock` (pinned in RAM) by default.
- **`Context`** — Lightweight per-request inference context. Owns a `llama_context`, holds a zval ref to its Model to prevent GC. Methods: `complete()`, `stream()`, `chat()`, `embed()`, `applyLoRA()`, `clearLoRA()`.
- **`LoRA`** — Loads a GGUF LoRA adapter tied to a Model. Multiple LoRAs can be hot-swapped on a Context in sub-millisecond time via `applyLoRA()`.
- **`CompletionIterator`** — Returned by `stream()`. Implements `Iterator` for `foreach` token-by-token streaming. Holds a sampler chain + optional grammar sampler, advances one token per `next()` call.
- **`Exception`** — Extends `\Exception`.

### Grammar/JSON schema sampling

The grammar sampler is kept **separate** from the sampling chain, matching llama.cpp's own architecture (see `common/sampling.cpp`). The flow:

1. Get logits from context
2. Apply grammar sampler (constrains candidates to grammar-valid tokens)
3. Apply sampling chain (top_k, top_p, min_p, temp, penalties, dist)
4. Accept selected token on both chain and grammar

Grammar exhaustion is handled in `llama_sampler_sample_safe()`:
- `apply()` may trigger `GGML_ASSERT(!stacks.empty())` — intercepted via `ggml_set_abort_callback` + `setjmp/longjmp`
- `accept()` may throw `std::runtime_error` — caught with try/catch, token is still returned since it was validly selected

When no grammar is active, `llama_sampler_sample_safe()` falls through to the standard `llama_sampler_sample()` fast path.

### Memory model

- **Model weights**: mmap'd from the GGUF file. Shared across all PHP-FPM workers via the kernel page cache. Physical memory cost = 1x model size regardless of worker count.
- **Model metadata**: Small per-worker heap allocation (~KB). Cached in a static `HashTable` (`persistent_models`) that persists across requests within a worker.
- **Context (KV cache)**: Per-request, freed when the Context object is destroyed.
- **LoRA adapters**: Per-adapter allocation, tied to model lifetime. Hot-swapping only patches pointers (0ms).

## Conventions

- PHP method names use camelCase (`applyLoRA`, `chatTemplate`, `nEmbd`)
- C functions follow PHP naming: `PHP_METHOD(Llama_Model, methodName)`
- All methods that need a live model/context check for NULL and throw `Llama\Exception`
- Options arrays use snake_case keys (`max_tokens`, `top_k`, `json_schema`, `repeat_penalty`)
- Sampler defaults: temperature=0.8, top_k=40, top_p=0.95, min_p=0.05, repeat_penalty=1.1, penalty_last_n=64

## Test models

Available at `/tmp/`:
- `tinyllama.gguf` — TinyLlama 1.1B Q2_K (461MB, fast, low quality)
- `qwen2.5-3b-instruct-q4_k_m.gguf` — Qwen2.5 3B Q4_K_M (2GB, good quality)
- `smallthinker-lora-f16.gguf` — SmallThinker reasoning LoRA for Qwen2.5-3B (533MB)
- `qwen-abliterated-lora.gguf` — Abliterated LoRA for Qwen2.5-3B (532MB)
