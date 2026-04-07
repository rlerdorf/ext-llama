/*
 * Safe sampler wrapper that handles grammar exhaustion.
 *
 * When a grammar is fully consumed, llama.cpp either:
 *  - GGML_ASSERT(!stacks.empty()) during apply, which calls ggml_abort()
 *  - throws runtime_error during accept
 *
 * We use ggml_set_abort_callback + setjmp/longjmp to intercept the
 * GGML_ASSERT abort, and try/catch for the accept exception.
 *
 * When no grammar is active, this falls through to the standard
 * llama_sampler_sample() fast path with zero overhead.
 */

#include "llama.h"
#include "ggml.h"
#include <stdexcept>
#include <vector>
#include <setjmp.h>

extern "C" {

static thread_local jmp_buf grammar_jmp;
static thread_local bool grammar_jmp_active = false;

static void grammar_abort_handler(const char *)
{
    if (grammar_jmp_active) {
        grammar_jmp_active = false;
        longjmp(grammar_jmp, 1);
    }
    abort();
}

int32_t llama_sampler_sample_safe(struct llama_sampler *chain, struct llama_context *ctx, int32_t idx, struct llama_sampler *grammar)
{
    /* No grammar: use the fast path */
    if (!grammar) {
        return llama_sampler_sample(chain, ctx, idx);
    }

    ggml_abort_callback_t prev = ggml_set_abort_callback(grammar_abort_handler);

    /* Build candidate array from logits */
    float *logits = llama_get_logits_ith(ctx, idx);
    if (!logits) {
        ggml_set_abort_callback(prev);
        return -1;
    }

    const struct llama_model *model = llama_get_model(ctx);
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token_data> candidates(n_vocab);
    for (int32_t i = 0; i < n_vocab; i++) {
        candidates[i] = { i, logits[i], 0.0f };
    }
    llama_token_data_array cur_p = { candidates.data(), (size_t)n_vocab, -1, false };

    /*
     * Phase 1: Apply grammar (constrains candidates).
     * May GGML_ASSERT if grammar stacks are empty (grammar done).
     */
    if (setjmp(grammar_jmp) != 0) {
        ggml_set_abort_callback(prev);
        return -1;
    }
    grammar_jmp_active = true;
    llama_sampler_apply(grammar, &cur_p);
    grammar_jmp_active = false;

    /* Phase 2: Apply sampling chain (top_k, top_p, temp, dist) */
    llama_sampler_apply(chain, &cur_p);

    if (cur_p.selected < 0 || cur_p.selected >= (int64_t)cur_p.size) {
        ggml_set_abort_callback(prev);
        return -1;
    }
    int32_t token = cur_p.data[cur_p.selected].id;

    /* Phase 3: Accept on chain (always safe) */
    llama_sampler_accept(chain, token);

    /* Accept on grammar. May throw if this token exhausts the grammar. */
    try {
        llama_sampler_accept(grammar, token);
    } catch (const std::runtime_error &) {
        /* Grammar done after this token. Token is still valid. */
    }

    ggml_set_abort_callback(prev);
    return token;
}

} /* extern "C" */
