/*
 * Thin C wrapper around llama.cpp's json_schema_to_grammar().
 * Compiled as C++ so it can call the C++ common library, but
 * exposes a C-linkage function for the PHP extension.
 */

#include "json-schema-to-grammar.h"
#include "llama.h"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

extern "C" {

/*
 * Convert a JSON schema string to a GBNF grammar string.
 * Returns a malloc'd string that the caller must free(), or NULL on error.
 * On error, *err_out (if non-NULL) is set to a malloc'd error message.
 */
char *llama_json_schema_to_grammar(const char *json_schema, char **err_out)
{
    if (err_out) *err_out = NULL;

    try {
        auto schema = nlohmann::ordered_json::parse(json_schema);
        std::string grammar = json_schema_to_grammar(schema);
        char *result = (char *)malloc(grammar.size() + 1);
        if (!result) return NULL;
        memcpy(result, grammar.c_str(), grammar.size() + 1);
        return result;
    } catch (const std::exception &e) {
        if (err_out) {
            const char *msg = e.what();
            size_t len = strlen(msg);
            *err_out = (char *)malloc(len + 1);
            if (*err_out) memcpy(*err_out, msg, len + 1);
        }
        return NULL;
    }
}

/*
 * Safe wrapper around llama_sampler_sample that handles grammar exhaustion.
 *
 * When a grammar is fully consumed, llama.cpp either:
 *  - GGML_ASSERT(!stacks.empty()) during apply → calls ggml_abort()
 *  - throws runtime_error during accept
 *
 * We use ggml_set_abort_callback to intercept the GGML_ASSERT abort,
 * and try/catch for the accept exception.
 */
#include "ggml.h"
#include <setjmp.h>

static thread_local jmp_buf grammar_jmp;
static thread_local bool grammar_jmp_active = false;

static void grammar_abort_handler(const char *)
{
    if (grammar_jmp_active) {
        grammar_jmp_active = false;
        longjmp(grammar_jmp, 1);
    }
    /* If not our abort, call the real abort */
    abort();
}

int32_t llama_sampler_sample_safe(struct llama_sampler *chain, struct llama_context *ctx, int32_t idx, struct llama_sampler *grammar)
{
    /* No grammar — use the fast path */
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

    /* Accept on grammar — may throw if this token exhausts the grammar */
    try {
        llama_sampler_accept(grammar, token);
    } catch (const std::runtime_error &) {
        /* Grammar done after this token — token is still valid */
    }

    ggml_set_abort_callback(prev);
    return token;
}

} /* extern "C" */
