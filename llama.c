#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "php_ini.h"
#include "ext/standard/info.h"
#include "zend_exceptions.h"
#include "zend_interfaces.h"
#include "zend_smart_str.h"
#include "php_llama.h"

/* From json_schema_shim.cpp */
extern char *llama_json_schema_to_grammar(const char *json_schema, char **err_out);
extern int32_t llama_sampler_sample_safe(struct llama_sampler *smpl, struct llama_context *ctx, int32_t idx, struct llama_sampler *grammar);

/* Class entries */
zend_class_entry *llama_ce_model;
zend_class_entry *llama_ce_context;
zend_class_entry *llama_ce_lora;
zend_class_entry *llama_ce_completion_iterator;
zend_class_entry *llama_ce_exception;

/* Object handlers */
static zend_object_handlers llama_model_handlers;
static zend_object_handlers llama_context_handlers;

/* Backend initialized flag */
static bool llama_backend_initialized = false;

/* Persistent model cache: maps realpath -> struct llama_model*
 * Models in this cache survive across requests within the same
 * PHP-FPM worker process. They are freed at module shutdown. */
typedef struct {
    struct llama_model *model;
    const struct llama_vocab *vocab;
} llama_persistent_model;

static HashTable persistent_models;

/* Silent log callback - suppresses llama.cpp's verbose output */
static void llama_log_callback_null(enum ggml_log_level level, const char *text, void *user_data)
{
    (void)level;
    (void)text;
    (void)user_data;
}

/* ============================================================
 * Llama\Model
 * ============================================================ */

static zend_object *llama_model_create_object(zend_class_entry *ce)
{
    llama_model_obj *intern = zend_object_alloc(sizeof(llama_model_obj), ce);
    intern->model = NULL;
    intern->vocab = NULL;
    intern->path = NULL;
    intern->persistent = false;
    zend_object_std_init(&intern->std, ce);
    object_properties_init(&intern->std, ce);
    intern->std.handlers = &llama_model_handlers;
    return &intern->std;
}

static void llama_model_free_object(zend_object *obj)
{
    llama_model_obj *intern = llama_model_from_obj(obj);
    if (intern->model && !intern->persistent) {
        llama_model_free(intern->model);
    }
    intern->model = NULL;
    if (intern->path) {
        efree(intern->path);
        intern->path = NULL;
    }
    zend_object_std_dtor(&intern->std);
}

/* Model::__construct(string $modelPath, array $params = []) */
PHP_METHOD(Llama_Model, __construct)
{
    char *model_path;
    size_t model_path_len;
    zval *params = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_STRING(model_path, model_path_len)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY(params)
    ZEND_PARSE_PARAMETERS_END();

    llama_model_obj *intern = Z_LLAMA_MODEL_P(ZEND_THIS);

    if (intern->model) {
        zend_throw_exception(llama_ce_exception, "Model already loaded", 0);
        RETURN_THROWS();
    }

    /* Check file exists */
    if (VCWD_ACCESS(model_path, F_OK) != 0) {
        zend_throw_exception_ex(llama_ce_exception, 0, "Model file not found: %s", model_path);
        RETURN_THROWS();
    }

    /* Resolve to real path for cache key */
    char resolved[PATH_MAX];
    if (!VCWD_REALPATH(model_path, resolved)) {
        zend_throw_exception_ex(llama_ce_exception, 0, "Failed to resolve path: %s", model_path);
        RETURN_THROWS();
    }

    /* Check persistent cache first */
    llama_persistent_model *cached = zend_hash_str_find_ptr(&persistent_models, resolved, strlen(resolved));
    if (cached) {
        intern->model = cached->model;
        intern->vocab = cached->vocab;
        intern->path = estrndup(resolved, strlen(resolved));
        intern->persistent = true;
        return;
    }

    struct llama_model_params model_params = llama_model_default_params();
    model_params.use_mlock = true; /* pin shared pages in RAM by default */

    /* Parse params array */
    if (params) {
        zval *val;
        if ((val = zend_hash_str_find(Z_ARRVAL_P(params), "n_gpu_layers", sizeof("n_gpu_layers") - 1)) != NULL) {
            model_params.n_gpu_layers = (int32_t)zval_get_long(val);
        }
        if ((val = zend_hash_str_find(Z_ARRVAL_P(params), "use_mmap", sizeof("use_mmap") - 1)) != NULL) {
            model_params.use_mmap = zend_is_true(val);
        }
        if ((val = zend_hash_str_find(Z_ARRVAL_P(params), "use_mlock", sizeof("use_mlock") - 1)) != NULL) {
            model_params.use_mlock = zend_is_true(val);
        }
        if ((val = zend_hash_str_find(Z_ARRVAL_P(params), "check_tensors", sizeof("check_tensors") - 1)) != NULL) {
            model_params.check_tensors = zend_is_true(val);
        }
    }

    struct llama_model *model = llama_model_load_from_file(resolved, model_params);
    if (!model) {
        zend_throw_exception_ex(llama_ce_exception, 0, "Failed to load model: %s", model_path);
        RETURN_THROWS();
    }

    /* Store in persistent cache */
    llama_persistent_model entry;
    entry.model = model;
    entry.vocab = llama_model_get_vocab(model);
    zend_hash_str_update_mem(&persistent_models, resolved, strlen(resolved), &entry, sizeof(entry));

    intern->model = model;
    intern->vocab = entry.vocab;
    intern->path = estrndup(resolved, strlen(resolved));
    intern->persistent = true;
}

/* Model::desc(): string */
PHP_METHOD(Llama_Model, desc)
{
    ZEND_PARSE_PARAMETERS_NONE();
    llama_model_obj *intern = Z_LLAMA_MODEL_P(ZEND_THIS);
    if (!intern->model) {
        zend_throw_exception(llama_ce_exception, "Model not loaded", 0);
        RETURN_THROWS();
    }
    char buf[256];
    llama_model_desc(intern->model, buf, sizeof(buf));
    RETURN_STRING(buf);
}

/* Model::size(): int */
PHP_METHOD(Llama_Model, size)
{
    ZEND_PARSE_PARAMETERS_NONE();
    llama_model_obj *intern = Z_LLAMA_MODEL_P(ZEND_THIS);
    if (!intern->model) {
        zend_throw_exception(llama_ce_exception, "Model not loaded", 0);
        RETURN_THROWS();
    }
    RETURN_LONG((zend_long)llama_model_size(intern->model));
}

/* Model::nParams(): int */
PHP_METHOD(Llama_Model, nParams)
{
    ZEND_PARSE_PARAMETERS_NONE();
    llama_model_obj *intern = Z_LLAMA_MODEL_P(ZEND_THIS);
    if (!intern->model) {
        zend_throw_exception(llama_ce_exception, "Model not loaded", 0);
        RETURN_THROWS();
    }
    RETURN_LONG((zend_long)llama_model_n_params(intern->model));
}

/* Model::nEmbd(): int */
PHP_METHOD(Llama_Model, nEmbd)
{
    ZEND_PARSE_PARAMETERS_NONE();
    llama_model_obj *intern = Z_LLAMA_MODEL_P(ZEND_THIS);
    if (!intern->model) {
        zend_throw_exception(llama_ce_exception, "Model not loaded", 0);
        RETURN_THROWS();
    }
    RETURN_LONG(llama_model_n_embd(intern->model));
}

/* Model::nLayer(): int */
PHP_METHOD(Llama_Model, nLayer)
{
    ZEND_PARSE_PARAMETERS_NONE();
    llama_model_obj *intern = Z_LLAMA_MODEL_P(ZEND_THIS);
    if (!intern->model) {
        zend_throw_exception(llama_ce_exception, "Model not loaded", 0);
        RETURN_THROWS();
    }
    RETURN_LONG(llama_model_n_layer(intern->model));
}

/* Model::chatTemplate(): ?string */
PHP_METHOD(Llama_Model, chatTemplate)
{
    ZEND_PARSE_PARAMETERS_NONE();
    llama_model_obj *intern = Z_LLAMA_MODEL_P(ZEND_THIS);
    if (!intern->model) {
        zend_throw_exception(llama_ce_exception, "Model not loaded", 0);
        RETURN_THROWS();
    }
    const char *tmpl = llama_model_chat_template(intern->model, NULL);
    if (tmpl) {
        RETURN_STRING(tmpl);
    }
    RETURN_NULL();
}

/* Model::meta(string $key): ?string */
PHP_METHOD(Llama_Model, meta)
{
    char *key;
    size_t key_len;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STRING(key, key_len)
    ZEND_PARSE_PARAMETERS_END();

    llama_model_obj *intern = Z_LLAMA_MODEL_P(ZEND_THIS);
    if (!intern->model) {
        zend_throw_exception(llama_ce_exception, "Model not loaded", 0);
        RETURN_THROWS();
    }

    char buf[512];
    int32_t ret = llama_model_meta_val_str(intern->model, key, buf, sizeof(buf));
    if (ret >= 0) {
        RETURN_STRINGL(buf, ret);
    }
    RETURN_NULL();
}

/* Model::tokenize(string $text, bool $addSpecial = true, bool $parseSpecial = false): array */
PHP_METHOD(Llama_Model, tokenize)
{
    char *text;
    size_t text_len;
    bool add_special = true;
    bool parse_special = false;

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_STRING(text, text_len)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(add_special)
        Z_PARAM_BOOL(parse_special)
    ZEND_PARSE_PARAMETERS_END();

    llama_model_obj *intern = Z_LLAMA_MODEL_P(ZEND_THIS);
    if (!intern->model) {
        zend_throw_exception(llama_ce_exception, "Model not loaded", 0);
        RETURN_THROWS();
    }

    /* First call to get token count */
    int32_t n_tokens = llama_tokenize(intern->vocab, text, (int32_t)text_len, NULL, 0, add_special, parse_special);
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
    }

    llama_token *tokens = emalloc(sizeof(llama_token) * n_tokens);
    int32_t actual = llama_tokenize(intern->vocab, text, (int32_t)text_len, tokens, n_tokens, add_special, parse_special);

    if (actual < 0) {
        efree(tokens);
        zend_throw_exception(llama_ce_exception, "Tokenization failed", 0);
        RETURN_THROWS();
    }

    array_init_size(return_value, actual);
    for (int32_t i = 0; i < actual; i++) {
        add_next_index_long(return_value, tokens[i]);
    }
    efree(tokens);
}

/* Model::detokenize(array $tokens): string */
PHP_METHOD(Llama_Model, detokenize)
{
    zval *tokens_arr;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(tokens_arr)
    ZEND_PARSE_PARAMETERS_END();

    llama_model_obj *intern = Z_LLAMA_MODEL_P(ZEND_THIS);
    if (!intern->model) {
        zend_throw_exception(llama_ce_exception, "Model not loaded", 0);
        RETURN_THROWS();
    }

    int32_t n_tokens = zend_hash_num_elements(Z_ARRVAL_P(tokens_arr));
    llama_token *tokens = emalloc(sizeof(llama_token) * n_tokens);

    int32_t i = 0;
    zval *val;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(tokens_arr), val) {
        tokens[i++] = (llama_token)zval_get_long(val);
    } ZEND_HASH_FOREACH_END();

    /* First call to get size */
    int32_t text_len = llama_detokenize(intern->vocab, tokens, n_tokens, NULL, 0, false, false);
    if (text_len < 0) {
        text_len = -text_len;
    }

    char *text = emalloc(text_len + 1);
    int32_t actual = llama_detokenize(intern->vocab, tokens, n_tokens, text, text_len, false, false);

    efree(tokens);

    if (actual < 0) {
        efree(text);
        zend_throw_exception(llama_ce_exception, "Detokenization failed", 0);
        RETURN_THROWS();
    }

    text[actual] = '\0';
    RETVAL_STRINGL(text, actual);
    efree(text);
}

/* ============================================================
 * Llama\LoRA
 * ============================================================ */

static zend_object_handlers llama_lora_handlers;

static zend_object *llama_lora_create_object(zend_class_entry *ce)
{
    llama_lora_obj *intern = zend_object_alloc(sizeof(llama_lora_obj), ce);
    intern->adapter = NULL;
    intern->path = NULL;
    ZVAL_UNDEF(&intern->model_zval);
    zend_object_std_init(&intern->std, ce);
    object_properties_init(&intern->std, ce);
    intern->std.handlers = &llama_lora_handlers;
    return &intern->std;
}

static void llama_lora_free_object(zend_object *obj)
{
    llama_lora_obj *intern = llama_lora_from_obj(obj);
    if (intern->adapter) {
        llama_adapter_lora_free(intern->adapter);
        intern->adapter = NULL;
    }
    if (intern->path) {
        efree(intern->path);
        intern->path = NULL;
    }
    zval_ptr_dtor(&intern->model_zval);
    zend_object_std_dtor(&intern->std);
}

/* LoRA::__construct(Model $model, string $path) */
PHP_METHOD(Llama_LoRA, __construct)
{
    zval *model_zval;
    char *path;
    size_t path_len;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(model_zval, llama_ce_model)
        Z_PARAM_STRING(path, path_len)
    ZEND_PARSE_PARAMETERS_END();

    llama_lora_obj *intern = Z_LLAMA_LORA_P(ZEND_THIS);
    llama_model_obj *model_intern = Z_LLAMA_MODEL_P(model_zval);

    if (!model_intern->model) {
        zend_throw_exception(llama_ce_exception, "Model not loaded", 0);
        RETURN_THROWS();
    }

    if (VCWD_ACCESS(path, F_OK) != 0) {
        zend_throw_exception_ex(llama_ce_exception, 0, "LoRA file not found: %s", path);
        RETURN_THROWS();
    }

    struct llama_adapter_lora *adapter = llama_adapter_lora_init(model_intern->model, path);
    if (!adapter) {
        zend_throw_exception_ex(llama_ce_exception, 0, "Failed to load LoRA adapter: %s", path);
        RETURN_THROWS();
    }

    intern->adapter = adapter;
    intern->path = estrndup(path, path_len);
    ZVAL_COPY(&intern->model_zval, model_zval);
}

/* LoRA::meta(string $key): ?string */
PHP_METHOD(Llama_LoRA, meta)
{
    char *key;
    size_t key_len;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STRING(key, key_len)
    ZEND_PARSE_PARAMETERS_END();

    llama_lora_obj *intern = Z_LLAMA_LORA_P(ZEND_THIS);
    if (!intern->adapter) {
        zend_throw_exception(llama_ce_exception, "LoRA adapter not loaded", 0);
        RETURN_THROWS();
    }

    char buf[512];
    int32_t ret = llama_adapter_meta_val_str(intern->adapter, key, buf, sizeof(buf));
    if (ret >= 0) {
        RETURN_STRINGL(buf, ret);
    }
    RETURN_NULL();
}

/* ============================================================
 * Llama\Context
 * ============================================================ */

static zend_object *llama_context_create_object(zend_class_entry *ce)
{
    llama_context_obj *intern = zend_object_alloc(sizeof(llama_context_obj), ce);
    intern->ctx = NULL;
    intern->model = NULL;
    intern->vocab = NULL;
    ZVAL_UNDEF(&intern->model_zval);
    zend_object_std_init(&intern->std, ce);
    object_properties_init(&intern->std, ce);
    intern->std.handlers = &llama_context_handlers;
    return &intern->std;
}

static void llama_context_free_object(zend_object *obj)
{
    llama_context_obj *intern = llama_context_from_obj(obj);
    if (intern->ctx) {
        llama_free(intern->ctx);
        intern->ctx = NULL;
    }
    zval_ptr_dtor(&intern->model_zval);
    zend_object_std_dtor(&intern->std);
}

/* Context::__construct(Model $model, array $params = []) */
PHP_METHOD(Llama_Context, __construct)
{
    zval *model_zval;
    zval *params = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_OBJECT_OF_CLASS(model_zval, llama_ce_model)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY(params)
    ZEND_PARSE_PARAMETERS_END();

    llama_context_obj *intern = Z_LLAMA_CONTEXT_P(ZEND_THIS);
    llama_model_obj *model_intern = Z_LLAMA_MODEL_P(model_zval);

    if (!model_intern->model) {
        zend_throw_exception(llama_ce_exception, "Model not loaded", 0);
        RETURN_THROWS();
    }

    struct llama_context_params ctx_params = llama_context_default_params();

    if (params) {
        zval *val;
        if ((val = zend_hash_str_find(Z_ARRVAL_P(params), "n_ctx", sizeof("n_ctx") - 1)) != NULL) {
            ctx_params.n_ctx = (uint32_t)zval_get_long(val);
        }
        if ((val = zend_hash_str_find(Z_ARRVAL_P(params), "n_batch", sizeof("n_batch") - 1)) != NULL) {
            ctx_params.n_batch = (uint32_t)zval_get_long(val);
        }
        if ((val = zend_hash_str_find(Z_ARRVAL_P(params), "n_threads", sizeof("n_threads") - 1)) != NULL) {
            ctx_params.n_threads = (int32_t)zval_get_long(val);
            ctx_params.n_threads_batch = ctx_params.n_threads;
        }
        if ((val = zend_hash_str_find(Z_ARRVAL_P(params), "embeddings", sizeof("embeddings") - 1)) != NULL) {
            ctx_params.embeddings = zend_is_true(val);
        }
        if ((val = zend_hash_str_find(Z_ARRVAL_P(params), "flash_attn", sizeof("flash_attn") - 1)) != NULL) {
            ctx_params.flash_attn_type = zend_is_true(val) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
    }

    struct llama_context *ctx = llama_init_from_model(model_intern->model, ctx_params);
    if (!ctx) {
        zend_throw_exception(llama_ce_exception, "Failed to create context", 0);
        RETURN_THROWS();
    }

    intern->ctx = ctx;
    intern->model = model_intern->model;
    intern->vocab = model_intern->vocab;
    ZVAL_COPY(&intern->model_zval, model_zval);
}

/* Helper: build sampler chain from options.
 * If grammar/json_schema is specified, *grammar_out receives a separate grammar sampler
 * that must be applied BEFORE the chain. Caller must free it. */
static struct llama_sampler *build_sampler_chain(zval *options, const struct llama_vocab *vocab, struct llama_sampler **grammar_out)
{
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler *smpl = llama_sampler_chain_init(sparams);

    float temperature = 0.8f;
    int32_t top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.05f;
    float repeat_penalty = 1.1f;
    int32_t penalty_last_n = 64;
    uint32_t seed = LLAMA_DEFAULT_SEED;

    if (options) {
        zval *val;
        if ((val = zend_hash_str_find(Z_ARRVAL_P(options), "temperature", sizeof("temperature") - 1)) != NULL) {
            temperature = (float)zval_get_double(val);
        }
        if ((val = zend_hash_str_find(Z_ARRVAL_P(options), "top_k", sizeof("top_k") - 1)) != NULL) {
            top_k = (int32_t)zval_get_long(val);
        }
        if ((val = zend_hash_str_find(Z_ARRVAL_P(options), "top_p", sizeof("top_p") - 1)) != NULL) {
            top_p = (float)zval_get_double(val);
        }
        if ((val = zend_hash_str_find(Z_ARRVAL_P(options), "min_p", sizeof("min_p") - 1)) != NULL) {
            min_p = (float)zval_get_double(val);
        }
        if ((val = zend_hash_str_find(Z_ARRVAL_P(options), "repeat_penalty", sizeof("repeat_penalty") - 1)) != NULL) {
            repeat_penalty = (float)zval_get_double(val);
        }
        if ((val = zend_hash_str_find(Z_ARRVAL_P(options), "penalty_last_n", sizeof("penalty_last_n") - 1)) != NULL) {
            penalty_last_n = (int32_t)zval_get_long(val);
        }
        if ((val = zend_hash_str_find(Z_ARRVAL_P(options), "seed", sizeof("seed") - 1)) != NULL) {
            seed = (uint32_t)zval_get_long(val);
        }
    }

    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));

    if (repeat_penalty != 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(penalty_last_n, repeat_penalty, 0.0f, 0.0f));
    }

    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

    /* Grammar constraint — separate from chain, applied before it */
    *grammar_out = NULL;
    if (options) {
        zval *val;
        char *grammar_str = NULL;
        bool grammar_needs_free = false;

        if ((val = zend_hash_str_find(Z_ARRVAL_P(options), "json_schema", sizeof("json_schema") - 1)) != NULL) {
            zend_string *schema_zs = zval_get_string(val);
            char *err = NULL;
            grammar_str = llama_json_schema_to_grammar(ZSTR_VAL(schema_zs), &err);
            zend_string_release(schema_zs);
            if (!grammar_str) {
                llama_sampler_free(smpl);
                zend_throw_exception_ex(llama_ce_exception, 0,
                    "Failed to convert JSON schema to grammar: %s", err ? err : "unknown error");
                free(err);
                return NULL;
            }
            grammar_needs_free = true;
        } else if ((val = zend_hash_str_find(Z_ARRVAL_P(options), "grammar", sizeof("grammar") - 1)) != NULL) {
            convert_to_string(val);
            grammar_str = Z_STRVAL_P(val);
        }

        if (grammar_str) {
            *grammar_out = llama_sampler_init_grammar(vocab, grammar_str, "root");
            if (grammar_needs_free) {
                free(grammar_str);
            }
            if (!*grammar_out) {
                llama_sampler_free(smpl);
                zend_throw_exception(llama_ce_exception, "Failed to create grammar sampler", 0);
                return NULL;
            }
        }
    }

    return smpl;
}

/* Context::complete(string $prompt, array $options = []): string */
PHP_METHOD(Llama_Context, complete)
{
    char *prompt;
    size_t prompt_len;
    zval *options = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_STRING(prompt, prompt_len)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY(options)
    ZEND_PARSE_PARAMETERS_END();

    llama_context_obj *intern = Z_LLAMA_CONTEXT_P(ZEND_THIS);
    if (!intern->ctx) {
        zend_throw_exception(llama_ce_exception, "Context not initialized", 0);
        RETURN_THROWS();
    }

    int32_t max_tokens = 256;
    if (options) {
        zval *val;
        if ((val = zend_hash_str_find(Z_ARRVAL_P(options), "max_tokens", sizeof("max_tokens") - 1)) != NULL) {
            max_tokens = (int32_t)zval_get_long(val);
        }
    }

    /* Tokenize prompt */
    int32_t n_prompt = llama_tokenize(intern->vocab, prompt, (int32_t)prompt_len, NULL, 0, true, false);
    if (n_prompt < 0) {
        n_prompt = -n_prompt;
    }

    llama_token *prompt_tokens = emalloc(sizeof(llama_token) * n_prompt);
    llama_tokenize(intern->vocab, prompt, (int32_t)prompt_len, prompt_tokens, n_prompt, true, false);

    /* Clear KV cache */
    llama_memory_clear(llama_get_memory(intern->ctx), false);

    /* Decode prompt using batch_get_one */
    struct llama_batch batch = llama_batch_get_one(prompt_tokens, n_prompt);
    if (llama_decode(intern->ctx, batch) != 0) {
        efree(prompt_tokens);
        zend_throw_exception(llama_ce_exception, "Failed to decode prompt", 0);
        RETURN_THROWS();
    }

    /* Build sampler */
    struct llama_sampler *grammar = NULL;
    struct llama_sampler *smpl = build_sampler_chain(options, intern->vocab, &grammar);
    if (!smpl) {
        efree(prompt_tokens);
        RETURN_THROWS();
    }

    /* Generate tokens */
    smart_str result = {0};
    char piece_buf[128];
    int32_t n_cur = n_prompt;

    for (int32_t i = 0; i < max_tokens; i++) {
        llama_token new_token = llama_sampler_sample_safe(smpl, intern->ctx, -1, grammar);

        /* Check for end of generation */
        if (new_token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(intern->vocab, new_token)) {
            break;
        }

        /* Convert token to text */
        int32_t piece_len = llama_token_to_piece(intern->vocab, new_token, piece_buf, sizeof(piece_buf), 0, false);
        if (piece_len > 0) {
            smart_str_appendl(&result, piece_buf, piece_len);
        }

        /* Prepare next batch */
        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(intern->ctx, batch) != 0) {
            break;
        }
        n_cur++;
    }

    if (grammar) llama_sampler_free(grammar);
    llama_sampler_free(smpl);
    efree(prompt_tokens);

    smart_str_0(&result);
    if (result.s) {
        RETVAL_STR(result.s);
    } else {
        RETVAL_EMPTY_STRING();
    }
}

/* Context::chat(array $messages, array $options = []): string */
PHP_METHOD(Llama_Context, chat)
{
    zval *messages;
    zval *options = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY(messages)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY(options)
    ZEND_PARSE_PARAMETERS_END();

    llama_context_obj *intern = Z_LLAMA_CONTEXT_P(ZEND_THIS);
    if (!intern->ctx) {
        zend_throw_exception(llama_ce_exception, "Context not initialized", 0);
        RETURN_THROWS();
    }

    /* Get chat template from model */
    const char *tmpl = llama_model_chat_template(intern->model, NULL);
    if (!tmpl) {
        zend_throw_exception(llama_ce_exception, "Model does not have a chat template", 0);
        RETURN_THROWS();
    }

    /* Build llama_chat_message array */
    int32_t n_msg = zend_hash_num_elements(Z_ARRVAL_P(messages));
    struct llama_chat_message *chat_msgs = emalloc(sizeof(struct llama_chat_message) * n_msg);

    int32_t idx = 0;
    zval *msg;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(messages), msg) {
        if (Z_TYPE_P(msg) != IS_ARRAY) {
            efree(chat_msgs);
            zend_throw_exception(llama_ce_exception, "Each message must be an array with 'role' and 'content'", 0);
            RETURN_THROWS();
        }
        zval *role = zend_hash_str_find(Z_ARRVAL_P(msg), "role", sizeof("role") - 1);
        zval *content = zend_hash_str_find(Z_ARRVAL_P(msg), "content", sizeof("content") - 1);
        if (!role || !content) {
            efree(chat_msgs);
            zend_throw_exception(llama_ce_exception, "Each message must have 'role' and 'content'", 0);
            RETURN_THROWS();
        }
        convert_to_string(role);
        convert_to_string(content);
        chat_msgs[idx].role = Z_STRVAL_P(role);
        chat_msgs[idx].content = Z_STRVAL_P(content);
        idx++;
    } ZEND_HASH_FOREACH_END();

    /* Apply chat template */
    int32_t buf_size = 4096;
    char *buf = emalloc(buf_size);
    int32_t needed = llama_chat_apply_template(tmpl, chat_msgs, n_msg, true, buf, buf_size);

    if (needed > buf_size) {
        buf_size = needed + 1;
        buf = erealloc(buf, buf_size);
        llama_chat_apply_template(tmpl, chat_msgs, n_msg, true, buf, buf_size);
    }

    efree(chat_msgs);

    /* Now call complete with the formatted prompt */
    /* Tokenize */
    int32_t n_prompt = llama_tokenize(intern->vocab, buf, needed, NULL, 0, false, true);
    if (n_prompt < 0) {
        n_prompt = -n_prompt;
    }

    llama_token *prompt_tokens = emalloc(sizeof(llama_token) * n_prompt);
    llama_tokenize(intern->vocab, buf, needed, prompt_tokens, n_prompt, false, true);
    efree(buf);

    int32_t max_tokens = 256;
    if (options) {
        zval *val;
        if ((val = zend_hash_str_find(Z_ARRVAL_P(options), "max_tokens", sizeof("max_tokens") - 1)) != NULL) {
            max_tokens = (int32_t)zval_get_long(val);
        }
    }

    /* Clear KV cache */
    llama_memory_clear(llama_get_memory(intern->ctx), false);

    /* Decode prompt */
    struct llama_batch batch = llama_batch_get_one(prompt_tokens, n_prompt);
    if (llama_decode(intern->ctx, batch) != 0) {
        efree(prompt_tokens);
        zend_throw_exception(llama_ce_exception, "Failed to decode chat prompt", 0);
        RETURN_THROWS();
    }

    /* Build sampler */
    struct llama_sampler *grammar = NULL;
    struct llama_sampler *smpl = build_sampler_chain(options, intern->vocab, &grammar);
    if (!smpl) {
        efree(prompt_tokens);
        RETURN_THROWS();
    }

    /* Generate */
    smart_str result = {0};
    char piece_buf[128];

    for (int32_t i = 0; i < max_tokens; i++) {
        llama_token new_token = llama_sampler_sample_safe(smpl, intern->ctx, -1, grammar);

        if (new_token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(intern->vocab, new_token)) {
            break;
        }

        int32_t piece_len = llama_token_to_piece(intern->vocab, new_token, piece_buf, sizeof(piece_buf), 0, false);
        if (piece_len > 0) {
            smart_str_appendl(&result, piece_buf, piece_len);
        }

        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(intern->ctx, batch) != 0) {
            break;
        }
    }

    if (grammar) llama_sampler_free(grammar);
    llama_sampler_free(smpl);
    efree(prompt_tokens);

    smart_str_0(&result);
    if (result.s) {
        RETVAL_STR(result.s);
    } else {
        RETVAL_EMPTY_STRING();
    }
}

/* Context::embed(string $text): array */
PHP_METHOD(Llama_Context, embed)
{
    char *text;
    size_t text_len;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STRING(text, text_len)
    ZEND_PARSE_PARAMETERS_END();

    llama_context_obj *intern = Z_LLAMA_CONTEXT_P(ZEND_THIS);
    if (!intern->ctx) {
        zend_throw_exception(llama_ce_exception, "Context not initialized", 0);
        RETURN_THROWS();
    }

    /* Tokenize */
    int32_t n_tokens = llama_tokenize(intern->vocab, text, (int32_t)text_len, NULL, 0, true, false);
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
    }

    llama_token *tokens = emalloc(sizeof(llama_token) * n_tokens);
    llama_tokenize(intern->vocab, text, (int32_t)text_len, tokens, n_tokens, true, false);

    /* Clear KV cache */
    llama_memory_clear(llama_get_memory(intern->ctx), false);

    /* Decode */
    struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    if (llama_decode(intern->ctx, batch) != 0) {
        efree(tokens);
        zend_throw_exception(llama_ce_exception, "Failed to decode for embeddings", 0);
        RETURN_THROWS();
    }

    efree(tokens);

    /* Get embeddings */
    float *embd = llama_get_embeddings_seq(intern->ctx, 0);
    if (!embd) {
        /* Try ith approach */
        embd = llama_get_embeddings_ith(intern->ctx, -1);
    }

    if (!embd) {
        zend_throw_exception(llama_ce_exception, "Failed to get embeddings. Was the context created with 'embeddings' => true?", 0);
        RETURN_THROWS();
    }

    int32_t n_embd = llama_model_n_embd(intern->model);
    array_init_size(return_value, n_embd);

    for (int32_t i = 0; i < n_embd; i++) {
        add_next_index_double(return_value, (double)embd[i]);
    }
}

/* Context::applyLoRA(LoRA $lora, float $scale = 1.0): void
 * Context::applyLoRA(array<LoRA> $loras, array<float> $scales): void */
PHP_METHOD(Llama_Context, applyLoRA)
{
    zval *arg;
    double scale = 1.0;
    zval *scales_arr = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(arg)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(scales_arr)
    ZEND_PARSE_PARAMETERS_END();

    llama_context_obj *intern = Z_LLAMA_CONTEXT_P(ZEND_THIS);
    if (!intern->ctx) {
        zend_throw_exception(llama_ce_exception, "Context not initialized", 0);
        RETURN_THROWS();
    }

    if (Z_TYPE_P(arg) == IS_OBJECT && instanceof_function(Z_OBJCE_P(arg), llama_ce_lora)) {
        /* Single LoRA */
        if (scales_arr && Z_TYPE_P(scales_arr) == IS_DOUBLE) {
            scale = Z_DVAL_P(scales_arr);
        } else if (scales_arr && Z_TYPE_P(scales_arr) == IS_LONG) {
            scale = (double)Z_LVAL_P(scales_arr);
        }
        llama_lora_obj *lora = Z_LLAMA_LORA_P(arg);
        if (!lora->adapter) {
            zend_throw_exception(llama_ce_exception, "LoRA adapter not loaded", 0);
            RETURN_THROWS();
        }
        float fscale = (float)scale;
        struct llama_adapter_lora *adapters[1] = { lora->adapter };
        int32_t ret = llama_set_adapters_lora(intern->ctx, adapters, 1, &fscale);
        if (ret != 0) {
            zend_throw_exception(llama_ce_exception, "Failed to apply LoRA adapter", 0);
            RETURN_THROWS();
        }
    } else if (Z_TYPE_P(arg) == IS_ARRAY) {
        /* Array of LoRAs */
        uint32_t n = zend_hash_num_elements(Z_ARRVAL_P(arg));
        struct llama_adapter_lora **adapters = emalloc(sizeof(struct llama_adapter_lora *) * n);
        float *scales = emalloc(sizeof(float) * n);

        uint32_t i = 0;
        zval *item;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(arg), item) {
            if (Z_TYPE_P(item) != IS_OBJECT || !instanceof_function(Z_OBJCE_P(item), llama_ce_lora)) {
                efree(adapters);
                efree(scales);
                zend_throw_exception(llama_ce_exception, "Array must contain Llama\\LoRA objects", 0);
                RETURN_THROWS();
            }
            llama_lora_obj *lora = Z_LLAMA_LORA_P(item);
            if (!lora->adapter) {
                efree(adapters);
                efree(scales);
                zend_throw_exception(llama_ce_exception, "LoRA adapter not loaded", 0);
                RETURN_THROWS();
            }
            adapters[i] = lora->adapter;
            scales[i] = 1.0f;
            i++;
        } ZEND_HASH_FOREACH_END();

        /* Apply per-adapter scales if provided */
        if (scales_arr && Z_TYPE_P(scales_arr) == IS_ARRAY) {
            uint32_t j = 0;
            zval *sv;
            ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(scales_arr), sv) {
                if (j < n) {
                    scales[j] = (float)zval_get_double(sv);
                }
                j++;
            } ZEND_HASH_FOREACH_END();
        }

        int32_t ret = llama_set_adapters_lora(intern->ctx, adapters, n, scales);
        efree(adapters);
        efree(scales);
        if (ret != 0) {
            zend_throw_exception(llama_ce_exception, "Failed to apply LoRA adapters", 0);
            RETURN_THROWS();
        }
    } else {
        zend_throw_exception(llama_ce_exception, "Argument must be a Llama\\LoRA or array of Llama\\LoRA", 0);
        RETURN_THROWS();
    }
}

/* Context::clearLoRA(): void */
PHP_METHOD(Llama_Context, clearLoRA)
{
    ZEND_PARSE_PARAMETERS_NONE();

    llama_context_obj *intern = Z_LLAMA_CONTEXT_P(ZEND_THIS);
    if (!intern->ctx) {
        zend_throw_exception(llama_ce_exception, "Context not initialized", 0);
        RETURN_THROWS();
    }

    llama_set_adapters_lora(intern->ctx, NULL, 0, NULL);
}

/* ============================================================
 * Llama\CompletionIterator (implements Iterator)
 * ============================================================ */

static zend_object_handlers llama_completion_iter_handlers;

static zend_object *llama_completion_iter_create_object(zend_class_entry *ce)
{
    llama_completion_iter_obj *intern = zend_object_alloc(sizeof(llama_completion_iter_obj), ce);
    intern->ctx = NULL;
    intern->vocab = NULL;
    intern->smpl = NULL;
    intern->grammar = NULL;
    intern->max_tokens = 0;
    intern->n_generated = 0;
    intern->finished = true;
    ZVAL_UNDEF(&intern->current);
    ZVAL_UNDEF(&intern->context_zval);
    zend_object_std_init(&intern->std, ce);
    object_properties_init(&intern->std, ce);
    intern->std.handlers = &llama_completion_iter_handlers;
    return &intern->std;
}

static void llama_completion_iter_free_object(zend_object *obj)
{
    llama_completion_iter_obj *intern = llama_completion_iter_from_obj(obj);
    if (intern->grammar) {
        llama_sampler_free(intern->grammar);
        intern->grammar = NULL;
    }
    if (intern->smpl) {
        llama_sampler_free(intern->smpl);
        intern->smpl = NULL;
    }
    zval_ptr_dtor(&intern->current);
    zval_ptr_dtor(&intern->context_zval);
    zend_object_std_dtor(&intern->std);
}

/* Advance one token — shared by rewind (first token) and next */
static void llama_completion_iter_advance(llama_completion_iter_obj *iter)
{
    zval_ptr_dtor(&iter->current);
    ZVAL_UNDEF(&iter->current);

    if (iter->finished || iter->n_generated >= iter->max_tokens) {
        iter->finished = true;
        return;
    }

    llama_token new_token = llama_sampler_sample_safe(iter->smpl, iter->ctx, -1, iter->grammar);

    if (new_token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(iter->vocab, new_token)) {
        iter->finished = true;
        return;
    }

    char piece_buf[128];
    int32_t piece_len = llama_token_to_piece(iter->vocab, new_token, piece_buf, sizeof(piece_buf), 0, false);
    if (piece_len > 0) {
        ZVAL_STRINGL(&iter->current, piece_buf, piece_len);
    } else {
        ZVAL_EMPTY_STRING(&iter->current);
    }

    /* Decode this token for the next step */
    struct llama_batch batch = llama_batch_get_one(&new_token, 1);
    if (llama_decode(iter->ctx, batch) != 0) {
        iter->finished = true;
        return;
    }

    iter->n_generated++;
}

/* Iterator::rewind() */
PHP_METHOD(Llama_CompletionIterator, rewind)
{
    ZEND_PARSE_PARAMETERS_NONE();
    /* First token was already generated during stream() setup — nothing to do */
}

/* Iterator::valid(): bool */
PHP_METHOD(Llama_CompletionIterator, valid)
{
    ZEND_PARSE_PARAMETERS_NONE();
    llama_completion_iter_obj *intern = Z_LLAMA_COMPLETION_ITER_P(ZEND_THIS);
    RETURN_BOOL(!intern->finished);
}

/* Iterator::current(): string */
PHP_METHOD(Llama_CompletionIterator, current)
{
    ZEND_PARSE_PARAMETERS_NONE();
    llama_completion_iter_obj *intern = Z_LLAMA_COMPLETION_ITER_P(ZEND_THIS);
    if (Z_TYPE(intern->current) == IS_UNDEF) {
        RETURN_EMPTY_STRING();
    }
    RETURN_COPY(&intern->current);
}

/* Iterator::key(): int */
PHP_METHOD(Llama_CompletionIterator, key)
{
    ZEND_PARSE_PARAMETERS_NONE();
    llama_completion_iter_obj *intern = Z_LLAMA_COMPLETION_ITER_P(ZEND_THIS);
    RETURN_LONG(intern->n_generated - 1);
}

/* Iterator::next() */
PHP_METHOD(Llama_CompletionIterator, next)
{
    ZEND_PARSE_PARAMETERS_NONE();
    llama_completion_iter_obj *intern = Z_LLAMA_COMPLETION_ITER_P(ZEND_THIS);
    llama_completion_iter_advance(intern);
}

/* Context::stream(string $prompt, array $options = []): Llama\CompletionIterator */
PHP_METHOD(Llama_Context, stream)
{
    char *prompt;
    size_t prompt_len;
    zval *options = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_STRING(prompt, prompt_len)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY(options)
    ZEND_PARSE_PARAMETERS_END();

    llama_context_obj *ctx_intern = Z_LLAMA_CONTEXT_P(ZEND_THIS);
    if (!ctx_intern->ctx) {
        zend_throw_exception(llama_ce_exception, "Context not initialized", 0);
        RETURN_THROWS();
    }

    int32_t max_tokens = 256;
    if (options) {
        zval *val;
        if ((val = zend_hash_str_find(Z_ARRVAL_P(options), "max_tokens", sizeof("max_tokens") - 1)) != NULL) {
            max_tokens = (int32_t)zval_get_long(val);
        }
    }

    /* Tokenize prompt */
    int32_t n_prompt = llama_tokenize(ctx_intern->vocab, prompt, (int32_t)prompt_len, NULL, 0, true, false);
    if (n_prompt < 0) {
        n_prompt = -n_prompt;
    }

    llama_token *prompt_tokens = emalloc(sizeof(llama_token) * n_prompt);
    llama_tokenize(ctx_intern->vocab, prompt, (int32_t)prompt_len, prompt_tokens, n_prompt, true, false);

    /* Clear KV cache */
    llama_memory_clear(llama_get_memory(ctx_intern->ctx), false);

    /* Decode prompt */
    struct llama_batch batch = llama_batch_get_one(prompt_tokens, n_prompt);
    if (llama_decode(ctx_intern->ctx, batch) != 0) {
        efree(prompt_tokens);
        zend_throw_exception(llama_ce_exception, "Failed to decode prompt", 0);
        RETURN_THROWS();
    }
    efree(prompt_tokens);

    /* Create the iterator object */
    object_init_ex(return_value, llama_ce_completion_iterator);
    llama_completion_iter_obj *iter = Z_LLAMA_COMPLETION_ITER_P(return_value);

    iter->ctx = ctx_intern->ctx;
    iter->vocab = ctx_intern->vocab;
    iter->smpl = build_sampler_chain(options, ctx_intern->vocab, &iter->grammar);
    if (!iter->smpl) {
        zval_ptr_dtor(return_value);
        RETURN_THROWS();
    }
    iter->max_tokens = max_tokens;
    iter->n_generated = 0;
    iter->finished = false;
    ZVAL_COPY(&iter->context_zval, ZEND_THIS);

    /* Generate the first token so valid() is true immediately */
    llama_completion_iter_advance(iter);
}

/* ============================================================
 * Arginfo / Method entries
 * ============================================================ */

ZEND_BEGIN_ARG_INFO_EX(arginfo_llama_model___construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, modelPath, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, params, IS_ARRAY, 0, "[]")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_model_desc, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_model_size, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

#define arginfo_llama_model_nParams arginfo_llama_model_size
#define arginfo_llama_model_nEmbd arginfo_llama_model_size
#define arginfo_llama_model_nLayer arginfo_llama_model_size

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_model_chatTemplate, 0, 0, IS_STRING, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_model_meta, 0, 1, IS_STRING, 1)
    ZEND_ARG_TYPE_INFO(0, key, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_model_tokenize, 0, 1, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, text, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, addSpecial, _IS_BOOL, 0, "true")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, parseSpecial, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_model_detokenize, 0, 1, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, tokens, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_llama_context___construct, 0, 0, 1)
    ZEND_ARG_OBJ_INFO(0, model, Llama\\Model, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, params, IS_ARRAY, 0, "[]")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_context_complete, 0, 1, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, prompt, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, options, IS_ARRAY, 0, "[]")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_context_chat, 0, 1, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, messages, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, options, IS_ARRAY, 0, "[]")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_llama_context_stream, 0, 1, Llama\\CompletionIterator, 0)
    ZEND_ARG_TYPE_INFO(0, prompt, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, options, IS_ARRAY, 0, "[]")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_context_embed, 0, 1, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, text, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_context_applyLoRA, 0, 1, IS_VOID, 0)
    ZEND_ARG_INFO(0, loraOrArray)
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, scaleOrScales, "1.0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_context_clearLoRA, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

/* LoRA arginfo */
ZEND_BEGIN_ARG_INFO_EX(arginfo_llama_lora___construct, 0, 0, 2)
    ZEND_ARG_OBJ_INFO(0, model, Llama\\Model, 0)
    ZEND_ARG_TYPE_INFO(0, path, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_lora_meta, 0, 1, IS_STRING, 1)
    ZEND_ARG_TYPE_INFO(0, key, IS_STRING, 0)
ZEND_END_ARG_INFO()

/* CompletionIterator arginfo */
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_iter_void, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_iter_valid, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_iter_current, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_iter_key, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

static const zend_function_entry llama_model_methods[] = {
    PHP_ME(Llama_Model, __construct,  arginfo_llama_model___construct,  ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Model, desc,         arginfo_llama_model_desc,         ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Model, size,         arginfo_llama_model_size,         ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Model, nParams,      arginfo_llama_model_nParams,      ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Model, nEmbd,        arginfo_llama_model_nEmbd,        ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Model, nLayer,       arginfo_llama_model_nLayer,       ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Model, chatTemplate, arginfo_llama_model_chatTemplate, ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Model, meta,         arginfo_llama_model_meta,         ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Model, tokenize,     arginfo_llama_model_tokenize,     ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Model, detokenize,   arginfo_llama_model_detokenize,   ZEND_ACC_PUBLIC)
    PHP_FE_END
};

static const zend_function_entry llama_lora_methods[] = {
    PHP_ME(Llama_LoRA, __construct, arginfo_llama_lora___construct, ZEND_ACC_PUBLIC)
    PHP_ME(Llama_LoRA, meta,        arginfo_llama_lora_meta,        ZEND_ACC_PUBLIC)
    PHP_FE_END
};

static const zend_function_entry llama_context_methods[] = {
    PHP_ME(Llama_Context, __construct, arginfo_llama_context___construct, ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Context, complete,    arginfo_llama_context_complete,    ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Context, stream,      arginfo_llama_context_stream,      ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Context, chat,        arginfo_llama_context_chat,        ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Context, embed,       arginfo_llama_context_embed,       ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Context, applyLoRA,   arginfo_llama_context_applyLoRA,   ZEND_ACC_PUBLIC)
    PHP_ME(Llama_Context, clearLoRA,   arginfo_llama_context_clearLoRA,   ZEND_ACC_PUBLIC)
    PHP_FE_END
};

static const zend_function_entry llama_completion_iter_methods[] = {
    PHP_ME(Llama_CompletionIterator, rewind,  arginfo_llama_iter_void,    ZEND_ACC_PUBLIC)
    PHP_ME(Llama_CompletionIterator, valid,   arginfo_llama_iter_valid,   ZEND_ACC_PUBLIC)
    PHP_ME(Llama_CompletionIterator, current, arginfo_llama_iter_current, ZEND_ACC_PUBLIC)
    PHP_ME(Llama_CompletionIterator, key,     arginfo_llama_iter_key,     ZEND_ACC_PUBLIC)
    PHP_ME(Llama_CompletionIterator, next,    arginfo_llama_iter_void,    ZEND_ACC_PUBLIC)
    PHP_FE_END
};

/* ============================================================
 * Module lifecycle
 * ============================================================ */

PHP_MINIT_FUNCTION(llama)
{
    /* Initialize llama backend */
    if (!llama_backend_initialized) {
        llama_log_set(llama_log_callback_null, NULL);
        llama_backend_init();
        zend_hash_init(&persistent_models, 4, NULL, NULL, 1); /* persistent=1 */
        llama_backend_initialized = true;
    }

    /* Register Llama\Exception */
    {
        zend_class_entry tmp_ce;
        INIT_NS_CLASS_ENTRY(tmp_ce, "Llama", "Exception", NULL);
        llama_ce_exception = zend_register_internal_class_ex(&tmp_ce, zend_ce_exception);
    }

    /* Register Llama\Model */
    zend_class_entry ce;
    INIT_NS_CLASS_ENTRY(ce, "Llama", "Model", llama_model_methods);
    llama_ce_model = zend_register_internal_class(&ce);
    llama_ce_model->ce_flags |= ZEND_ACC_NO_DYNAMIC_PROPERTIES;
    llama_ce_model->create_object = llama_model_create_object;

    memcpy(&llama_model_handlers, &std_object_handlers, sizeof(zend_object_handlers));
    llama_model_handlers.offset = XtOffsetOf(llama_model_obj, std);
    llama_model_handlers.free_obj = llama_model_free_object;
    llama_model_handlers.clone_obj = NULL;

    /* Register Llama\LoRA */
    INIT_NS_CLASS_ENTRY(ce, "Llama", "LoRA", llama_lora_methods);
    llama_ce_lora = zend_register_internal_class(&ce);
    llama_ce_lora->ce_flags |= ZEND_ACC_NO_DYNAMIC_PROPERTIES;
    llama_ce_lora->create_object = llama_lora_create_object;

    memcpy(&llama_lora_handlers, &std_object_handlers, sizeof(zend_object_handlers));
    llama_lora_handlers.offset = XtOffsetOf(llama_lora_obj, std);
    llama_lora_handlers.free_obj = llama_lora_free_object;
    llama_lora_handlers.clone_obj = NULL;

    /* Register Llama\Context */
    INIT_NS_CLASS_ENTRY(ce, "Llama", "Context", llama_context_methods);
    llama_ce_context = zend_register_internal_class(&ce);
    llama_ce_context->ce_flags |= ZEND_ACC_NO_DYNAMIC_PROPERTIES;
    llama_ce_context->create_object = llama_context_create_object;

    memcpy(&llama_context_handlers, &std_object_handlers, sizeof(zend_object_handlers));
    llama_context_handlers.offset = XtOffsetOf(llama_context_obj, std);
    llama_context_handlers.free_obj = llama_context_free_object;
    llama_context_handlers.clone_obj = NULL;

    /* Register Llama\CompletionIterator implementing Iterator */
    INIT_NS_CLASS_ENTRY(ce, "Llama", "CompletionIterator", llama_completion_iter_methods);
    llama_ce_completion_iterator = zend_register_internal_class(&ce);
    llama_ce_completion_iterator->ce_flags |= ZEND_ACC_NO_DYNAMIC_PROPERTIES | ZEND_ACC_FINAL;
    llama_ce_completion_iterator->create_object = llama_completion_iter_create_object;
    zend_class_implements(llama_ce_completion_iterator, 1, zend_ce_iterator);

    memcpy(&llama_completion_iter_handlers, &std_object_handlers, sizeof(zend_object_handlers));
    llama_completion_iter_handlers.offset = XtOffsetOf(llama_completion_iter_obj, std);
    llama_completion_iter_handlers.free_obj = llama_completion_iter_free_object;
    llama_completion_iter_handlers.clone_obj = NULL;

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(llama)
{
    if (llama_backend_initialized) {
        /* Free all persistent models */
        llama_persistent_model *entry;
        ZEND_HASH_FOREACH_PTR(&persistent_models, entry) {
            if (entry->model) {
                llama_model_free(entry->model);
            }
        } ZEND_HASH_FOREACH_END();
        zend_hash_destroy(&persistent_models);

        llama_backend_free();
        llama_backend_initialized = false;
    }
    return SUCCESS;
}

PHP_MINFO_FUNCTION(llama)
{
    char buf[32];

    php_info_print_table_start();
    php_info_print_table_row(2, "llama.cpp support", "enabled");
    php_info_print_table_row(2, "Extension version", PHP_LLAMA_VERSION);

    snprintf(buf, sizeof(buf), "%u", zend_hash_num_elements(&persistent_models));
    php_info_print_table_row(2, "Persistent models loaded", buf);

    /* List cached model paths */
    zend_string *key;
    ZEND_HASH_FOREACH_STR_KEY(&persistent_models, key) {
        php_info_print_table_row(2, "  Cached model", ZSTR_VAL(key));
    } ZEND_HASH_FOREACH_END();

    php_info_print_table_end();
}

/* Module entry */
zend_module_entry llama_module_entry = {
    STANDARD_MODULE_HEADER,
    "llama",
    NULL, /* functions */
    PHP_MINIT(llama),
    PHP_MSHUTDOWN(llama),
    NULL, /* RINIT */
    NULL, /* RSHUTDOWN */
    PHP_MINFO(llama),
    PHP_LLAMA_VERSION,
    STANDARD_MODULE_PROPERTIES
};

#ifdef COMPILE_DL_LLAMA
ZEND_GET_MODULE(llama)
#endif
