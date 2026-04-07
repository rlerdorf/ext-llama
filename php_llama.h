#ifndef PHP_LLAMA_H
#define PHP_LLAMA_H

#include "php.h"
#include "llama.h"

#define PHP_LLAMA_VERSION "0.1.1"

extern zend_module_entry llama_module_entry;
#define phpext_llama_ptr &llama_module_entry

/* Class entries */
extern zend_class_entry *llama_ce_model;
extern zend_class_entry *llama_ce_context;
extern zend_class_entry *llama_ce_lora;
extern zend_class_entry *llama_ce_completion_iterator;
extern zend_class_entry *llama_ce_exception;

/* Object structs */
typedef struct {
    struct llama_model *model;
    const struct llama_vocab *vocab;
    char *path;
    bool persistent; /* if true, model is in the persistent cache and must not be freed here */
    zend_object std;
} llama_model_obj;

typedef struct {
    struct llama_context *ctx;
    struct llama_model *model; /* borrowed ref, not owned */
    const struct llama_vocab *vocab;
    zval model_zval; /* prevent GC of model while context alive */
    zend_object std;
} llama_context_obj;

typedef struct {
    struct llama_adapter_lora *adapter;
    char *path;
    zval model_zval; /* prevent GC of model */
    zend_object std;
} llama_lora_obj;

static inline llama_lora_obj *llama_lora_from_obj(zend_object *obj) {
    return (llama_lora_obj *)((char *)obj - XtOffsetOf(llama_lora_obj, std));
}

#define Z_LLAMA_LORA_P(zv) llama_lora_from_obj(Z_OBJ_P(zv))

typedef struct {
    struct llama_context *ctx;   /* borrowed, not owned */
    const struct llama_vocab *vocab;
    struct llama_sampler *smpl;  /* owned — freed on dtor */
    struct llama_sampler *grammar; /* owned, may be NULL */
    int32_t max_tokens;
    int32_t n_generated;
    bool finished;
    zval current;               /* current token piece (string) */
    zval context_zval;          /* prevent GC of Context while iterating */
    zend_object std;
} llama_completion_iter_obj;

static inline llama_completion_iter_obj *llama_completion_iter_from_obj(zend_object *obj) {
    return (llama_completion_iter_obj *)((char *)obj - XtOffsetOf(llama_completion_iter_obj, std));
}

#define Z_LLAMA_COMPLETION_ITER_P(zv) llama_completion_iter_from_obj(Z_OBJ_P(zv))

/* Fetch internal object from zend_object */
static inline llama_model_obj *llama_model_from_obj(zend_object *obj) {
    return (llama_model_obj *)((char *)obj - XtOffsetOf(llama_model_obj, std));
}

static inline llama_context_obj *llama_context_from_obj(zend_object *obj) {
    return (llama_context_obj *)((char *)obj - XtOffsetOf(llama_context_obj, std));
}

#define Z_LLAMA_MODEL_P(zv) llama_model_from_obj(Z_OBJ_P(zv))
#define Z_LLAMA_CONTEXT_P(zv) llama_context_from_obj(Z_OBJ_P(zv))

PHP_MINIT_FUNCTION(llama);
PHP_MSHUTDOWN_FUNCTION(llama);
PHP_MINFO_FUNCTION(llama);

#endif /* PHP_LLAMA_H */
