/*
 * Thin C wrapper around llama.cpp's json_schema_to_grammar().
 * Compiled as C++ so it can call the C++ common library, but
 * exposes a C-linkage function for the PHP extension.
 *
 * Only built when libcommon.a is available (HAVE_JSON_SCHEMA).
 */

#include "json-schema-to-grammar.h"
#include <nlohmann/json.hpp>
#include <string>
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

} /* extern "C" */
