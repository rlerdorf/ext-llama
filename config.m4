dnl config.m4 for extension llama

PHP_ARG_WITH([llama],
  [for llama.cpp support],
  [AS_HELP_STRING([--with-llama],
    [Include llama.cpp support])])

if test "$PHP_LLAMA" != "no"; then
  dnl Check for llama.h
  SEARCH_PATH="/usr/local /usr /home/rasmus/src/llama-cpp"

  SEARCH_FOR="include/llama.h"
  AC_MSG_CHECKING([for llama.cpp headers])
  for i in $PHP_LLAMA $SEARCH_PATH; do
    if test -r $i/$SEARCH_FOR; then
      LLAMA_DIR=$i
      AC_MSG_RESULT(found in $i)
      break
    fi
  done

  if test -z "$LLAMA_DIR"; then
    AC_MSG_RESULT([not found])
    AC_MSG_ERROR([Please install llama.cpp or specify the path with --with-llama=DIR])
  fi

  PHP_ADD_INCLUDE($LLAMA_DIR/include)
  PHP_ADD_INCLUDE($LLAMA_DIR/ggml/include)
  PHP_ADD_INCLUDE($LLAMA_DIR/common)

  dnl Check for libllama
  LIBLLAMA_SEARCH="$LLAMA_DIR/build/bin $LLAMA_DIR/lib $LLAMA_DIR/build/src"
  for dir in $LIBLLAMA_SEARCH; do
    if test -f "$dir/libllama.so" || test -f "$dir/libllama.so.0"; then
      PHP_ADD_LIBPATH($dir, LLAMA_SHARED_LIBADD)
      break
    fi
  done

  PHP_ADD_LIBRARY(llama, 1, LLAMA_SHARED_LIBADD)
  PHP_ADD_LIBRARY(stdc++, 1, LLAMA_SHARED_LIBADD)
  PHP_SUBST(LLAMA_SHARED_LIBADD)

  dnl Find nlohmann/json include path
  for dir in "$LLAMA_DIR/vendor" "$LLAMA_DIR/common" "$LLAMA_DIR/build/_deps/json-src/include" "/usr/include" "/usr/local/include"; do
    if test -f "$dir/nlohmann/json.hpp"; then
      PHP_ADD_INCLUDE($dir)
      break
    fi
  done

  dnl Static link libcommon for json-schema-to-grammar
  LLAMA_COMMON_LIB="$LLAMA_DIR/build/common/libcommon.a"
  if test -f "$LLAMA_COMMON_LIB"; then
    LLAMA_SHARED_LIBADD="$LLAMA_SHARED_LIBADD $LLAMA_COMMON_LIB"
  fi

  PHP_REQUIRE_CXX()
  PHP_NEW_EXTENSION(llama, llama.c json_schema_shim.cpp, $ext_shared,, -DLLAMA_SHARED)
fi
