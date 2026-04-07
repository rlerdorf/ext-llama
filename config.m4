dnl config.m4 for extension llama

PHP_ARG_WITH([llama],
  [for llama.cpp support],
  [AS_HELP_STRING([--with-llama],
    [Include llama.cpp support])])

if test "$PHP_LLAMA" != "no"; then
  dnl Find llama.h - check both source tree layout (include/llama.h)
  dnl and system-installed layout (llama.h directly in include path)
  AC_MSG_CHECKING([for llama.cpp headers])
  LLAMA_DIR=""
  LLAMA_INCDIR=""

  SEARCH_PATH="/usr/local /usr"
  if test "$PHP_LLAMA" != "yes" && test -n "$PHP_LLAMA"; then
    SEARCH_PATH="$PHP_LLAMA $SEARCH_PATH"
  fi

  for i in $SEARCH_PATH; do
    if test -r "$i/include/llama.h"; then
      dnl Source tree or prefix-style install
      LLAMA_DIR=$i
      LLAMA_INCDIR="$i/include"
      AC_MSG_RESULT(found in $i/include)
      break
    fi
  done

  if test -z "$LLAMA_DIR"; then
    AC_MSG_RESULT([not found])
    AC_MSG_ERROR([Please install llama.cpp or specify the path with --with-llama=DIR])
  fi

  PHP_ADD_INCLUDE($LLAMA_INCDIR)

  dnl ggml headers may be in a separate directory (source tree layout)
  if test -d "$LLAMA_DIR/ggml/include"; then
    PHP_ADD_INCLUDE($LLAMA_DIR/ggml/include)
  fi

  dnl Find libllama
  LIBLLAMA_FOUND=no
  for dir in "$LLAMA_DIR/build/bin" "$LLAMA_DIR/lib" "$LLAMA_DIR/lib64" "$LLAMA_DIR/build/src"; do
    if test -f "$dir/libllama.so" || test -f "$dir/libllama.so.0" || test -f "$dir/libllama.dylib"; then
      PHP_ADD_LIBPATH($dir, LLAMA_SHARED_LIBADD)
      LIBLLAMA_FOUND=yes
      break
    fi
  done

  PHP_ADD_LIBRARY(llama, 1, LLAMA_SHARED_LIBADD)
  PHP_ADD_LIBRARY(stdc++, 1, LLAMA_SHARED_LIBADD)
  PHP_SUBST(LLAMA_SHARED_LIBADD)

  dnl Check for json-schema-to-grammar support (optional, requires common library)
  LLAMA_SOURCES="llama.c sampler_safe.cpp"
  HAVE_JSON_SCHEMA=no

  AC_MSG_CHECKING([for llama.cpp common library (json-schema-to-grammar)])
  for dir in "$LLAMA_DIR/common" "$LLAMA_INCDIR"; do
    if test -f "$dir/json-schema-to-grammar.h"; then
      PHP_ADD_INCLUDE($dir)

      dnl Find nlohmann/json (required by json-schema-to-grammar)
      for jdir in "$LLAMA_DIR/vendor" "$LLAMA_DIR/common" "$LLAMA_DIR/build/_deps/json-src/include" "/usr/include" "/usr/local/include"; do
        if test -f "$jdir/nlohmann/json.hpp"; then
          PHP_ADD_INCLUDE($jdir)
          break
        fi
      done

      dnl Find libcommon.a
      for ldir in "$LLAMA_DIR/build/common" "$LLAMA_DIR/lib" "$LLAMA_DIR/lib64"; do
        if test -f "$ldir/libcommon.a"; then
          LLAMA_SHARED_LIBADD="$LLAMA_SHARED_LIBADD $ldir/libcommon.a"
          LLAMA_SOURCES="$LLAMA_SOURCES json_schema_shim.cpp"
          HAVE_JSON_SCHEMA=yes
          break
        fi
      done
      break
    fi
  done

  if test "$HAVE_JSON_SCHEMA" = "yes"; then
    AC_MSG_RESULT(yes)
    AC_DEFINE(HAVE_JSON_SCHEMA, 1, [Whether json-schema-to-grammar is available])
  else
    AC_MSG_RESULT(no - json_schema option will not be available)
  fi

  PHP_REQUIRE_CXX()
  PHP_NEW_EXTENSION(llama, $LLAMA_SOURCES, $ext_shared,, -DLLAMA_SHARED)
fi
