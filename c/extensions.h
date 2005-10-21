#include "Python.h"
#define NO_IMPORT_ARRAY
#include <Numeric/arrayobject.h>

#ifndef DOUBLECOMPLEXDEFINED
#  define DOUBLECOMPLEXDEFINED 1
#  ifdef NO_C99_COMPLEX
     typedef struct { double r, i; } double_complex;
#  else
#    include <complex.h>
     typedef double complex double_complex;
#  endif
#endif

#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION < 4
#  define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#endif

#define INTP(a) ((int*)((a)->data))
#define DOUBLEP(a) ((double*)((a)->data))
#define COMPLEXP(a) ((double_complex*)((a)->data))
