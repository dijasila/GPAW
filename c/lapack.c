/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Copyright (C) 2005-2007  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"
#ifdef GPAW_NO_UNDERSCORE_LAPACK
#  define zgttrf_ zgttrf
#  define zgttrs_ zgttrs
#endif
void zgttrf_(int* n, void* dl, void* d, void* du,
            void* du2, int* ipiv, int* info);
void zgttrs_(char* tran, int* n, int* nrhs, void* dl,
               void* d, void* du, void* du2,
               int* ipiv, void* b, int* ldb, int* info);


PyObject* linear_solve_tridiag(PyObject *self, PyObject *args)
{
 PyArrayObject* A;
 PyArrayObject* du;
 PyArrayObject* du2;
 PyArrayObject* dl;
 PyArrayObject* phi;
 int dim=0, one=1, info=0;
 if(!PyArg_ParseTuple(args,"iOOOOO", &dim, &A, &du, &dl, &du2, &phi))
   return NULL;
 int ldb = dim;
 int *ipiv = GPAW_MALLOC(int, dim);
 zgttrf_(&dim, (void*)COMPLEXP(dl), (void*)COMPLEXP(A), (void*)COMPLEXP(du), (void*)COMPLEXP(du2), ipiv, &info);
 zgttrs_("N", &dim, &one, (void*)COMPLEXP(dl), (void*)COMPLEXP(A), (void*)COMPLEXP(du),
                                   (void*)COMPLEXP(du2), ipiv, (void*)COMPLEXP(phi), &ldb, &info);
 free(ipiv);
 return Py_BuildValue("i",info);
}
