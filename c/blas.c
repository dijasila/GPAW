/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Copyright (C) 2007       CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#ifndef GPAW_WITHOUT_BLAS
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"

#ifdef GPAW_NO_UNDERSCORE_BLAS
#  define dsyrk_  dsyrk
#  define zherk_  zherk
#  define dsyr2k_ dsyr2k
#  define zher2k_ zher2k
#  define dgemm_  dgemm
#  define zgemm_  zgemm
#endif


void dsyrk_(char *uplo, char *trans, int *n, int *k,
            double *alpha, double *a, int *lda, double *beta,
            double *c, int *ldc);
void zherk_(char *uplo, char *trans, int *n, int *k,
            double *alpha, void *a, int *lda,
            double *beta,
            void *c, int *ldc);
void dsyr2k_(char *uplo, char *trans, int *n, int *k,
             double *alpha, double *a, int *lda,
             double *b, int *ldb, double *beta,
             double *c, int *ldc);
void zher2k_(char *uplo, char *trans, int *n, int *k,
             void *alpha, void *a, int *lda,
             void *b, int *ldb, double *beta,
             void *c, int *ldc);
void dgemm_(char *transa, char *transb, int *m, int * n,
            int *k, double *alpha, double *a, int *lda,
            double *b, int *ldb, double *beta,
            double *c, int *ldc);
void zgemm_(char *transa, char *transb, int *m, int * n,
            int *k, void *alpha, void *a, int *lda,
            void *b, int *ldb, void *beta,
            void *c, int *ldc);


PyObject* gemm(PyObject *self, PyObject *args)
{
  Py_complex alpha;
  PyArrayObject* a;
  PyArrayObject* b;
  Py_complex beta;
  PyArrayObject* c;
  char t = 'n';
  char* transa = &t;
  if (!PyArg_ParseTuple(args, "DOODO|s", &alpha, &a, &b, &beta, &c, &transa))
    return NULL;
  int m, k, lda, ldb, ldc;
  if (*transa == 'n')
    {
      m = PyArray_DIMS(a)[1];
      for (int i = 2; i < PyArray_NDIM(a); i++)
        m *= PyArray_DIMS(a)[i];
      k = PyArray_DIMS(a)[0];
      lda = MAX(1, PyArray_STRIDES(a)[0] / PyArray_STRIDES(a)[PyArray_NDIM(a) - 1]);
      ldb = MAX(1, PyArray_STRIDES(b)[0] / PyArray_STRIDES(b)[1]);
      ldc = MAX(1, PyArray_STRIDES(c)[0] / PyArray_STRIDES(c)[PyArray_NDIM(c) - 1]);
    }
  else
    {
      k = PyArray_DIMS(a)[1];
      for (int i = 2; i < PyArray_NDIM(a); i++)
        k *= PyArray_DIMS(a)[i];
      m = PyArray_DIMS(a)[0];
      lda = MAX(1, k);
      ldb = MAX(1, PyArray_STRIDES(b)[0] / PyArray_STRIDES(b)[PyArray_NDIM(b) - 1]);
      ldc = MAX(1, PyArray_STRIDES(c)[0] / PyArray_STRIDES(c)[1]);

    }
  int n = PyArray_DIMS(b)[0];
  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    dgemm_(transa, "n", &m, &n, &k,
           &(alpha.real),
           DOUBLEP(a), &lda,
           DOUBLEP(b), &ldb,
           &(beta.real),
           DOUBLEP(c), &ldc);
  else
    zgemm_(transa, "n", &m, &n, &k,
           &alpha,
           (void*)COMPLEXP(a), &lda,
           (void*)COMPLEXP(b), &ldb,
           &beta,
           (void*)COMPLEXP(c), &ldc);
  Py_RETURN_NONE;
}


PyObject* mmm(PyObject *self, PyObject *args)
{
    Py_complex alpha;
    PyArrayObject* M1;
    char* trans1;
    PyArrayObject* M2;
    char* trans2;
    Py_complex beta;
    PyArrayObject* M3;

    if (!PyArg_ParseTuple(args, "DOsOsDO",
                          &alpha, &M1, &trans1, &M2, &trans2, &beta, &M3))
        return NULL;

    int m = PyArray_DIM(M3, 1);
    int n = PyArray_DIM(M3, 0);
    int k;

    int bytes = PyArray_ITEMSIZE(M3);
    int lda = MAX(1, PyArray_STRIDE(M2, 0) / bytes);
    int ldb = MAX(1, PyArray_STRIDE(M1, 0) / bytes);
    int ldc = MAX(1, PyArray_STRIDE(M3, 0) / bytes);

    void* a = PyArray_DATA(M2);
    void* b = PyArray_DATA(M1);
    void* c = PyArray_DATA(M3);

    if (*trans2 == 'N' || *trans2 == 'n')
        k = PyArray_DIM(M2, 0);
    else
        k = PyArray_DIM(M2, 1);

    if (bytes == 8)
        dgemm_(trans2, trans1, &m, &n, &k,
               &(alpha.real), a, &lda, b, &ldb, &(beta.real), c, &ldc);
    else
        zgemm_(trans2, trans1, &m, &n, &k,
               &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

    Py_RETURN_NONE;
}


PyObject* rk(PyObject *self, PyObject *args)
{
    double alpha;
    PyArrayObject* a;
    double beta;
    PyArrayObject* c;
    char t = 'c';
    char* trans = &t;
    if (!PyArg_ParseTuple(args, "dOdO|s", &alpha, &a, &beta, &c, &trans))
        return NULL;

    int n = PyArray_DIMS(c)[0];

    int k, lda;

    if (*trans == 'c') {
        k = PyArray_DIMS(a)[1];
        for (int d = 2; d < PyArray_NDIM(a); d++)
            k *= PyArray_DIMS(a)[d];
        lda = MAX(k, 1);
    }
    else {
        k = PyArray_DIMS(a)[0];
        lda = MAX(n, 1);
    }

    int ldc = PyArray_STRIDES(c)[0] / PyArray_STRIDES(c)[1];
    if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
        dsyrk_("u", trans, &n, &k,
               &alpha, DOUBLEP(a), &lda, &beta,
               DOUBLEP(c), &ldc);
    else
        zherk_("u", trans, &n, &k,
               &alpha, (void*)COMPLEXP(a), &lda, &beta,
               (void*)COMPLEXP(c), &ldc);
    Py_RETURN_NONE;
}

PyObject* r2k(PyObject *self, PyObject *args)
{
    Py_complex alpha;
    PyArrayObject* a;
    PyArrayObject* b;
    double beta;
    PyArrayObject* c;
    char t = 'c';
    char* trans = &t;

    if (!PyArg_ParseTuple(args, "DOOdO|s", &alpha, &a, &b, &beta, &c, &trans))
        return NULL;

    int n = PyArray_DIMS(c)[0];
    int k, lda;
    if (*trans == 'c') {
        k = PyArray_DIMS(a)[1];
        for (int d = 2; d < PyArray_NDIM(a); d++)
            k *= PyArray_DIMS(a)[d];
        lda = MAX(k, 1);
    } else {
        k = PyArray_DIMS(a)[0];
        lda = MAX(n, 1);
    }
  int ldc = PyArray_STRIDES(c)[0] / PyArray_STRIDES(c)[1];

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    dsyr2k_("u", trans, &n, &k,
            (double*)(&alpha), DOUBLEP(a), &lda,
            DOUBLEP(b), &lda, &beta,
            DOUBLEP(c), &ldc);
  else
    zher2k_("u", trans, &n, &k,
            (void*)(&alpha), (void*)COMPLEXP(a), &lda,
            (void*)COMPLEXP(b), &lda, &beta,
            (void*)COMPLEXP(c), &ldc);

  Py_RETURN_NONE;
}
#endif
