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
#  define dscal_  dscal
#  define zscal_  zscal
#  define daxpy_  daxpy
#  define zaxpy_  zaxpy
#  define dsyrk_  dsyrk
#  define zherk_  zherk
#  define dsyr2k_ dsyr2k
#  define zher2k_ zher2k
#  define dgemm_  dgemm
#  define zgemm_  zgemm
#  define dgemv_  dgemv
#  define zgemv_  zgemv
#  define ddot_   ddot
#endif


void dscal_(int*n, double* alpha, double* x, int* incx);

void zscal_(int*n, void* alpha, void* x, int* incx);

void daxpy_(int* n, double* alpha,
            double* x, int *incx,
            double* y, int *incy);
void zaxpy_(int* n, void* alpha,
            void* x, int *incx,
            void* y, int *incy);
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
void dgemv_(char *trans, int *m, int * n,
            double *alpha, double *a, int *lda,
            double *x, int *incx, double *beta,
            double *y, int *incy);
void zgemv_(char *trans, int *m, int * n,
            void *alpha, void *a, int *lda,
            void *x, int *incx, void *beta,
            void *y, int *incy);
double ddot_(int *n, void *dx, int *incx, void *dy, int *incy);

// The definitions of zdotc and zdotu might be
//     void zdotc_(void *ret_val, int *n, void *zx, int *incx, void *zy, int *incy);
//     void zdotu_(void *ret_val, int *n, void *zx, int *incx, void *zy, int *incy);
// or
//     double_complex zdotc_(int *n, void *zx, int *incx, void *zy, int *incy);
//     double_complex zdotu_(int *n, void *zx, int *incx, void *zy, int *incy);
// depending on the library (MKL, OpenBLAS) or its compilation options.
// To have a common definition, we take the functions through the cblas interface:
void cblas_zdotc_sub(int n, void *x, int incx, void *y, int incy, void *ret);
void cblas_zdotu_sub(int n, void *x, int incx, void *y, int incy, void *ret);


PyObject* scal(PyObject *self, PyObject *args)
{
  Py_complex alpha;
  PyArrayObject* x;
  if (!PyArg_ParseTuple(args, "DO", &alpha, &x))
    return NULL;
  int n = PyArray_DIMS(x)[0];
  Py_BEGIN_ALLOW_THREADS;
  for (int d = 1; d < PyArray_NDIM(x); d++)
    n *= PyArray_DIMS(x)[d];
  int incx = 1;

  if (PyArray_DESCR(x)->type_num == NPY_DOUBLE)
    dscal_(&n, &(alpha.real), DOUBLEP(x), &incx);
  else
    zscal_(&n, &alpha, (void*)COMPLEXP(x), &incx);

  Py_END_ALLOW_THREADS;
  Py_RETURN_NONE;
}


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
  Py_BEGIN_ALLOW_THREADS;
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
  Py_END_ALLOW_THREADS;
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

    void* a = PyArray_DATA(M2);
    void* b = PyArray_DATA(M1);
    void* c = PyArray_DATA(M3);

    int bytes = PyArray_ITEMSIZE(M3);

    int m = PyArray_DIM(M3, 1);
    int n = PyArray_DIM(M3, 0);
    int lda = PyArray_STRIDE(M2, 0) / bytes;
    int ldb = PyArray_STRIDE(M1, 0) / bytes;
    int ldc = MAX(MAX(1, m), PyArray_STRIDE(M3, 0) / bytes);

    int k;

    if (*trans2 == 'N' || *trans2 == 'n') {
        k = PyArray_DIM(M2, 0);
        lda = MAX(MAX(1, m), lda);
    }
    else {
        k = PyArray_DIM(M2, 1);
        lda = MAX(MAX(1, k), lda);
    }

    if (*trans1 == 'N' || *trans1 == 'n')
        ldb = MAX(MAX(1, k), ldb);
    else
        ldb = MAX(MAX(1, n), ldb);

    if (bytes == 8)
        dgemm_(trans2, trans1, &m, &n, &k,
               &(alpha.real), a, &lda, b, &ldb, &(beta.real), c, &ldc);
    else
        zgemm_(trans2, trans1, &m, &n, &k,
               &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

    Py_RETURN_NONE;
}


PyObject* gemv(PyObject *self, PyObject *args)
{
  Py_complex alpha;
  PyArrayObject* a;
  PyArrayObject* x;
  Py_complex beta;
  PyArrayObject* y;
  char t = 't';
  char* trans = &t;
  if (!PyArg_ParseTuple(args, "DOODO|s", &alpha, &a, &x, &beta, &y, &trans))
    return NULL;

  int m, n, lda, itemsize, incx, incy;

  Py_BEGIN_ALLOW_THREADS;
  if (*trans == 'n')
    {
      m = PyArray_DIMS(a)[1];
      for (int i = 2; i < PyArray_NDIM(a); i++)
        m *= PyArray_DIMS(a)[i];
      n = PyArray_DIMS(a)[0];
      lda = MAX(1, m);
    }
  else
    {
      n = PyArray_DIMS(a)[0];
      for (int i = 1; i < PyArray_NDIM(a)-1; i++)
        n *= PyArray_DIMS(a)[i];
      m = PyArray_DIMS(a)[PyArray_NDIM(a)-1];
      lda = MAX(1, m);
    }

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    itemsize = sizeof(double);
  else
    itemsize = sizeof(double_complex);

  incx = PyArray_STRIDES(x)[0]/itemsize;
  incy = 1;

  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    dgemv_(trans, &m, &n,
           &(alpha.real),
           DOUBLEP(a), &lda,
           DOUBLEP(x), &incx,
           &(beta.real),
           DOUBLEP(y), &incy);
  else
    zgemv_(trans, &m, &n,
           &alpha,
           (void*)COMPLEXP(a), &lda,
           (void*)COMPLEXP(x), &incx,
           &beta,
           (void*)COMPLEXP(y), &incy);
  Py_END_ALLOW_THREADS;
  Py_RETURN_NONE;
}


PyObject* axpy(PyObject *self, PyObject *args)
{
  Py_complex alpha;
  PyArrayObject* x;
  PyArrayObject* y;
  if (!PyArg_ParseTuple(args, "DOO", &alpha, &x, &y))
    return NULL;
  Py_BEGIN_ALLOW_THREADS;
  int n = PyArray_DIMS(x)[0];
  for (int d = 1; d < PyArray_NDIM(x); d++)
    n *= PyArray_DIMS(x)[d];
  int incx = 1;
  int incy = 1;
  if (PyArray_DESCR(x)->type_num == NPY_DOUBLE)
    daxpy_(&n, &(alpha.real),
           DOUBLEP(x), &incx,
           DOUBLEP(y), &incy);
  else
    zaxpy_(&n, &alpha,
           (void*)COMPLEXP(x), &incx,
           (void*)COMPLEXP(y), &incy);
  Py_END_ALLOW_THREADS;
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

    Py_BEGIN_ALLOW_THREADS;
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

    int ldc = MAX(MAX(1, n), PyArray_STRIDES(c)[0] / PyArray_ITEMSIZE(c));
    if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
        dsyrk_("u", trans, &n, &k,
               &alpha, DOUBLEP(a), &lda, &beta,
               DOUBLEP(c), &ldc);
    else
        zherk_("u", trans, &n, &k,
               &alpha, (void*)COMPLEXP(a), &lda, &beta,
               (void*)COMPLEXP(c), &ldc);
    Py_END_ALLOW_THREADS;
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
  int ldc = MAX(MAX(1, n), PyArray_STRIDES(c)[0] / PyArray_ITEMSIZE(c));

  Py_BEGIN_ALLOW_THREADS;
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
  Py_END_ALLOW_THREADS;

  Py_RETURN_NONE;
}

PyObject* dotc(PyObject *self, PyObject *args)
{
  PyArrayObject* a;
  PyArrayObject* b;
  if (!PyArg_ParseTuple(args, "OO", &a, &b))
    return NULL;
  int n = PyArray_DIMS(a)[0];
  for (int i = 1; i < PyArray_NDIM(a); i++)
    n *= PyArray_DIMS(a)[i];
  int incx = 1;
  int incy = 1;
  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      double result;
      Py_BEGIN_ALLOW_THREADS;
      result = ddot_(&n, (void*)DOUBLEP(a),
             &incx, (void*)DOUBLEP(b), &incy);
      Py_END_ALLOW_THREADS;
      return PyFloat_FromDouble(result);
    }
  else
    {
      double_complex* ap = COMPLEXP(a);
      double_complex* bp = COMPLEXP(b);
      double_complex result;
      Py_BEGIN_ALLOW_THREADS;
      cblas_zdotc_sub(n, ap, incx, bp, incy, &result);
      Py_END_ALLOW_THREADS;
      return PyComplex_FromDoubles(creal(result), cimag(result));
    }
}


PyObject* dotu(PyObject *self, PyObject *args)
{
  PyArrayObject* a;
  PyArrayObject* b;
  if (!PyArg_ParseTuple(args, "OO", &a, &b))
    return NULL;
  int n = PyArray_DIMS(a)[0];
  for (int i = 1; i < PyArray_NDIM(a); i++)
    n *= PyArray_DIMS(a)[i];
  int incx = 1;
  int incy = 1;
  if (PyArray_DESCR(a)->type_num == NPY_DOUBLE)
    {
      double result;
      Py_BEGIN_ALLOW_THREADS;
      result = ddot_(&n, (void*)DOUBLEP(a),
             &incx, (void*)DOUBLEP(b), &incy);
      Py_END_ALLOW_THREADS;
      return PyFloat_FromDouble(result);
    }
  else
    {
      double_complex* ap = COMPLEXP(a);
      double_complex* bp = COMPLEXP(b);
      double_complex result;
      Py_BEGIN_ALLOW_THREADS;
      cblas_zdotu_sub(n, ap, incx, bp, incy, &result);
      Py_END_ALLOW_THREADS;
      return PyComplex_FromDoubles(creal(result), cimag(result));
    }
}
#endif
