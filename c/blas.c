#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"

#ifdef GPAW_AIX
#  define daxpy_ daxpy
#  define zaxpy_ zaxpy
#  define dsyrk_ dsyrk
#  define zherk_ zherk
#  define dsyr2k_ dsyr2k
#  define zher2k_ zher2k
#  define dgemm_ dgemm
#  define zgemm_ zgemm
#  define ddot_ ddot
#endif

#ifdef GPAW_CUDA
#include <cublas.h>
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
#endif

void daxpy_(int* n, double* alpha,
	    double* x, int *incx, 
	    double* y, int *incy);
void zaxpy_(int* n, void* alpha,
	    void* x, int *incx, 
	    void* y, int *incy);
int dsyrk_(char *uplo, char *trans, int *n, int *k, 
	   double *alpha, double *a, int *lda, double *beta, 
	   double *c, int *ldc);
int zherk_(char *uplo, char *trans, int *n, int *k, 
	   double *alpha, void *a, int *lda,
	   double *beta, 
	   void *c, int *ldc);
int dsyr2k_(char *uplo, char *trans, int *n, int *k, 
	    double *alpha, double *a, int *lda, 
	    double *b, int *ldb, double *beta, 
	    double *c, int *ldc);
int zher2k_(char *uplo, char *trans, int *n, int *k, 
	    void *alpha, void *a, int *lda,
	    void *b, int *ldb, double *beta, 
	    void *c, int *ldc);
int dgemm_(char *transa, char *transb, int *m, int * n,
	   int *k, double *alpha, double *a, int *lda, 
	   double *b, int *ldb, double *beta, 
	   double *c, int *ldc);
int zgemm_(char *transa, char *transb, int *m, int * n,
	   int *k, void *alpha, void *a, int *lda, 
	   void *b, int *ldb, void *beta,
	   void *c, int *ldc);
double ddot_(int *n, void *dx, int *incx, void *dy, int *incy);

PyObject* gemm(PyObject *self, PyObject *args)
{
  Py_complex alpha;
  PyArrayObject* a;
  PyArrayObject* b;
  Py_complex beta;
  PyArrayObject* c;
  char transa = 'n';
#ifdef GPAW_CUDA
double* devPtrA;
double* devPtrB;
double* devPtrC;
#endif
  if (!PyArg_ParseTuple(args, "DOODO|c", &alpha, &a, &b, &beta, &c, &transa)) 
    return NULL;
  int m, k, lda, ldb, ldc;
  if (transa == 'n')
    {
      m = a->dimensions[1];
      for (int i = 2; i < a->nd; i++)
	m *= a->dimensions[i];
      k = a->dimensions[0];
      lda = m;
      ldb = b->strides[0] / b->strides[1];
      ldc = m;
    } 
  else
    {
      k = a->dimensions[1];
      for (int i = 2; i < a->nd; i++)
	k *= a->dimensions[i];
      m = a->dimensions[0];
      lda = k;
      ldb = k;
      ldc = c->strides[0] / c->strides[1];
    } 
  int n = b->dimensions[0];
  if (a->descr->type_num == PyArray_DOUBLE)
#ifdef GPAW_CUDA
    {

	  cublasAlloc (m*k, sizeof(double), (void**)&devPtrA);
	  cublasAlloc (k*n, sizeof(double), (void**)&devPtrB);
	  cublasAlloc (m*n, sizeof(double), (void**)&devPtrC);

	  cublasSetMatrix (m, k, sizeof(double), DOUBLEP(a), m, devPtrA, m);
	  cublasSetMatrix (k, n, sizeof(double), DOUBLEP(b), k, devPtrB, k);
	  cublasSetMatrix (m, n, sizeof(double), DOUBLEP(c), m, devPtrC, m);

	  cublasDgemm(transa, 'n', m, n, k, alpha.real,
		      devPtrA, lda, devPtrB, ldb, beta.real,
		      devPtrC, ldc);

	  cublasGetMatrix (m, n, sizeof(double), devPtrC, m, DOUBLEP(c), m);
	  cublasFree (devPtrA);
	  cublasFree (devPtrB);
	  cublasFree (devPtrC);
    }
#else   
    dgemm_(&transa, "n", &m, &n, &k, 
           &(alpha.real),
           DOUBLEP(a), &lda, 
           DOUBLEP(b), &ldb,
           &(beta.real), 
           DOUBLEP(c), &ldc);
#endif
  else
    zgemm_(&transa, "n", &m, &n, &k, 
           &alpha,
           (void*)COMPLEXP(a), &lda, 
           (void*)COMPLEXP(b), &ldb,
           &beta, 
           (void*)COMPLEXP(c), &ldc);
  Py_RETURN_NONE;
}


PyObject* axpy(PyObject *self, PyObject *args)
{
  PyObject* alpha;
  PyArrayObject* x;
  PyArrayObject* y;
  if (!PyArg_ParseTuple(args, "OOO", &alpha, &x, &y)) 
    return NULL;
  int n = x->dimensions[0];
  for (int d = 1; d < x->nd; d++)
    n *= x->dimensions[d];
  int incx = 1;
  int incy = 1;
#ifdef GPAW_CUDDA
  double* devPtrx;
  double* devPtry;

  cublasAlloc (n, sizeof(double), (void**)&devPtrx);
  cublasAlloc (n, sizeof(double), (void**)&devPtry);
  cublasSetVector (n, sizeof(double), DOUBLEP(x), 1, devPtrx, 1);
  cublasSetVector (n, sizeof(double), DOUBLEP(y), 1, devPtry, 1);

  PyFloatObject* palpha = (PyFloatObject*)alpha;
  cublasDaxpy(n, palpha->ob_fval, devPtrx, 1, devPtry, 1);

  cublasGetVector(n, sizeof(double), devPtry, 1, DOUBLEP(y), 1);
  cublasFree(devPtrx);
  cublasFree(devPtry);
#else
  if (PyFloat_Check(alpha))
    {
      if (x->descr->type_num == PyArray_CDOUBLE)
	n *= 2;
      PyFloatObject* palpha = (PyFloatObject*)alpha;
      daxpy_(&n, &(palpha->ob_fval), 
            DOUBLEP(x), &incx,
            DOUBLEP(y), &incy);
    }
  else
    {
      PyComplexObject* palpha = (PyComplexObject*)alpha;
      zaxpy_(&n, (void*)(&(palpha->cval)), 
             (void*)COMPLEXP(x), &incx,
             (void*)COMPLEXP(y), &incy);
    }
#endif  
  Py_RETURN_NONE;
}

PyObject* rk(PyObject *self, PyObject *args)
{
  double alpha;
  PyArrayObject* a;
  double beta;
  PyArrayObject* c;
#ifdef GPAW_CUDA
double* devPtrA;
double* devPtrC;
#endif
  if (!PyArg_ParseTuple(args, "dOdO", &alpha, &a, &beta, &c)) 
    return NULL;
  int n = a->dimensions[0];
  int k = a->dimensions[1];
  for (int d = 2; d < a->nd; d++)
    k *= a->dimensions[d];
  int ldc = c->strides[0] / c->strides[1];
  if (a->descr->type_num == PyArray_DOUBLE)
#ifdef GPAW_CUDA
    {

	  cublasAlloc (n*k, sizeof(double), (void**)&devPtrA);
	  cublasAlloc (n*n, sizeof(double), (void**)&devPtrC);

	  cublasSetMatrix (n, k, sizeof(double), DOUBLEP(a), n, devPtrA, n);
	  cublasSetMatrix (n, n, sizeof(double), DOUBLEP(c), n, devPtrC, n);

	  cublasDsyrk('u', 'c', n, k, alpha,
		       devPtrA, k, beta,
		      devPtrC, ldc);

	  cublasGetMatrix (n, n, sizeof(double), devPtrC, n, DOUBLEP(c), n);
	  cublasFree (devPtrA);
	  cublasFree (devPtrC);
    }
#else   
    dsyrk_("u", "c", &n, &k, 
           &alpha, DOUBLEP(a), &k, &beta,
           DOUBLEP(c), &ldc);
#endif
  else
    zherk_("u", "c", &n, &k, 
           &alpha, (void*)COMPLEXP(a), &k, &beta,
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
#ifdef GPAW_CUDA
double* devPtrA;
double* devPtrB;
double* devPtrC;
#endif
  if (!PyArg_ParseTuple(args, "DOOdO", &alpha, &a, &b, &beta, &c)) 
    return NULL;
  int n = a->dimensions[0];
  int k = a->dimensions[1];
  for (int d = 2; d < a->nd; d++)
    k *= a->dimensions[d];
  if (a->descr->type_num == PyArray_DOUBLE)
#ifdef GPAW_CUDA
    {

	  cublasAlloc (n*k, sizeof(double), (void**)&devPtrA);
	  cublasAlloc (n*k, sizeof(double), (void**)&devPtrB);
	  cublasAlloc (n*n, sizeof(double), (void**)&devPtrC);

	  cublasSetMatrix (n, k, sizeof(double), DOUBLEP(a), n, devPtrA, n);
	  cublasSetMatrix (n, k, sizeof(double), DOUBLEP(b), n, devPtrB, n);
	  cublasSetMatrix (n, n, sizeof(double), DOUBLEP(c), n, devPtrC, n);

	  cublasDsyr2k('u', 'c', n, k, alpha.real,
		       devPtrA, k, devPtrB, k, beta,
		      devPtrC, n);

	  cublasGetMatrix (n, n, sizeof(double), devPtrC, n, DOUBLEP(c), n);
	  cublasFree (devPtrA);
	  cublasFree (devPtrB);
	  cublasFree (devPtrC);
    }
#else   
    dsyr2k_("u", "c", &n, &k, 
            (double*)(&alpha), DOUBLEP(a), &k, 
            DOUBLEP(b), &k, &beta,
            DOUBLEP(c), &n);
#endif
  else
    zher2k_("u", "c", &n, &k, 
            (void*)(&alpha), (void*)COMPLEXP(a), &k, 
            (void*)COMPLEXP(b), &k, &beta,
            (void*)COMPLEXP(c), &n);
  Py_RETURN_NONE;
}

PyObject* dotc(PyObject *self, PyObject *args)
{
  PyArrayObject* a;
  PyArrayObject* b;
  if (!PyArg_ParseTuple(args, "OO", &a, &b)) 
    return NULL;
  int n = a->dimensions[0];
  for (int i = 1; i < a->nd; i++)
    n *= a->dimensions[i];
  int incx = 1;
  int incy = 1;
  if (a->descr->type_num == PyArray_DOUBLE)
    {
      double result;
      result = ddot_(&n, (void*)DOUBLEP(a), 
	     &incx, (void*)DOUBLEP(b), &incy);
      return PyFloat_FromDouble(result);
    }
  else
    {
      double_complex* ap = COMPLEXP(a);
      double_complex* bp = COMPLEXP(b);
#ifndef NO_C99_COMPLEX
      double_complex z = 0.0;
      for (int i = 0; i < n; i++)
	z += conj(ap[i]) * bp[i];
      return PyComplex_FromDoubles(creal(z), cimag(z));
#else
      double_complex z = {0.0, 0.0};
      for (int i = 0; i < n; i++)
        {
          z.r += ap[i].r * bp[i].r + ap[i].i * bp[i].i;
          z.i += ap[i].r * bp[i].i - ap[i].i * bp[i].r;
        }
      return PyComplex_FromDoubles(z.r, z.i);
#endif
    }
}


PyObject* dotu(PyObject *self, PyObject *args)
{
  PyArrayObject* a;
  PyArrayObject* b;
  if (!PyArg_ParseTuple(args, "OO", &a, &b)) 
    return NULL;
  int n = a->dimensions[0];
  for (int i = 1; i < a->nd; i++)
    n *= a->dimensions[i];
  int incx = 1;
  int incy = 1;
  if (a->descr->type_num == PyArray_DOUBLE)
    {
      double result;
      result = ddot_(&n, (void*)DOUBLEP(a), 
	     &incx, (void*)DOUBLEP(b), &incy);
      return PyFloat_FromDouble(result);
    }
  else
    {
      double_complex* ap = COMPLEXP(a);
      double_complex* bp = COMPLEXP(b);
#ifndef NO_C99_COMPLEX
      double_complex z = 0.0;
      for (int i = 0; i < n; i++)
	z += ap[i] * bp[i];
      return PyComplex_FromDoubles(creal(z), cimag(z));
#else
      double_complex z = {0.0, 0.0};
      for (int i = 0; i < n; i++)
        {
          z.r += ap[i].r * bp[i].r - ap[i].i * bp[i].i;
          z.i += ap[i].r * bp[i].i + ap[i].i * bp[i].r;
        }
      return PyComplex_FromDoubles(z.r, z.i);
#endif
    }
}
