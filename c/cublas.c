#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

PyObject* cugemm(PyObject *self, PyObject *args)
{
  cublasHandle_t handle;
  cuDoubleComplex alpha;
  cuDoubleComplex* a;
  cuDoubleComplex* b;
  cuDoubleComplex beta;
  cuDoubleComplex* c;
  cublasOperation_t transa = CUBLAS_OP_N;

  int n, m, k, lda, ldb, ldc;
  cublasOperation_t transb = CUBLAS_OP_N;

  if (!PyArg_ParseTuple(args, "LDLLDLiiiiii|ii",&handle, &alpha, &a, &b, &beta, &c,&n, &m, &k, &lda, &ldb, &ldc, &transb, &transa))
    return NULL;
  cublasZgemm(handle, 
              transa,
              transb, 
              m,
              n,
              k,
              &alpha, /* host or device pointer */  
              a, 
              lda,
              b,
              ldb, 
              &beta, /* host or device pointer */  
              c,
              ldc);  

  Py_RETURN_NONE;
}

PyObject* cuCgemv(PyObject *self, PyObject *args)
{
  cublasHandle_t handle;
  float alpha, beta;
  cuComplex alphacomplex, betacomplex;
  cuComplex* a;
  cuComplex* x;
  cuComplex* y;
  cublasOperation_t trans = CUBLAS_OP_T;
  int m, n, lda, incx, incy;
  cuComplex junk;
  if (!PyArg_ParseTuple(args, "LiifLiLifLi|i",&handle, &m, &n, &alpha, &a, &lda, &x, &incx, &beta, &y, &incy, &trans))
    return NULL;
  alphacomplex = make_cuFloatComplex(alpha,0.0);
  betacomplex = make_cuFloatComplex(beta,0.0);
  /*  printf("alpha %f %f\n",cuCrealf(alphacomplex),cuCimagf(alphacomplex)); */

  cublasCgemv(handle, trans, m, n,
	      &alphacomplex, /* host or device pointer */
	      a, lda,
	      x, incx,
	      &betacomplex, /* host or device pointer */
	      y, incy);

  Py_RETURN_NONE;
}


PyObject* cuZgemv(PyObject *self, PyObject *args)
{
  cublasHandle_t handle;
  cuDoubleComplex alpha, beta;
  cuDoubleComplex* a;
  cuDoubleComplex* x;
  cuDoubleComplex* y;
  cublasOperation_t trans = CUBLAS_OP_T;
  int m, n, lda, incx, incy;
  if (!PyArg_ParseTuple(args, "LiiDLiLiDLi|i",&handle, &m, &n, &alpha, &a, &lda, &x, &incx, &beta, &y, &incy, &trans))
    return NULL;

/*   printf("m %d n %d lda %d incx %d incy %d\n", */
/* 	 m,n,lda,incx,incy); */

  cublasZgemv(handle, trans, m, n,
	      &alpha, /* host or device pointer */
	      a, lda,
	      x, incx,
	      &beta, /* host or device pointer */
	      y, incy);

  Py_RETURN_NONE;
}

PyObject* cuCreate(PyObject *self, PyObject *args)
{
  cublasHandle_t handle;
  cublasStatus_t cudaStat;
  int devid=0;
  if (!PyArg_ParseTuple(args, "|i",&devid))
    return NULL;

  cudaSetDevice(devid);
  cudaStat = cublasCreate(&handle);
  if (cudaStat != cudaSuccess) {
    printf("cublasCreate failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("IL",cudaStat,handle);
}

PyObject* cuMalloc(PyObject *self, PyObject *args)
{
  Py_ssize_t nbytes;
  if (!PyArg_ParseTuple(args,"n",&nbytes))
    return NULL;
  cudaError_t cudaStat;
  void *devPtr;
  cudaStat = cudaMalloc(&devPtr,nbytes);
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("IL",cudaStat,devPtr);
}

PyObject* cuFree(PyObject *self, PyObject *args)
{
  void *devPtr;
  if (!PyArg_ParseTuple(args,"L",&devPtr))
    return NULL;
  cudaError_t cudaStat;
  cudaStat = cudaFree(devPtr);
  if (cudaStat != cudaSuccess) {
    printf("device memory free failed %d at %p\n",cudaStat,devPtr);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}

#define VOIDP(a) ((void*) ((a)->data))

PyObject* cuSetMatrix(PyObject *self, PyObject *args)
{
  int m,n,elemsize,lda,ldb;
  void *devPtr;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args,"iiiOiLi",&m,&n,&elemsize,&x,&lda,&devPtr,&ldb))
    return NULL;
  const void *a = VOIDP(x);
/*   printf("%d %d %d %p %d %p %d\n",m,n,elemsize,a,lda,devPtr,ldb); */
  cudaError_t cudaStat;
  cudaStat = cublasSetMatrix(m,n,elemsize,a,lda,devPtr,ldb);
  if (cudaStat != cudaSuccess) {
    printf("cublasSetMatrix failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}

PyObject* cuGetMatrix(PyObject *self, PyObject *args)
{
  int m,n,elemsize,lda,ldb;
  const void *devPtr;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args,"iiiLiOi",&m,&n,&elemsize,&devPtr,&lda,&x,&ldb))
    return NULL;
  void *b = VOIDP(x);
/*   printf("%d %d %d %p %d %p %d\n",m,n,elemsize,devPtr,lda,b,ldb); */
  cudaError_t cudaStat;
  cudaStat = cublasGetMatrix(m,n,elemsize,devPtr,lda,b,ldb);
  if (cudaStat != cudaSuccess) {
    printf("cublasGetMatrix failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}

PyObject* cuSetVector(PyObject *self, PyObject *args)
{
  int n,elemsize,incx,incy;
  void *devPtr;
  PyArrayObject *x;
  if (!PyArg_ParseTuple(args,"iiOiLi",&n,&elemsize,&x,&incx,&devPtr,&incy))
    return NULL;
  const void *ptr = VOIDP(x);
/*   printf("%d %d %p %d %p %d\n",n,elemsize,ptr,incx,devPtr,incy); */
  cudaError_t cudaStat;
  cudaStat = cublasSetVector(n,elemsize,ptr,incx,devPtr,incy);
  if (cudaStat != cudaSuccess) {
    printf("cublasSetVector failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}

PyObject* cuGetVector(PyObject *self, PyObject *args)
{
  int n,elemsize,incx,incy;
  const void *devPtr;
  PyArrayObject *y;
  if (!PyArg_ParseTuple(args,"iiLiOi",&n,&elemsize,&devPtr,&incx,&y,&incy))
    return NULL;
  void *ptr = VOIDP(y);
/*   printf("%d %d %p %d %p %d\n",n,elemsize,devPtr,incx,ptr,incy); */
  cudaError_t cudaStat;
  cudaStat = cublasGetVector(n,elemsize,devPtr,incx,ptr,incy);
  if (cudaStat != cudaSuccess) {
    printf("cublasGetVector failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}

PyObject* cuZher(PyObject *self, PyObject *args)
{
  cublasHandle_t handle;
  cublasFillMode_t uplo;
  int n;
  const double alpha;
  int incx;
  int lda; 
  void *A,*x;

  if (!PyArg_ParseTuple(args,"LiidLiLi",&handle,&uplo,&n,&alpha,&x,&incx,
                        &A,&lda))
    return NULL;

  cudaError_t cudaStat;
/*   printf("%p %d %d %f %p %d %p %d\n",handle,uplo,n,alpha,x,incx,A,lda); */
  cudaStat = cublasZher(handle,uplo,n,&alpha,x,incx,A,lda);
  if (cudaStat != cudaSuccess) {
    printf("cublasZher failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}

PyObject* cuCher(PyObject *self, PyObject *args)
{
  cublasHandle_t handle;
  cublasFillMode_t uplo;
  int n;
  const float alpha;
  int incx;
  int lda; 
  void *A,*x;

  if (!PyArg_ParseTuple(args,"LiifLiLi",&handle,&uplo,&n,&alpha,&x,&incx,
                        &A,&lda))
    return NULL;

  cudaError_t cudaStat;
/*   printf("%p %d %d %f %p %d %p %d\n",handle,uplo,n,alpha,x,incx,A,lda); */
  cudaStat = cublasCher(handle,uplo,n,&alpha,x,incx,A,lda);
  if (cudaStat != cudaSuccess) {
    printf("cublasZher failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}
