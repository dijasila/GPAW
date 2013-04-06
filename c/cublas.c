#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include "cuErrCheck.h"
#include "cukernels.h"

PyObject* cuZscal(PyObject *self, PyObject *args)
{
  cublasHandle_t handle;
  int n, incx;
  cuDoubleComplex alpha;
  void *x;

  if (!PyArg_ParseTuple(args, "LiDLi",&handle, &n, &alpha, &x, &incx))
    return NULL;
  CublasSafeCall(cublasZscal(handle, n, &alpha, x, incx));
  CudaCheckError();
  Py_RETURN_NONE;
}


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
  CublasSafeCall(cublasZgemm(handle, 
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
                           ldc));
  CudaCheckError();

  Py_RETURN_NONE;
}


PyObject* cudgmm(PyObject *self, PyObject *args)
{
  cublasHandle_t handle;
  cublasSideMode_t mode = CUBLAS_SIDE_LEFT; /* left : 0 , right : 1*/
  cuDoubleComplex* A;
  cuDoubleComplex* x;
  cuDoubleComplex* C;
  int n, m, lda, incx, ldc;

  if (!PyArg_ParseTuple(args, "LLLLiiiii|i",&handle, &A, &x, &C, &n, &m, &lda, &ldc, &incx, &mode))
    return NULL;
  CublasSafeCall(cublasZdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc));
  CudaCheckError();
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

  CublasSafeCall(cublasCgemv(handle, trans, m, n,
                           &alphacomplex, /* host or device pointer */
                           a, lda,
                           x, incx,
                           &betacomplex, /* host or device pointer */
                           y, incy));

  CudaCheckError();
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

  CublasSafeCall(cublasZgemv(handle, trans, m, n,
                           &alpha, /* host or device pointer */
                           a, lda,
                           x, incx,
                           &beta, /* host or device pointer */
                           y, incy));

  CudaCheckError();
  Py_RETURN_NONE;
}

PyObject* cuCreate(PyObject *self, PyObject *args)
{
  cublasHandle_t handle;
  cublasStatus_t cudaStat;
  int devid=-1;
  if (!PyArg_ParseTuple(args, "|i",&devid))
    return NULL;

  if (devid>=0) cudaSetDevice(devid);
  CublasSafeCall(cudaStat=cublasCreate(&handle));
  CudaCheckError();
  if (cudaStat != cudaSuccess) {
    printf("cublasCreate failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("IL",cudaStat,handle);
}

PyObject* cuDestroy(PyObject *self, PyObject *args)
{
  cublasHandle_t handle;
  cublasStatus_t cudaStat;
  if (!PyArg_ParseTuple(args, "i",&handle))
    return NULL;

  CublasSafeCall(cudaStat = cublasDestroy(handle));
  CudaCheckError();
  if (cudaStat != cudaSuccess) {
    printf("cublasDestroy failed %d\n",cudaStat);
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
  CudaSafeCall(cudaStat = cudaMalloc(&devPtr,nbytes));
  CudaCheckError();
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
  CudaSafeCall(cudaStat = cudaFree(devPtr));
  CudaCheckError();
  if (cudaStat != cudaSuccess) {
    printf("device memory free failed %d at %p\n",cudaStat,devPtr);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}


PyObject* cuMemset(PyObject *self, PyObject *args)
{
  void *devPtr;
  int a, n;
  if (!PyArg_ParseTuple(args,"Lii",&devPtr,&a,&n))
    return NULL;
  cudaError_t cudaStat;
  CudaSafeCall(cudaStat = cudaMemset(devPtr, a, n));
  CudaCheckError();
  if (cudaStat != cudaSuccess) {
    printf("device memory set failed %d at %p\n",cudaStat,devPtr);
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
  CublasSafeCall(cudaStat = cublasSetMatrix(m,n,elemsize,a,lda,devPtr,ldb));
  CudaCheckError();
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
  CublasSafeCall(cudaStat = cublasGetMatrix(m,n,elemsize,devPtr,lda,b,ldb));
  CudaCheckError();
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
  CublasSafeCall(cudaStat = cublasSetVector(n,elemsize,ptr,incx,devPtr,incy));
  CudaCheckError();
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
  CublasSafeCall(cudaStat = cublasGetVector(n,elemsize,devPtr,incx,ptr,incy));
  CudaCheckError();
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
  CublasSafeCall(cudaStat = cublasZher(handle,uplo,n,&alpha,x,incx,A,lda));
  CudaCheckError();
  if (cudaStat != cudaSuccess) {
    printf("cublasZher failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}


PyObject* cuZherk(PyObject *self, PyObject *args)
{
  cublasHandle_t handle;
  cublasFillMode_t uplo;
  cublasOperation_t trans = CUBLAS_OP_N;
  int n, k;
  const double alpha, beta;
  int lda, ldc;
  void *A,*C;

  if (!PyArg_ParseTuple(args,"LiiidLidLi",&handle,&uplo,&n,&k, &alpha,&A,&lda,
			&beta, &C, &ldc))
    return NULL;

  cudaError_t cudaStat;
  /*   printf("%p %d %d %f %p %d %p %d\n",handle,uplo,n,alpha,x,incx,A,lda); */
  CublasSafeCall(cudaStat = cublasZherk(handle,uplo,trans, n,k,&alpha,A,lda,&beta,C,ldc));
  CudaCheckError();
  if (cudaStat != cudaSuccess) {
    printf("cublasZherk failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}


PyObject* cuProfilerStart(PyObject *self, PyObject *args)
{
  cudaError_t cudaStat;
  CudaSafeCall(cudaStat = cudaProfilerStart());
  CudaCheckError();
  if (cudaStat != cudaSuccess) {
    printf("cudaProfilerStart failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}


PyObject* cuProfilerStop(PyObject *self, PyObject *args)
{
  cudaError_t cudaStat;
  CudaSafeCall(cudaStat = cudaProfilerStop());
  CudaCheckError();
  if (cudaStat != cudaSuccess) {
    printf("cudaProfilerStop failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}


PyObject* cuDevSynch(PyObject *self, PyObject *args)
{
  cudaError_t cudaStat;
  cudaStat = cudaDeviceSynchronize();
  if (cudaStat != cudaSuccess) {
    printf("cudaDevicesynchronize failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}


PyObject* cuDevReset(PyObject *self, PyObject *args)
{
  cudaError_t cudaStat;
  cudaStat = cudaDeviceReset();
  if (cudaStat != cudaSuccess) {
    printf("cudaDeviceReset failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}


PyObject* cuGetLastError(PyObject *self, PyObject *args)
{
  cudaError_t cudaStat;
  cudaStat = cudaGetLastError();
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
  CublasSafeCall(cudaStat = cublasCher(handle,uplo,n,&alpha,x,incx,A,lda));
  CudaCheckError();
  if (cudaStat != cudaSuccess) {
    printf("cublasZher failed %d\n",cudaStat);
    return NULL;
  }
  return Py_BuildValue("i",cudaStat);
}


PyObject* cuAdd(PyObject *self, PyObject *args)
{
  int n;
  void *a, *b, *c;

  if (!PyArg_ParseTuple(args, "LLLi",&a, &b, &c, &n))
    return NULL;
  cudaAdd(a, b, c, n);
  CudaCheckError();
  Py_RETURN_NONE;
}

PyObject* cuMul(PyObject *self, PyObject *args)
{
  int n;
  void *a, *b, *c;

  if (!PyArg_ParseTuple(args, "LLLi",&a, &b, &c, &n))
    return NULL;
  cudaMul(a, b, c, n);
  CudaCheckError();
  Py_RETURN_NONE;
}

PyObject* cuMulc(PyObject *self, PyObject *args)
{
  int n;
  void *a, *b, *c;

  if (!PyArg_ParseTuple(args, "LLLi",&a, &b, &c, &n))
    return NULL;
  cudaMulc(a, b, c, n);
  CudaCheckError();
  Py_RETURN_NONE;
}

PyObject* cuMap_G2Q(PyObject *self, PyObject *args)
{
  int n, nmultix, nG0;
  void *a, *b, *c;

  if (!PyArg_ParseTuple(args, "LLLiii",&a, &b, &c, &n, &nG0, &nmultix))
    return NULL;
  cudaMap_G2Q(a, b, c, n, nG0,  nmultix);
  CudaCheckError();
  Py_RETURN_NONE;
}

PyObject* cuMap_Q2G(PyObject *self, PyObject *args)
{
  int n, nmultix, nG0;
  void *a, *b, *c;

  if (!PyArg_ParseTuple(args, "LLLiii",&a, &b, &c, &n, &nG0, &nmultix))
    return NULL;
  cudaMap_Q2G(a, b, c, n, nG0, nmultix);
  CudaCheckError();
  Py_RETURN_NONE;
}

PyObject* cuDensity_matrix_R(PyObject *self, PyObject *args)
{
  int n, nmultix;
  void *a, *b, *c;

  if (!PyArg_ParseTuple(args, "LLLii",&a, &b, &c, &n, &nmultix))
    return NULL;
  cudaDensity_matrix_R(a, b, c, n, nmultix);
  CudaCheckError();
  Py_RETURN_NONE;
}

PyObject* cuOpt_phase(PyObject *self, PyObject *args)
{
  int n, nmultix, conjugate;
  void *a, *b, *c;

  if (!PyArg_ParseTuple(args, "LLLiii",&a, &b, &c, &n, &nmultix, &conjugate))
    return NULL;
  cudaOpt_phase(a, b, c, n, nmultix, conjugate);
  CudaCheckError();
  Py_RETURN_NONE;
}


PyObject* cuOpt_rhoG0_copy(PyObject *self, PyObject *args)
{
  int nG0, nmultix;
  void *rho_u, *rho_uG;

  if (!PyArg_ParseTuple(args, "LLii",&rho_u, &rho_uG, &nG0, &nmultix))
    return NULL;
  cudaOpt_rhoG0_copy(rho_u, rho_uG, nG0, nmultix);
  CudaCheckError();
  Py_RETURN_NONE;
}


PyObject* cuOpt_dE(PyObject *self, PyObject *args)
{
  int nG0, nmultix, s, k, n, nkpt, nband;
  void *rho_uG, *e_skn, *m_m;

  if (!PyArg_ParseTuple(args, "LiiLiiiLii",&rho_uG, &nG0, &nmultix, &e_skn, &s, &k, &n, &m_m, &nkpt, &nband))
    return NULL;
  cudaOpt_dE(rho_uG, nG0, nmultix, e_skn, s, k, n, m_m, nkpt, nband);
  CudaCheckError();
  Py_RETURN_NONE;
}

PyObject* cuGet_C_wu(PyObject *self, PyObject *args)
{
  int nmultix, s, k1, k2, n, nu, nkpt, nband, nw;
  void *e_skn, *m_u, *f_skn, *C_wu, *w_w, *alpha_wu;

  if (!PyArg_ParseTuple(args, "LLLLLiiiiLiiiii",&e_skn, &f_skn, &w_w, &C_wu, &alpha_wu, &s, &k1, &k2, &n, &m_u, &nu, &nmultix, &nkpt, &nband, &nw))
    return NULL;
  cudaC_wu( e_skn, f_skn, w_w, C_wu, alpha_wu,
                         s, k1, k2, n, m_u, nu, nmultix, nkpt, nband, nw);
  CudaCheckError();
  Py_RETURN_NONE;
}


PyObject* cuTrans_wfs(PyObject *self, PyObject *args)
{
  int n, nmultix;
  void *a, *b, *index;

  if (!PyArg_ParseTuple(args, "LLLii",&a, &b, &index, &n, &nmultix))
    return NULL;
  cudaTransform_wfs(a, b, index, n, nmultix);
  CudaCheckError();
  Py_RETURN_NONE;
}

PyObject* cuTrans_wfs_noindex(PyObject *self, PyObject *args)
{
  int n0, n1, n2;
  void *a, *b, *op_cc, *dk_c;

  if (!PyArg_ParseTuple(args, "LLLLiii",&a, &b, &op_cc, &dk_c, &n0, &n1, &n2))
    return NULL;
  cudaTransform_wfs_noindex(a, b, op_cc, dk_c, n0, n1, n2);
  CudaCheckError();
  Py_RETURN_NONE;
}

PyObject* cuConj_vector(PyObject *self, PyObject *args)
{
  int n;
  void *a;

  if (!PyArg_ParseTuple(args, "Li",&a, &n))
    return NULL;
  cudaConj(a, n);
  CudaCheckError();
  Py_RETURN_NONE;
}

PyObject* cuCopy_vector(PyObject *self, PyObject *args)
{
  int n;
  void *a, *b;

  if (!PyArg_ParseTuple(args, "LLi",&a, &b, &n))
    return NULL;
  cudaCopy(a, b, n);
  CudaCheckError();
  Py_RETURN_NONE;
}


PyObject* cuGet_P_ani(PyObject *self, PyObject *args)
{
  int Na, s, ik, n, NN;
  int time_rev;
  double *spos_ac, *ibzk_kc, **R_asii;
  int *op_scc, *a_sa, *Ni_a, *n_n, *offset;
  cuDoubleComplex **P_ani, **Pout_ani;
  
  if (!PyArg_ParseTuple(args, "LLLLLLLLiiiiLiLi",&spos_ac, &ibzk_kc, &op_scc, &a_sa, &R_asii, 
			&P_ani, &Pout_ani, &Ni_a, &time_rev, &Na, &s, &ik, &n_n, &n, &offset, &NN))
    return NULL;
  cudaP_ani(spos_ac, ibzk_kc, op_scc, a_sa, R_asii, P_ani, Pout_ani, Ni_a, time_rev, Na, s, ik, n_n, n, offset, NN);
  CudaCheckError();
  Py_RETURN_NONE;
}


PyObject* cuGet_P_ap(PyObject *self, PyObject *args)
{
  int Na, n, NN;
  cuDoubleComplex **P1_ai, **P2_aui, **P_aup;
  int *Ni_a, *offset;
  
  if (!PyArg_ParseTuple(args, "LLLLLiii",&P1_ai, &P2_aui, &P_aup, &Ni_a, &offset, &Na, &n, &NN))
    return NULL;
  cudaP_aup(P1_ai, P2_aui, P_aup, Ni_a, Na, n, offset, NN);
  CudaCheckError();
  Py_RETURN_NONE;
}


PyObject* cuGet_Q_anL(PyObject *self, PyObject *args)
{
  int mband, Na;
  cuDoubleComplex **P1_ami, **P2_ai, **Q_amL;
  double **Delta_apL;
  int *Ni_a, *nL_a;
  
  if (!PyArg_ParseTuple(args, "LLLLiiLL",&P1_ami, &P2_ai, &Delta_apL, &Q_amL, &mband, 
			&Na, &Ni_a, &nL_a))
    return NULL;
  cudaQ_anL(P1_ami, P2_ai, Delta_apL, Q_amL, mband, Na, Ni_a, nL_a);
  CudaCheckError();
  Py_RETURN_NONE;
}
