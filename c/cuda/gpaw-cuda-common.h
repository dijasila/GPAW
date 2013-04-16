#ifndef GPAW_CUDA_EXT_H
#define GPAW_CUDA_EXT_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <float.h>
#include <Python.h>

//#define DEBUG_CUDA 

#define GPAW_CUDA_BLOCKS_MIN  (16)
#define GPAW_CUDA_BLOCKS_MAX  (96)
#define GPAW_CUDA_PITCH       (16)  // In doubles
#define GPAW_CUDA_ASYNC_SIZE  (8*1024)
#define GPAW_CUDA_RJOIN_SIZE  (16*1024)
#define GPAW_CUDA_SJOIN_SIZE  (16*1024)
#define GPAW_CUDA_RJOIN_SAME_SIZE  (96*1024)
#define GPAW_CUDA_SJOIN_SAME_SIZE  (96*1024)
#define GPAW_CUDA_OVERLAP_SIZE  (GPAW_CUDA_ASYNC_SIZE)
#define GPAW_CUDA_ABS_TOL       (1e-13)
#define GPAW_CUDA_ABS_TOL_EXCT  (DBL_EPSILON)


#define GPAW_BOUNDARY_NORMAL  (1<<(0))
#define GPAW_BOUNDARY_SKIP    (1<<(1))
#define GPAW_BOUNDARY_ONLY    (1<<(2))
#define GPAW_BOUNDARY_X0      (1<<(3))
#define GPAW_BOUNDARY_X1      (1<<(4))
#define GPAW_BOUNDARY_Y0      (1<<(5))
#define GPAW_BOUNDARY_Y1      (1<<(6))
#define GPAW_BOUNDARY_Z0      (1<<(7))
#define GPAW_BOUNDARY_Z1      (1<<(8))


typedef struct
{
  int ncoefs;
  double* coefs_gpu;
  long* offsets_gpu;
  int ncoefs0;
  double* coefs0_gpu;
  int ncoefs1;
  double* coefs1_gpu;
  int ncoefs2;
  double* coefs2_gpu;
  double coef_relax;
  long n[3];
  long j[3];
} bmgsstencil_gpu;


#define gpaw_cudaSafeCall(err)   __gpaw_cudaSafeCall(err,__FILE__,__LINE__)

static inline cudaError_t __gpaw_cudaSafeCall( cudaError_t err ,char *file,int line)
{
  if( cudaSuccess != err) {
    char str[100];
    snprintf(str,100,"%s(%i): Cuda error: %s.\n",
	     file, line, cudaGetErrorString( err));
    PyErr_SetString(PyExc_RuntimeError,str);
    fprintf(stderr, str);
  }
  return err;
}


#define gpaw_cublasSafeCall(err)   __gpaw_cublasSafeCall(err,__FILE__,__LINE__)

static inline cublasStatus __gpaw_cublasSafeCall( cublasStatus err ,char *file,int line)
{
  if( CUBLAS_STATUS_SUCCESS != err) {
    char str[100];
    snprintf(str,100,"%s(%i): Cublas error: %X.\n",
	     file, line, err);
    PyErr_SetString(PyExc_RuntimeError,str);
    fprintf(stderr,str);
  }
  return err;
}



#define GPAW_CUDAMALLOC(pp,T,n) gpaw_cudaSafeCall(cudaMalloc((void**)(pp),sizeof(T)*(n)));

#define GPAW_CUDAMEMCPY(p1,p2,T,n,type) gpaw_cudaSafeCall(cudaMemcpy(p1,p2,sizeof(T)*(n),type));

#define GPAW_CUDAMEMCPY_A(p1,p2,T,n,type,stream) gpaw_cudaSafeCall(cudaMemcpyAsync(p1,p2,sizeof(T)*(n),type,stream));

#define GPAW_CUDAMALLOC_HOST(pp,T,n) gpaw_cudaSafeCall(cudaHostAlloc((void**)(pp),sizeof(T)*(n),cudaHostAllocPortable));


#define NEXTPITCHDIV(n)  (((n)>0) ? ((n)+(GPAW_CUDA_PITCH)-1-((n)-1)%(GPAW_CUDA_PITCH)) : 0)

#endif
