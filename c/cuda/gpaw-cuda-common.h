#ifndef GPAW_CUDA_EXT_H
#define GPAW_CUDA_EXT_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

#define GPAW_CUDA_BLOCKS  8

typedef struct
{
  int ncoefs;
  double* coefs_gpu;
  long* offsets_gpu;
  int ncoefs0;
  double* coefs0_gpu;
  int* offsets0_gpu;
  int ncoefs12;
  double* coefs12_gpu;
  int* offsets12_gpu;
  long n[3];
  long j[3];
} bmgsstencil_gpu;


#define gpaw_cudaSafeCall(err)   __gpaw_cudaSafeCall(err,__FILE__,__LINE__)

static inline void __gpaw_cudaSafeCall( cudaError_t err ,char *file,int line)
{
  if( cudaSuccess != err) {
    fprintf(stderr, "%s(%i): Cuda error: %s.\n",
	    file, line, cudaGetErrorString( err) );
    //    exit(-1);
  }
}


#define gpaw_cublasSafeCall(err)   __gpaw_cublasSafeCall(err,__FILE__,__LINE__)

static inline void __gpaw_cublasSafeCall( cublasStatus err ,char *file,int line)
{
  if( CUBLAS_STATUS_SUCCESS != err) {
    fprintf(stderr, "%s(%i): Cublas error: %X.\n",
	    file, line, err);
    //    exit(-1);
  }
}

#define GPAW_CUDAMALLOC(pp,T,n) gpaw_cudaSafeCall(cudaMalloc((void**)(pp),sizeof(T)*(n)));

#define GPAW_CUDAMEMCPY(p1,p2,T,n,type) gpaw_cudaSafeCall(cudaMemcpy(p1,p2,sizeof(T)*n,type));

#define GPAW_CUDAMALLOC_HOST(pp,T,n) gpaw_cudaSafeCall(cudaHostAlloc((void**)(pp),sizeof(T)*(n),cudaHostAllocDefault));

#endif
