#include "cublas_v2.h" 
#include "complex.h"
#include <stdio.h>

__global__ void add( cuDoubleComplex* a, cuDoubleComplex* b, cuDoubleComplex* c, int N ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N)
    c[tid] = cuCadd(a[tid], b[tid]);
}

__global__ void mul( cuDoubleComplex *a, cuDoubleComplex *b, cuDoubleComplex *c, int N ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) c[tid] = cuCmul(a[tid], b[tid]);
}

__global__ void mulc( cuDoubleComplex *a, cuDoubleComplex *b, cuDoubleComplex *c, int N ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) c[tid] = cuCmul(cuConj(a[tid]), b[tid]);
}

__global__ void map_G2Q( cuDoubleComplex *a, cuDoubleComplex *b, int *c, int n ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) b[c[tid]] = a[tid];
}

__global__ void trans_wfs( cuDoubleComplex *a, cuDoubleComplex *b, int *index, cuDoubleComplex *phase, int n ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) b[index[tid]] = cuCmul(a[tid], phase[tid]);
}




extern "C" {
void cudaAdd( double complex* dev_a, double complex* dev_b, double complex* dev_c, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  add<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (cuDoubleComplex*)dev_c, N);
}
}

extern "C" {
void cudaMul( double complex* dev_a, double complex* dev_b, double complex* dev_c, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  mul<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (cuDoubleComplex*)dev_c, N);
}
}

extern "C" {
void cudaMulc( double complex* dev_a, double complex* dev_b, double complex* dev_c, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  mulc<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (cuDoubleComplex*)dev_c, N);
}
}

extern "C" {
void cudaMap_G2Q( double complex* dev_a, double complex* dev_b, int* dev_c, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  map_G2Q<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, N);
}
}

extern "C" {
  void cudaTransform_wfs( double complex* dev_a, double complex* dev_b, int* dev_c, double complex* dev_d, int N ) {
    int threads = 128;
    int blocks = N/threads + (N%threads == 0 ? 0:1);
    trans_wfs<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, (cuDoubleComplex*)dev_d, N );
  }
}
