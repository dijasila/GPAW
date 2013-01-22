#include "cublas_v2.h" 
#include "complex.h"
#include <stdio.h>
#include <"math_constants.h">

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

__global__ void map_Q2G( cuDoubleComplex *a, cuDoubleComplex *b, int *c, int n ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) b[tid] = a[c[tid]];
}

__global__ void trans_wfs( cuDoubleComplex *a, cuDoubleComplex *b, int *index, cuDoubleComplex *phase, int n ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) b[index[tid]] = cuCmul(a[tid], phase[tid]);
}

__global__ void conj( cuDoubleComplex *a, int N ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) a[tid] = cuConj(a[tid]);
}

__global__ void copy( cuDoubleComplex *a, cuDoubleComplex *b, int N ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) b[tid] = a[tid];
}

__global__ void P_ai( double *spos1_c, double *spos2_c, double *k_c, int *op_cc, 
		      cuDoubleComplex *R_ii, cuDoubleComplex *Pin_i, cuDoubleComplex *Pout_i, int Ni){
  int tid = threadIdx.x;
  double complex x=0;
  double S_c[3] = [0,0,0];

  if (tid < 3){
    for (int dim=0; dim<3; dim++){
      S_c[tid] += spos1_c[dim] * op_cc[dim*3+tid] ;
      __syncthreads();
    }
    S_c[tid] -= spos2_c[tid];

    x += cos(2*CUDART_PI*S_c[tid] * k_c[tid]) + I * sin(2*CUDART_PI*S_c[tid] * k_c[tid]);
  }
  __syncthreads();

  if (tid < Ni){
    for (int j=0; j<3; j++){
      Pout_i[tid] += R_ii[tid*Nj+j] * Rin_i[j];
      _syncthreads();
    }
    Pout_i[tid] *= x;
  }


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
void cudaMap_Q2G( double complex* dev_a, double complex* dev_b, int* dev_c, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  map_Q2G<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, N);
}
}

extern "C" {
  void cudaTransform_wfs( double complex* dev_a, double complex* dev_b, int* dev_c, double complex* dev_d, int N ) {
    int threads = 128;
    int blocks = N/threads + (N%threads == 0 ? 0:1);
    trans_wfs<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, (cuDoubleComplex*)dev_d, N );
  }
}

extern "C" {
  void cudaConj( double complex* dev_a, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  conj<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, N);
}
}

extern "C" {
  void cudaCopy( double complex* dev_a, double complex* dev_b, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  copy<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, N);
}
}

extern "C" {
  void cudaP_ai( double* dev_spos_c, int* dev_op_cc, double* dev_S_c) {
  int threads = 128;
  int blocks = 1;
  P_ai<<<blocks, threads>>>( dev_spos_c, dev_op_cc, dev_S_c);
}
}



