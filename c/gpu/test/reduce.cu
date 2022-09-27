#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>

extern "C" {
#include </usr/include/complex.h>
#include <Python.h>
    typedef double complex double_complex;
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "lfc.h"
#include "bmgs-cuda/bmgs-cuda.h"
}

#ifndef BMGSCOMPLEX
#define BLOCK_SIZEX 16
#define BLOCK_SIZEY 8
#include "cuda.h"
#include "cuda_runtime_api.h"

#define SIZ 99
#define MAX(a,b) ((a)>(b))?(a):(b)

static unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}
#endif // !BMGSCOMPLEX

__global__ void Zcuda(reduce3)(Tcuda *g_idata, Tcuda *g_odata, unsigned int n)
{
    extern __shared__ Tcuda Zcuda(sdata)[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    Tcuda mySum = (i < n) ? g_idata[i] : MAKED(0);
    if (i + blockDim.x < n)
        IADD(mySum, g_idata[i + blockDim.x]);

    Zcuda(sdata)[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            Zcuda(sdata)[tid] = mySum = ADD(mySum, Zcuda(sdata)[tid + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = Zcuda(sdata)[0];
}


#ifndef BMGSCOMPLEX
#define BMGSCOMPLEX
#include "reduce.cu"

int main(void)
{
    double complex *a, *b;
    cuDoubleComplex *a_gpu, *b_gpu;
    struct timeval t0, t1;

    srand((unsigned int) time(NULL));

    a = (double complex*) malloc(SIZ * sizeof(double complex));
    b = (double complex*) malloc(SIZ * sizeof(double complex));

    for (int i=0; i < SIZ; i++) {
        a[i] = rand() + rand() * I;
        b[i] = a[i];
    }

    for (int i=0; i < SIZ; i++) {
        fprintf(stdout, "%f ", a[i]);
    }
    fprintf(stdout, "\n");

    cudaMalloc(&a_gpu, SIZ * sizeof(cuDoubleComplex));
    cudaMemcpy(a_gpu, a, SIZ * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);

    cudaMalloc(&b_gpu, SIZ * sizeof(cuDoubleComplex));
    cudaMemcpy(b_gpu, b, SIZ * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);

    int threads = 64;
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(cuDoubleComplex)
                                   : threads * sizeof(cuDoubleComplex);

    gettimeofday(&t0, NULL);
    cudaThreadSynchronize();
    for (int i=0; i < 1; i++) {
        int iter = SIZ;
        while (iter > 1) {
            dim3 dimBlock(threads, 1, 1);
            dim3 dimGrid(MAX(iter / threads, 1), 1, 1);
            reduce3z<<<dimGrid, dimBlock, smemSize>>>(b_gpu, b_gpu, iter);
            iter = nextPow2(iter) / (threads * 2);
            cudaMemcpy(b, b_gpu, sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToHost);
            for (int i=0; i < 1; i++) {
                fprintf(stdout, "%f ", b[i]);
            }
            fprintf(stdout, "\n");
        }
    }
    cudaThreadSynchronize();
    gettimeofday(&t1, NULL);

    cudaMemcpy(b, b_gpu, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    for (int i=0; i < 1; i++) {
        fprintf(stdout, "%f ", b[i]);
    }
    fprintf(stdout, "\n");

    double complex cc = 0 + 0 * I;
    for (int ii=0; ii < SIZ; ii++)
        cc += a[ii];

    fprintf(stdout, "sum %f %f %f\n", creal(cc), creal(b[0]), creal(cc-b[0]));
    fprintf(stdout, "sumi %f %f %f\n", cimag(cc), cimag(b[0]), cimag(cc-b[0]));

    double flops = t1.tv_sec * 1.0 + t1.tv_usec / 1000000.0
                 - t0.tv_sec * 1.0 - t0.tv_usec / 1000000.0;
    fprintf(stdout, "time %g ms\n", flops * 1000);
}
#endif
