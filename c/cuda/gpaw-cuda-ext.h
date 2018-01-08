#ifndef GPAW_CUDA_EXT_H
#define GPAW_CUDA_EXT_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#define gpaw_cudaSafeCall(err) __gpaw_cudaSafeCall(err, __FILE__, __LINE__)

static inline void __gpaw_cudaSafeCall(cudaError_t err, char *file, int line)
{
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : Cuda error : %s.\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

#endif
