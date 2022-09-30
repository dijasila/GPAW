#ifndef GPU_GPU_H
#define GPU_GPU_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <float.h>
#include <Python.h>

#define GPU_BLOCKS_MIN            (16)
#define GPU_BLOCKS_MAX            (96)
#define GPU_PITCH                 (16)  /* in doubles */

#define GPU_ASYNC_SIZE            (8*1024)
#define GPU_RJOIN_SIZE            (16*1024)
#define GPU_SJOIN_SIZE            (16*1024)
#define GPU_RJOIN_SAME_SIZE       (96*1024)
#define GPU_SJOIN_SAME_SIZE       (96*1024)
#define GPU_OVERLAP_SIZE          (GPU_ASYNC_SIZE)

#define GPU_ERROR_ABS_TOL         (1e-13)
#define GPU_ERROR_ABS_TOL_EXCT    (DBL_EPSILON)

#define GPAW_BOUNDARY_NORMAL      (1<<(0))
#define GPAW_BOUNDARY_SKIP        (1<<(1))
#define GPAW_BOUNDARY_ONLY        (1<<(2))
#define GPAW_BOUNDARY_X0          (1<<(3))
#define GPAW_BOUNDARY_X1          (1<<(4))
#define GPAW_BOUNDARY_Y0          (1<<(5))
#define GPAW_BOUNDARY_Y1          (1<<(6))
#define GPAW_BOUNDARY_Z0          (1<<(7))
#define GPAW_BOUNDARY_Z1          (1<<(8))

#define gpuSafeCall(err)          __cudaSafeCall(err, __FILE__, __LINE__)
#define gpublasSafeCall(err)      __cublasSafeCall(err, __FILE__, __LINE__)
#define gpuCheckLastError()       gpuSafeCall(cudaGetLastError())

#define gpuMemcpyDeviceToHost     cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice     cudaMemcpyHostToDevice
#define gpuDeviceProp             cudaDeviceProp
#define gpuSuccess                cudaSuccess
#define gpuEventDefault           cudaEventDefault
#define gpuEventBlockingSync      cudaEventBlockingSync
#define gpuEventDisableTiming     cudaEventDisableTiming

#define gpuStream_t               cudaStream_t
#define gpuEvent_t                cudaEvent_t

#define gpuSetDevice(id)          gpuSafeCall(cudaSetDevice(id))
#define gpuGetDevice(dev)         gpuSafeCall(cudaGetDevice(dev))
#define gpuGetDeviceProperties(prop, dev) \
        gpuSafeCall(cudaGetDeviceProperties(prop, dev))
#define gpuDeviceSynchronize()    gpuSafeCall(cudaDeviceSynchronize())

#define gpuFree(p)                gpuSafeCall(cudaFree(p))
#define gpuFreeHost(p)            gpuSafeCall(cudaFreeHost(p))
#define gpuMalloc(pp, size)       gpuSafeCall(cudaMalloc((void**) (pp), size))
#define gpuHostAlloc(pp, size) \
        gpuSafeCall(cudaHostAlloc((void**) (pp), size, cudaHostAllocPortable))
#define gpuMemcpy(dst, src, count, kind) \
        gpuSafeCall(cudaMemcpy(dst, src, count, kind))
#define gpuMemcpyAsync(dst, src, count, kind, stream) \
        gpuSafeCall(cudaMemcpyAsync(dst, src, count, kind, stream))

#define gpuStreamCreate(stream)   gpuSafeCall(cudaStreamCreate(stream))
#define gpuStreamDestroy(stream)  gpuSafeCall(cudaStreamDestroy(stream))
#define gpuStreamWaitEvent(stream, event, flags) \
        gpuSafeCall(cudaStreamWaitEvent(stream, event, flags))
#define gpuStreamSynchronize(stream) \
        gpuSafeCall(cudaStreamSynchronize(stream))

#define gpuEventCreate(event)     gpuSafeCall(cudaEventCreate(event))
#define gpuEventCreateWithFlags(event, flags) \
        gpuSafeCall(cudaEventCreateWithFlags(event, flags))
#define gpuEventDestroy(event)    gpuSafeCall(cudaEventDestroy(event))
#define gpuEventQuery(event)      cudaEventQuery(event)
#define gpuEventRecord(event, stream) \
        gpuSafeCall(cudaEventRecord(event, stream))
#define gpuEventSynchronize(event) \
        gpuSafeCall(cudaEventSynchronize(event))
#define gpuEventElapsedTime(ms, start, end) \
        gpuSafeCall(cudaEventElapsedTime(ms, start, end))

#define NEXTPITCHDIV(n) \
        (((n) > 0) ? ((n) + GPU_PITCH - 1 - ((n) - 1) % GPU_PITCH) : 0)

#ifndef MAX
#  define MAX(a,b)  (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#  define MIN(a,b)  (((a) < (b)) ? (a) : (b))
#endif

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

#ifndef BMGS_H
typedef struct
{
    int ncoefs;
    double* coefs;
    long* offsets;
    long n[3];
    long j[3];
} bmgsstencil;
#endif

extern struct cudaDeviceProp _gpaw_cuda_dev_prop;
extern int _gpaw_cuda_dev;

static inline cudaError_t __cudaSafeCall(cudaError_t err,
                                         const char *file, int line)
{
    if (cudaSuccess != err) {
        char str[100];
        snprintf(str, 100, "%s(%i): Cuda error: %s.\n",
                 file, line, cudaGetErrorString(err));
        PyErr_SetString(PyExc_RuntimeError, str);
        fprintf(stderr, str);
    }
    return err;
}

static inline cublasStatus_t __cublasSafeCall(cublasStatus_t err,
                                              const char *file, int line)
{
    if (CUBLAS_STATUS_SUCCESS != err) {
        char str[100];
        switch (err) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                snprintf(str, 100,
                         "%s(%i): Cublas error: CUBLAS_STATUS_NOT_INITIALIZED.\n",
                         file, line);
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                snprintf(str, 100,
                         "%s(%i): Cublas error: CUBLAS_STATUS_ALLOC_FAILED.\n",
                         file, line);
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                snprintf(str, 100,
                         "%s(%i): Cublas error: CUBLAS_STATUS_INVALID_VALUE.\n",
                         file, line);
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                snprintf(str, 100,
                         "%s(%i): Cublas error: CUBLAS_STATUS_ARCH_MISMATCH.\n",
                         file, line);
                break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                snprintf(str, 100,
                         "%s(%i): Cublas error: CUBLAS_STATUS_MAPPING_ERROR.\n",
                         file, line);
                break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                snprintf(str, 100,
                         "%s(%i): Cublas error: CUBLAS_STATUS_EXECUTION_FAILED.\n",
                         file, line);
                break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                snprintf(str, 100,
                         "%s(%i): Cublas error: CUBLAS_STATUS_INTERNAL_ERROR.\n",
                         file, line);
                break;
            default:
                snprintf(str, 100, "%s(%i): Cublas error: Unknown error %X.\n",
                         file, line, err);
        }
        PyErr_SetString(PyExc_RuntimeError, str);
        fprintf(stderr, str);
    }
    return err;
}

static inline unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#endif
