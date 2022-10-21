#ifndef GPU_CUDA_H
#define GPU_CUDA_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define gpuMemcpyKind             cudaMemcpyKind
#define gpuMemcpyDeviceToHost     cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice     cudaMemcpyHostToDevice
#define gpuDeviceProp             cudaDeviceProp
#define gpuSuccess                cudaSuccess
#define gpuEventDefault           cudaEventDefault
#define gpuEventBlockingSync      cudaEventBlockingSync
#define gpuEventDisableTiming     cudaEventDisableTiming

#define gpuStream_t               cudaStream_t
#define gpuEvent_t                cudaEvent_t
#define gpuError_t                cudaError_t
#define gpublasStatus_t           cublasStatus_t

#define gpuDoubleComplex          cuDoubleComplex
#define make_gpuDoubleComplex     make_cuDoubleComplex
#define gpuCreal                  cuCreal
#define gpuCimag                  cuCimag
#define gpuCadd                   cuCadd
#define gpuCmul                   cuCmul
#define gpuConj                   cuConj

#define GPUBLAS_STATUS_SUCCESS           CUBLAS_STATUS_SUCCESS
#define GPUBLAS_STATUS_NOT_INITIALIZED   CUBLAS_STATUS_NOT_INITIALIZED
#define GPUBLAS_STATUS_ALLOC_FAILED      CUBLAS_STATUS_ALLOC_FAILED
#define GPUBLAS_STATUS_INVALID_VALUE     CUBLAS_STATUS_INVALID_VALUE
#define GPUBLAS_STATUS_ARCH_MISMATCH     CUBLAS_STATUS_ARCH_MISMATCH
#define GPUBLAS_STATUS_MAPPING_ERROR     CUBLAS_STATUS_MAPPING_ERROR
#define GPUBLAS_STATUS_EXECUTION_FAILED  CUBLAS_STATUS_EXECUTION_FAILED
#define GPUBLAS_STATUS_INTERNAL_ERROR    CUBLAS_STATUS_INTERNAL_ERROR

#define gpuCheckLastError()       gpuSafeCall(cudaGetLastError())
#define gpuGetErrorString(err)    cudaGetErrorString(err)

#define gpuSetDevice(id)          gpuSafeCall(cudaSetDevice(id))
#define gpuGetDevice(dev)         gpuSafeCall(cudaGetDevice(dev))
#define gpuGetDeviceProperties(prop, dev) \
        gpuSafeCall(cudaGetDeviceProperties(prop, dev))
#define gpuDeviceSynchronize()    gpuSafeCall(cudaDeviceSynchronize())

#define gpuFree(p)                if ((p) != NULL) gpuSafeCall(cudaFree(p))
#define gpuFreeHost(p)            if ((p) != NULL) gpuSafeCall(cudaFreeHost(p))
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

#endif
