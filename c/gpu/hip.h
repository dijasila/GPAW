#ifndef GPU_HIP_H
#define GPU_HIP_H

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#define gpuMemcpyKind             hipMemcpyKind
#define gpuMemcpyDeviceToHost     hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice     hipMemcpyHostToDevice
#define gpuSuccess                hipSuccess
#define gpuEventDefault           hipEventDefault
#define gpuEventBlockingSync      hipEventBlockingSync
#define gpuEventDisableTiming     hipEventDisableTiming

#define gpuStream_t               hipStream_t
#define gpuEvent_t                hipEvent_t
#define gpuError_t                hipError_t
#define gpuDeviceProp             hipDeviceProp_t

#define gpuDoubleComplex          hipDoubleComplex
#define gpublasDoubleComplex      hipblasDoubleComplex
#define make_gpuDoubleComplex     make_hipDoubleComplex
#define gpuCreal                  hipCreal
#define gpuCimag                  hipCimag
#define gpuCadd                   hipCadd
#define gpuCsub                   hipCsub
#define gpuCmul                   hipCmul
#define gpuConj                   hipConj

#define gpuCheckLastError()       gpuSafeCall(hipGetLastError())
#define gpuGetErrorString(err)    hipGetErrorString(err)

#define gpuSetDevice(id)          gpuSafeCall(hipSetDevice(id))
#define gpuGetDevice(dev)         gpuSafeCall(hipGetDevice(dev))
#define gpuGetDeviceProperties(prop, dev) \
        gpuSafeCall(hipGetDeviceProperties(prop, dev))
#define gpuDeviceSynchronize()    gpuSafeCall(hipDeviceSynchronize())

#define gpuFree(p)                if ((p) != NULL) gpuSafeCall(hipFree(p))
#define gpuFreeHost(p)            if ((p) != NULL) gpuSafeCall(hipHostFree(p))
#define gpuMalloc(pp, size)       gpuSafeCall(hipMalloc((void**) (pp), size))
#define gpuHostAlloc(pp, size) \
        gpuSafeCall(hipHostMalloc((void**) (pp), size, hipHostMallocPortable))
#define gpuMemcpy(dst, src, count, kind) \
        gpuSafeCall(hipMemcpy(dst, src, count, kind))
#define gpuMemcpyAsync(dst, src, count, kind, stream) \
        gpuSafeCall(hipMemcpyAsync(dst, src, count, kind, stream))

#define gpuStreamCreate(stream)   gpuSafeCall(hipStreamCreate(stream))
#define gpuStreamDestroy(stream)  gpuSafeCall(hipStreamDestroy(stream))
#define gpuStreamWaitEvent(stream, event, flags) \
        gpuSafeCall(hipStreamWaitEvent(stream, event, flags))
#define gpuStreamSynchronize(stream) \
        gpuSafeCall(hipStreamSynchronize(stream))

#define gpuEventCreate(event)     gpuSafeCall(hipEventCreate(event))
#define gpuEventCreateWithFlags(event, flags) \
        gpuSafeCall(hipEventCreateWithFlags(event, flags))
#define gpuEventDestroy(event)    gpuSafeCall(hipEventDestroy(event))
#define gpuEventQuery(event)      hipEventQuery(event)
#define gpuEventRecord(event, stream) \
        gpuSafeCall(hipEventRecord(event, stream))
#define gpuEventSynchronize(event) \
        gpuSafeCall(hipEventSynchronize(event))
#define gpuEventElapsedTime(ms, start, end) \
        gpuSafeCall(hipEventElapsedTime(ms, start, end))

#define gpuLaunchKernel(kernel, dimGrid, dimBlock, shared, stream, ...) \
        kernel<<<dimGrid, dimBlock, shared, stream>>>(__VA_ARGS__)

#define gpublasStatus_t           hipblasStatus_t
#define gpublasHandle_t           hipblasHandle_t
#define gpublasOperation_t        hipblasOperation_t

#define gpublasCreate             hipblasCreate
#define gpublasSetStream          hipblasSetStream
#define gpublasGetMatrixAsync     hipblasGetMatrixAsync
#define gpublasSetMatrixAsync     hipblasSetMatrixAsync
#define gpublasDsyrk              hipblasDsyrk
#define gpublasDsyr2k             hipblasDsyr2k
#define gpublasDscal              hipblasDscal
#define gpublasZscal              hipblasZscal
#define gpublasDgemm              hipblasDgemm
#define gpublasZgemm              hipblasZgemm
#define gpublasDgemv              hipblasDgemv
#define gpublasZgemv              hipblasZgemv
#define gpublasDaxpy              hipblasDaxpy
#define gpublasZaxpy              hipblasZaxpy
#define gpublasZherk              hipblasZherk
#define gpublasZher2k             hipblasZher2k
#define gpublasDdot               hipblasDdot
#define gpublasZdotc              hipblasZdotc
#define gpublasZdotu              hipblasZdotu

#define GPUBLAS_OP_N                     HIPBLAS_OP_N
#define GPUBLAS_OP_T                     HIPBLAS_OP_T
#define GPUBLAS_OP_C                     HIPBLAS_OP_C
#define GPUBLAS_FILL_MODE_UPPER          HIPBLAS_FILL_MODE_UPPER
#define GPUBLAS_STATUS_SUCCESS           HIPBLAS_STATUS_SUCCESS
#define GPUBLAS_STATUS_NOT_INITIALIZED   HIPBLAS_STATUS_NOT_INITIALIZED
#define GPUBLAS_STATUS_ALLOC_FAILED      HIPBLAS_STATUS_ALLOC_FAILED
#define GPUBLAS_STATUS_INVALID_VALUE     HIPBLAS_STATUS_INVALID_VALUE
#define GPUBLAS_STATUS_ARCH_MISMATCH     HIPBLAS_STATUS_ARCH_MISMATCH
#define GPUBLAS_STATUS_MAPPING_ERROR     HIPBLAS_STATUS_MAPPING_ERROR
#define GPUBLAS_STATUS_EXECUTION_FAILED  HIPBLAS_STATUS_EXECUTION_FAILED
#define GPUBLAS_STATUS_INTERNAL_ERROR    HIPBLAS_STATUS_INTERNAL_ERROR

#endif
