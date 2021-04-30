#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>

#include "gpaw-cuda-int.h"
#include "gpaw-cuda-debug.h"

#ifndef CUGPAWCOMPLEX
#  define BLOCK_SIZEX 32
#  define BLOCK_SIZEY 16
#  define BLOCK_MAX 32
#  define GRID_MAX 65535
#  define BLOCK_TOTALMAX 512
#  define XDIV 4
#  define Tfunc launch_func

typedef void (*launch_func)(const double *, const int *,
                            double *, const int *, const int *, int,
                            cudaStream_t);
typedef void (*launch_funcz)(const cuDoubleComplex *, const int *,
                             cuDoubleComplex *, const int *, const int *, int,
                             cudaStream_t);

static unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

extern int gpaw_cuda_debug;

static int debug_size_in = 0;
static int debug_size_out = 0;
static double *debug_in_cpu;
static double *debug_in_gpu;
static double *debug_out_cpu;
static double *debug_out_gpu;

static void debug_allocate(int ng, int ng2, int blocks)
{
    debug_size_in = ng * blocks;
    debug_size_out = ng2 * blocks;

    debug_in_cpu = GPAW_MALLOC(double, debug_size_in);
    debug_in_gpu = GPAW_MALLOC(double, debug_size_in);
    debug_out_cpu = GPAW_MALLOC(double, debug_size_out);
    debug_out_gpu = GPAW_MALLOC(double, debug_size_out);
}

static void debug_deallocate()
{
    free(debug_in_cpu);
    free(debug_in_gpu);
    free(debug_out_cpu);
    free(debug_out_gpu);
    debug_size_in = 0;
    debug_size_out = 0;
}

static void debug_memcpy_pre(const double *in, double *out)
{
    GPAW_CUDAMEMCPY(debug_in_cpu, in, double, debug_size_in,
                    cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(debug_out_cpu, out, double, debug_size_out,
                    cudaMemcpyDeviceToHost);
}

static void debug_memcpy_post(const double *in, double *out)
{
    GPAW_CUDAMEMCPY(debug_in_gpu, in, double, debug_size_in,
                    cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(debug_out_gpu, out, double, debug_size_out,
                    cudaMemcpyDeviceToHost);
}
#else
#  undef Tfunc
#  define Tfunc launch_funcz
#endif

__global__ void Zcuda(bmgs_paste_cuda_kernel)(
        const double* a, const int3 c_sizea, double* b, const int3 c_sizeb,
        int blocks, int xdiv)
{
    int xx = gridDim.x / xdiv;
    int yy = gridDim.y / blocks;

    int blocksi = blockIdx.y / yy;
    int i1 = (blockIdx.y - blocksi * yy) * blockDim.y + threadIdx.y;

    int xind = blockIdx.x / xx;
    int i2 = (blockIdx.x - xind * xx) * blockDim.x + threadIdx.x;

    b += i2 + (i1 + (xind + blocksi * c_sizeb.x) * c_sizeb.y) * c_sizeb.z;
    a += i2 + (i1 + (xind + blocksi * c_sizea.x) * c_sizea.y) * c_sizea.z;

    while (xind < c_sizea.x) {
        if ((i2 < c_sizea.z) && (i1 < c_sizea.y)) {
            b[0] = a[0];
        }
        b += xdiv * c_sizeb.y * c_sizeb.z;
        a += xdiv * c_sizea.y * c_sizea.z;
        xind += xdiv;
    }
}

__global__ void Zcuda(bmgs_paste_zero_cuda_kernel)(
        const Tcuda* a, const int3 c_sizea, Tcuda* b, const int3 c_sizeb,
        const int3 c_startb, const int3 c_blocks_bc, int blocks)
{
    int xx = gridDim.x / XDIV;
    int yy = gridDim.y / blocks;

    int blocksi = blockIdx.y / yy;
    int i1bl = blockIdx.y - blocksi * yy;
    int i1tid = threadIdx.y;
    int i1 = i1bl * BLOCK_SIZEY + i1tid;

    int xind = blockIdx.x / xx;
    int i2bl = blockIdx.x - xind * xx;
    int i2tid = threadIdx.x;
    int i2 = i2bl * BLOCK_SIZEX + i2tid;

    int xlen = (c_sizea.x + XDIV - 1) / XDIV;
    int xstart = xind * xlen;
    int xend = MIN(xstart + xlen, c_sizea.x);

    b += c_sizeb.x * c_sizeb.y * c_sizeb.z * blocksi;
    a += c_sizea.x * c_sizea.y * c_sizea.z * blocksi;

    if (xind==0) {
        Tcuda *bb = b + i2 + i1 * c_sizeb.z;
#pragma unroll 3
        for (int i0=0; i0 < c_startb.x; i0++) {
            if ((i2 < c_sizeb.z) && (i1 < c_sizeb.y)) {
                bb[0] = MAKED(0);
            }
            bb += c_sizeb.y * c_sizeb.z;
        }
    }
    if (xind == XDIV - 1) {
        Tcuda *bb = b + (c_startb.x + c_sizea.x) * c_sizeb.y * c_sizeb.z
                  + i2 + i1 * c_sizeb.z;
#pragma unroll 3
        for (int i0 = c_startb.x + c_sizea.x; i0 < c_sizeb.x; i0++) {
            if ((i2 < c_sizeb.z) && (i1 < c_sizeb.y)) {
                bb[0] = MAKED(0);
            }
            bb += c_sizeb.y * c_sizeb.z;
        }
    }

    int i1blbc = gridDim.y / blocks - i1bl - 1;
    int i2blbc = gridDim.x / XDIV - i2bl - 1;

    if (i1blbc<c_blocks_bc.y || i2blbc<c_blocks_bc.z) {
        int i1bc = i1blbc * BLOCK_SIZEY + i1tid;
        int i2bc = i2blbc * BLOCK_SIZEX + i2tid;

        b += (c_startb.x + xstart) * c_sizeb.y * c_sizeb.z;
        for (int i0=xstart; i0 < xend; i0++) {
            if ((i1bc < c_startb.y) && (i2 < c_sizeb.z)) {
                b[i2 + i1bc * c_sizeb.z] = MAKED(0);
            }
            if ((i1bc + c_sizea.y + c_startb.y < c_sizeb.y)
                    && (i2 < c_sizeb.z)) {
                b[i2 + i1bc * c_sizeb.z
                  + (c_sizea.y + c_startb.y) * c_sizeb.z] = MAKED(0);
            }
            if ((i2bc < c_startb.z) && (i1 < c_sizeb.y)) {
                b[i2bc + i1 * c_sizeb.z] = MAKED(0);
            }
            if ((i2bc + c_sizea.z + c_startb.z < c_sizeb.z)
                    && (i1 < c_sizeb.y)) {
                b[i2bc + i1 * c_sizeb.z + c_sizea.z + c_startb.z] = MAKED(0);
            }
            b += c_sizeb.y * c_sizeb.z;
        }
    } else {
        b += c_startb.z + (c_startb.y + c_startb.x * c_sizeb.y) * c_sizeb.z;

        b += i2 + i1 * c_sizeb.z + xstart * c_sizeb.y * c_sizeb.z;
        a += i2 + i1 * c_sizea.z + xstart * c_sizea.y * c_sizea.z;
        for (int i0=xstart; i0 < xend; i0++) {
            if ((i2 < c_sizea.z) && (i1 < c_sizea.y)) {
                b[0] = a[0];
            }
            b += c_sizeb.y * c_sizeb.z;
            a += c_sizea.y * c_sizea.z;
        }
    }
}

extern "C" {
    void Zcuda(debug_bmgs_paste)(const int sizea[3], const int sizeb[3],
                                 const int startb[3], int blocks,
                                 int ng, int ng2, int zero)
    {
        for (int m=0; m < blocks; m++) {
            if (zero)
                memset(debug_out_cpu + m * ng2, 0, ng2 * sizeof(double));
#ifndef CUGPAWCOMPLEX
            bmgs_paste_cpu(debug_in_cpu + m * ng, sizea,
                           debug_out_cpu + m * ng2, sizeb,
                           startb);
#else
            bmgs_pastez_cpu(debug_in_cpu + m * ng, sizea,
                            debug_out_cpu + m * ng2, sizeb,
                            startb);
#endif
        }
        double in_err = 0;
        for (int i=0; i < debug_size_in; i++) {
            in_err = MAX(in_err, fabs(debug_in_cpu[i] - debug_in_gpu[i]));
        }
        double out_err = 0;
        for (int i=0; i < debug_size_out; i++) {
            out_err = MAX(out_err, fabs(debug_out_cpu[i] - debug_out_gpu[i]));
        }
        if (in_err > GPAW_CUDA_ABS_TOL_EXCT) {
            if (zero)
                fprintf(stderr, "Debug CUDA paste zero (in): error %g\n",
                        in_err);
            else
                fprintf(stderr, "Debug CUDA paste (in): error %g\n", in_err);
        }
        if (out_err > GPAW_CUDA_ABS_TOL_EXCT) {
            if (zero)
                fprintf(stderr, "Debug CUDA paste zero (out): error %g\n",
                        out_err);
            else
                fprintf(stderr, "Debug CUDA paste (out): error %g\n", out_err);
        }
    }

    static void Zcuda(_bmgs_paste_cuda_gpu)(
            const Tcuda* a, const int sizea[3],
            Tcuda* b, const int sizeb[3],
            const int startb[3], int blocks,
            cudaStream_t stream)
    {
        int3 hc_sizea, hc_sizeb;
        hc_sizea.x = sizea[0];
        hc_sizea.y = sizea[1];
        hc_sizea.z = sizea[2] * sizeof(Tcuda) / sizeof(double);
        hc_sizeb.x = sizeb[0];
        hc_sizeb.y = sizeb[1];
        hc_sizeb.z = sizeb[2] * sizeof(Tcuda) / sizeof(double);

        int blockx = MIN(nextPow2(hc_sizea.z), BLOCK_MAX);
        int blocky = MIN(MIN(nextPow2(hc_sizea.y), BLOCK_TOTALMAX / blockx),
                         BLOCK_MAX);
        dim3 dimBlock(blockx, blocky);
        int gridx = ((hc_sizea.z + dimBlock.x - 1) / dimBlock.x);
        int xdiv = MAX(1, MIN(hc_sizea.x, GRID_MAX / gridx));
        int gridy = blocks * ((hc_sizea.y + dimBlock.y - 1) / dimBlock.y);

        gridx = xdiv * gridx;
        dim3 dimGrid(gridx, gridy);
        b += startb[2] + (startb[1] + startb[0] * sizeb[1]) * sizeb[2];
        Zcuda(bmgs_paste_cuda_kernel)<<<dimGrid, dimBlock, 0, stream>>>
            ((double*) a, hc_sizea, (double*) b, hc_sizeb, blocks, xdiv);
        gpaw_cudaSafeCall(cudaGetLastError());
    }

    static void Zcuda(_bmgs_paste_zero_cuda_gpu)(
            const Tcuda* a, const int sizea[3],
            Tcuda* b, const int sizeb[3],
            const int startb[3], int blocks,
            cudaStream_t stream)
    {
        int3 bc_blocks;
        int3 hc_sizea, hc_sizeb, hc_startb;
        hc_sizea.x = sizea[0];
        hc_sizea.y = sizea[1];
        hc_sizea.z = sizea[2];
        hc_sizeb.x = sizeb[0];
        hc_sizeb.y = sizeb[1];
        hc_sizeb.z = sizeb[2];
        hc_startb.x = startb[0];
        hc_startb.y = startb[1];
        hc_startb.z = startb[2];

        bc_blocks.y = hc_sizeb.y - hc_sizea.y > 0
                    ? MAX((hc_sizeb.y - hc_sizea.y + BLOCK_SIZEY - 1)
                            / BLOCK_SIZEY, 1)
                    : 0;
        bc_blocks.z = hc_sizeb.z - hc_sizea.z > 0
                    ? MAX((hc_sizeb.z - hc_sizea.z + BLOCK_SIZEX - 1)
                            / BLOCK_SIZEX, 1)
                    : 0;

        int gridy = blocks * ((sizeb[1] + BLOCK_SIZEY - 1) / BLOCK_SIZEY
                              + bc_blocks.y);
        int gridx = XDIV * ((sizeb[2] + BLOCK_SIZEX - 1) / BLOCK_SIZEX
                            + bc_blocks.z);

        dim3 dimBlock(BLOCK_SIZEX, BLOCK_SIZEY);
        dim3 dimGrid(gridx, gridy);

        Zcuda(bmgs_paste_zero_cuda_kernel)<<<dimGrid, dimBlock, 0, stream>>>
            ((Tcuda*) a, hc_sizea, (Tcuda*) b, hc_sizeb, hc_startb,
             bc_blocks, blocks);
        gpaw_cudaSafeCall(cudaGetLastError());
    }

    void Zcuda(_bmgs_paste_launcher)(Tfunc function, int zero,
                                     const Tcuda* a, const int sizea[3],
                                     Tcuda* b, const int sizeb[3],
                                     const int startb[3], int blocks,
                                     cudaStream_t stream)
    {
        const double *in = (double *) a;
        double *out = (double *) b;

#ifndef CUGPAWCOMPLEX
        int ng = sizea[0] * sizea[1] * sizea[2];
        int ng2 = sizeb[0] * sizeb[1] * sizeb[2];
#else
        int ng = sizea[0] * sizea[1] * sizea[2] * 2;
        int ng2 = sizeb[0] * sizeb[1] * sizeb[2] * 2;
#endif
        if (gpaw_cuda_debug) {
            debug_allocate(ng, ng2, blocks);
            debug_memcpy_pre(in, out);
        }
        (*function)(a, sizea, b, sizeb, startb, blocks, stream);
        if (gpaw_cuda_debug) {
            debug_memcpy_post(in, out);
            Zcuda(debug_bmgs_paste)(sizea, sizeb, startb, blocks, ng, ng2,
                                    zero);
            debug_deallocate();
        }
    }

    void Zcuda(bmgs_paste_cuda_gpu)(const Tcuda* a, const int sizea[3],
                                    Tcuda* b, const int sizeb[3],
                                    const int startb[3], int blocks,
                                    cudaStream_t stream)
    {
        if (!(sizea[0] && sizea[1] && sizea[2]))
            return;
        Zcuda(_bmgs_paste_launcher)(
                &(Zcuda(_bmgs_paste_cuda_gpu)), 0,
                a, sizea, b, sizeb, startb, blocks, stream);
    }

    void Zcuda(bmgs_paste_zero_cuda_gpu)(const Tcuda* a, const int sizea[3],
                                         Tcuda* b, const int sizeb[3],
                                         const int startb[3], int blocks,
                                         cudaStream_t stream)
    {
        if (!(sizea[0] && sizea[1] && sizea[2]))
            return;
        Zcuda(_bmgs_paste_launcher)(
                &(Zcuda(_bmgs_paste_zero_cuda_gpu)), 1,
                a, sizea, b, sizeb, startb, blocks, stream);
    }
}

#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "paste-cuda.cu"

extern "C" {
    double bmgs_paste_cuda_cpu(const double* a, const int sizea[3],
                               double* b, const int sizeb[3],
                               const int startb[3])
    {
        double *adev, *bdev;
        struct timeval t0, t1;
        double flops;
        int asize = sizea[0] * sizea[1] * sizea[2];
        int bsize = sizeb[0] * sizeb[1] * sizeb[2];

        gpaw_cudaSafeCall(cudaMalloc(&adev, sizeof(double) * asize));
        gpaw_cudaSafeCall(cudaMalloc(&bdev, sizeof(double) * bsize));
        gpaw_cudaSafeCall(
                cudaMemcpy(adev, a, sizeof(double) * asize,
                           cudaMemcpyHostToDevice));

        gettimeofday(&t0, NULL);
        bmgs_paste_cuda_gpu(adev, sizea, bdev, sizeb, startb, 1, 0);
        cudaThreadSynchronize();
        gpaw_cudaSafeCall(cudaGetLastError());
        gettimeofday(&t1,NULL);

        gpaw_cudaSafeCall(
                cudaMemcpy(b, bdev, sizeof(double) * bsize,
                           cudaMemcpyDeviceToHost));
        gpaw_cudaSafeCall(cudaFree(adev));
        gpaw_cudaSafeCall(cudaFree(bdev));

        flops = t1.tv_sec * 1.0 + t1.tv_usec / 1000000.0 - t0.tv_sec * 1.0
              - t0.tv_usec / 1000000.0;
        return flops;
    }

    double bmgs_paste_zero_cuda_cpu(const double* a, const int sizea[3],
                                    double* b, const int sizeb[3],
                                    const int startb[3])
    {
        double *adev, *bdev;
        struct timeval t0, t1;
        double flops;
        int asize = sizea[0] * sizea[1] * sizea[2];
        int bsize = sizeb[0] * sizeb[1] * sizeb[2];

        gpaw_cudaSafeCall(cudaMalloc(&adev, sizeof(double) * asize));
        gpaw_cudaSafeCall(cudaMalloc(&bdev, sizeof(double) * bsize));
        gpaw_cudaSafeCall(
                cudaMemcpy(adev, a, sizeof(double) * asize,
                           cudaMemcpyHostToDevice));

        gettimeofday(&t0, NULL);
        bmgs_paste_zero_cuda_gpu(adev, sizea, bdev, sizeb, startb, 1, 0);
        cudaThreadSynchronize();
        gpaw_cudaSafeCall(cudaGetLastError());
        gettimeofday(&t1,NULL);

        gpaw_cudaSafeCall(
                cudaMemcpy(b, bdev, sizeof(double) * bsize,
                           cudaMemcpyDeviceToHost));
        gpaw_cudaSafeCall(cudaFree(adev));
        gpaw_cudaSafeCall(cudaFree(bdev));

        flops = t1.tv_sec * 1.0 + t1.tv_usec / 1000000.0 - t0.tv_sec * 1.0
              - t0.tv_usec / 1000000.0;
        return flops;
    }
}
#endif
