#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <string.h>

#include "gpaw-cuda-int.h"
#include "gpaw-cuda-debug.h"

#ifndef CUGPAWCOMPLEX
#define BLOCK_MAX 32
#define GRID_MAX 65535
#define BLOCK_TOTALMAX 256

extern int gpaw_cuda_debug;

static int debug_size_in = 0;
static int debug_size_out = 0;
static double *debug_in_cpu;
static double *debug_in_gpu;
static double *debug_out_cpu;
static double *debug_out_gpu;

/*
 * Allocate debug buffers and precalculate sizes.
 */
static void debug_allocate(int ng, int ng2, int blocks)
{
    debug_size_in = ng * blocks;
    debug_size_out = ng2 * blocks;

    debug_in_cpu = GPAW_MALLOC(double, debug_size_in);
    debug_in_gpu = GPAW_MALLOC(double, debug_size_in);
    debug_out_cpu = GPAW_MALLOC(double, debug_size_out);
    debug_out_gpu = GPAW_MALLOC(double, debug_size_out);
}

/*
 * Deallocate debug buffers and set sizes to zero.
 */
static void debug_deallocate()
{
    free(debug_in_cpu);
    free(debug_in_gpu);
    free(debug_out_cpu);
    free(debug_out_gpu);
    debug_size_in = 0;
    debug_size_out = 0;
}

/*
 * Copy initial GPU arrays to debug buffers on the CPU.
 */
static void debug_memcpy_pre(const double *in, double *out)
{
    GPAW_CUDAMEMCPY(debug_in_cpu, in, double, debug_size_in,
                    cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(debug_out_cpu, out, double, debug_size_out,
                    cudaMemcpyDeviceToHost);
}

/*
 * Copy final GPU arrays to debug buffers on the CPU.
 */
static void debug_memcpy_post(const double *in, double *out)
{
    GPAW_CUDAMEMCPY(debug_in_gpu, in, double, debug_size_in,
                    cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(debug_out_gpu, out, double, debug_size_out,
                    cudaMemcpyDeviceToHost);
}
#endif

/*
 * Copy a slice of an array on the CPU and compare to results from the GPU.
 */
extern "C"
static void Zcuda(debug_bmgs_cut)(
        const int sizea[3], const int starta[3], const int sizeb[3],
#ifdef CUGPAWCOMPLEX
        cuDoubleComplex phase,
#endif
        int blocks, int ng, int ng2)
{
    for (int m=0; m < blocks; m++) {
#ifndef CUGPAWCOMPLEX
        bmgs_cut_cpu(debug_in_cpu + m * ng, sizea, starta,
                     debug_out_cpu + m * ng2, sizeb);
#else
        bmgs_cutmz_cpu(debug_in_cpu + m * ng, sizea, starta,
                       debug_out_cpu + m * ng2, sizeb, (void *) &phase);
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
        fprintf(stderr, "Debug CUDA cut (in): error %g\n", in_err);
    }
    if (out_err > GPAW_CUDA_ABS_TOL_EXCT) {
        fprintf(stderr, "Debug CUDA cut (out): error %g\n", out_err);
    }
}

/*
 * CUDA kernel to copy a slice of an array.
 */
__global__ void Zcuda(bmgs_cut_cuda_kernel)(
        const Tcuda* a, const int3 c_sizea, Tcuda* b, const int3 c_sizeb,
#ifdef CUGPAWCOMPLEX
        cuDoubleComplex phase,
#endif
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

    while (xind < c_sizeb.x) {
        if ((i2 < c_sizeb.z) && (i1 < c_sizeb.y)) {
#ifndef CUGPAWCOMPLEX
            b[0] = a[0];
#else
            b[0] = MULTT(phase, a[0]);
#endif
        }
        b += xdiv * c_sizeb.y * c_sizeb.z;
        a += xdiv * c_sizea.y * c_sizea.z;
        xind += xdiv;
    }
}

/*
 * Launch CUDA kernel to copy a slice of an array on the GPU.
 */
extern "C"
static void Zcuda(_bmgs_cut_cuda_gpu)(
        const Tcuda* a, const int sizea[3], const int starta[3],
        Tcuda* b, const int sizeb[3],
#ifdef CUGPAWCOMPLEX
        cuDoubleComplex phase,
#endif
        int blocks, cudaStream_t stream)
{
    int3 hc_sizea, hc_sizeb;
    hc_sizea.x = sizea[0];
    hc_sizea.y = sizea[1];
    hc_sizea.z = sizea[2];
    hc_sizeb.x = sizeb[0];
    hc_sizeb.y = sizeb[1];
    hc_sizeb.z = sizeb[2];

    int blockx = MIN(nextPow2(hc_sizeb.z), BLOCK_MAX);
    int blocky = MIN(
            MIN(nextPow2(hc_sizeb.y), BLOCK_TOTALMAX / blockx),
            BLOCK_MAX);
    dim3 dimBlock(blockx, blocky);
    int gridx = ((hc_sizeb.z + dimBlock.x - 1) / dimBlock.x);
    int xdiv = MAX(1, MIN(hc_sizeb.x, GRID_MAX / gridx));
    int gridy = blocks * ((hc_sizeb.y + dimBlock.y - 1) / dimBlock.y);

    gridx = xdiv * gridx;
    dim3 dimGrid(gridx, gridy);

    a += starta[2] + (starta[1] + starta[0] * hc_sizea.y) * hc_sizea.z;

    Zcuda(bmgs_cut_cuda_kernel)<<<dimGrid, dimBlock, 0, stream>>>(
            (Tcuda*) a, hc_sizea, (Tcuda*) b, hc_sizeb,
#ifdef CUGPAWCOMPLEX
         phase,
#endif
         blocks, xdiv);
    gpaw_cudaSafeCall(cudaGetLastError());
}

/*
 * Copy a slice of an array on the GPU. If the array contains complex
 * numbers, then multiply each element with the given phase.
 *
 * For example:
 *       . . . .               (OR for complex numbers)
 *   a = . 1 2 . -> b = 1 2     -> b = phase*1 phase*2
 *       . 3 4 .        3 4            phase*3 phase*4
 *       . . . .
 *
 * arguments:
 *   a      -- input array
 *   sizea  -- dimensions of the array a
 *   starta -- offset to the start of the slice
 *   b      -- output array
 *   sizeb  -- dimensions of the array b
 *   phase  -- phase (only for complex)
 *   blocks -- number of blocks
 *   stream -- CUDA stream to use
 */
extern "C"
void Zcuda(bmgs_cut_cuda_gpu)(
        const Tcuda* a, const int sizea[3], const int starta[3],
        Tcuda* b, const int sizeb[3],
#ifdef CUGPAWCOMPLEX
        cuDoubleComplex phase,
#endif
        int blocks, cudaStream_t stream)
{
    if (!(sizea[0] && sizea[1] && sizea[2]))
        return;
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
#ifndef CUGPAWCOMPLEX
    _bmgs_cut_cuda_gpu(a, sizea, starta, b, sizeb, blocks, stream);
#else
    _bmgs_cut_cuda_gpuz(a, sizea, starta, b, sizeb, phase, blocks, stream);
#endif
    if (gpaw_cuda_debug) {
        debug_memcpy_post(in, out);
#ifndef CUGPAWCOMPLEX
        debug_bmgs_cut(sizea, starta, sizeb, blocks, ng, ng2);
#else
        debug_bmgs_cutz(sizea, starta, sizeb, phase, blocks, ng, ng2);
#endif
        debug_deallocate();
    }
}

#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "cut-cuda.cu"
#endif
