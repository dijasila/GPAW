#include "../gpu.h"
#include "../gpu-complex.h"
#include "numpy/arrayobject.h"
#include "assert.h"

__global__ void pw_insert_many_16(int nb,
                                  int nG,
                                  int nQ,
                                  gpuDoubleComplex* c_nG,
                                  npy_int32* Q_G,
                                  double scale,
                                  gpuDoubleComplex* tmp_nQ)
{
    int G = threadIdx.x + blockIdx.x * blockDim.x;
    int b = threadIdx.y + blockIdx.y * blockDim.y;
    __shared__ npy_int32 locQ_G[16];
    if (threadIdx.y == 0)
        locQ_G[threadIdx.x] = Q_G[G];
    __syncthreads();
    
    if ((G < nG) && (b < nb))
    {
        npy_int32 Q = locQ_G[threadIdx.x];
        tmp_nQ[Q + b * nQ] = gpuCmulD(c_nG[G + b * nG], scale);
    }
}

__global__ void add_to_density_16(int nb,
                                  int nR,
                                  double* f_n,
                                  gpuDoubleComplex* psit_nR,
                                  double* rho_R)
{
    //int b = threadIdx.x + blockIdx.x * blockDim.x;
    int R = threadIdx.x + blockIdx.x * blockDim.x;
    double rho = 0.0;
    for (int b=0; b< nb; b++)
    {
        int idx = b * nR + R;
        rho += f_n[b] * (psit_nR[idx].x * psit_nR[idx].x + psit_nR[idx].y * psit_nR[idx].y);
    }
    rho_R[R] = rho;
}


__global__ void pw_insert_16(int nG,
                             int nQ,
                             gpuDoubleComplex* c_G,
                             npy_int32* Q_G,
                             double scale,
                             gpuDoubleComplex* tmp_Q)
{
    int G = threadIdx.x + blockIdx.x * blockDim.x;
    if (G < nG)
        tmp_Q[Q_G[G]] = gpuCmulD(c_G[G], scale);
}

/*
__global__ void _pw_insert_8(int nG,
                             int nQ,
                             double* c_G,
                             npy_int32* Q_G,
                             double scale,
                             double* tmp_Q)
{
    int G = threadIdx.x + blockIdx.x * blockDim.x;
    if (G < nG)
        tmp_Q[Q_G[G]] = c_G[G] * scale;
}

__global__ void _pw_insert_8_many(int nb,
                                  int nG,
                                  int nQ,
                                  double* c_G,
                                  npy_int32* Q_G,
                                  double scale,
                                  double* tmp_Q)
{
    int G = threadIdx.x + blockIdx.x * blockDim.x;
    int b = threadIdx.y + blockIdx.y * blockDim.y;
    __shared__ double locQ_G[16];
    if (threadIdx.y == 0)
        locQ_G[threadIdx.x] = c_G[G];
    __syncthreads();
    
    if ((G < nG) && (b < nb))
    {
        npy_int32 Q = locQ_G[threadIdx.x];
        tmp_Q[Q + b * nQ] = c_G[G + b * nG] * scale;
    }
}
*/

extern "C"
void add_to_density_gpu_launch_kernel(int nb,
                                      int nR,
                                      double* f_n,
                                      gpuDoubleComplex* psit_nR,
                                      double* rho_R)
{
    gpuLaunchKernel(add_to_density_16,
                    dim3((nR+255)/256),
                    dim3(256),
                    0, 0,
                    nb, nR,
                    f_n,
                    psit_nR,
                    rho_R);
}

extern "C"
void pw_insert_gpu_launch_kernel(int itemsize,
                             int nb,
                             int nG,
                             int nQ,
                             double* c_nG,
                             npy_int32* Q_G,
                             double scale,
                             double* tmp_nQ)
{
    if (itemsize == 16)
    {
        if (nb == 1)
        {
           gpuLaunchKernel(pw_insert_16,
                           dim3((nG+15)/16, (nb+15)/16),
                           dim3(16, 16),
                           0, 0,
                           nG, nQ,
                           (gpuDoubleComplex*) c_nG, Q_G,
                           scale,
                           (gpuDoubleComplex*) tmp_nQ);
        }
        else
        {
           gpuLaunchKernel(pw_insert_many_16,
                           dim3((nG+15)/16, (nb+15)/16),
                           dim3(16, 16),
                           0, 0,
                           nb, nG, nQ,
                           (gpuDoubleComplex*) c_nG,
                           Q_G,
                           scale,
                           (gpuDoubleComplex*) tmp_nQ);
        }
    }
    else
    {
        assert(0);
    }
}


__global__ void pwlfc_expand_kernel_8(double* f_Gs,
                                       gpuDoubleComplex *emiGR_Ga,
                                       double *Y_GL,
                                       int* l_s,
                                       int* a_J,
                                       int* s_J,
                                       int* I_J,
                                       double* f_GI,
                                       int nG,
                                       int nJ,
                                       int nL,
                                       int nI,
                                       int natoms,
                                       int nsplines,
                                       bool cc)
{
    int G = threadIdx.x + blockIdx.x * blockDim.x;
    int J = threadIdx.y + blockIdx.y * blockDim.y;
    gpuDoubleComplex imag_powers[4] = {make_gpuDoubleComplex(1.0,0),
                                       make_gpuDoubleComplex(0.0,-1.0),
                                       make_gpuDoubleComplex(-1.0,0),
                                       make_gpuDoubleComplex(0, 1.0)};
    if ((G < nG) && (J < nJ))
    {
        f_Gs += G*nsplines;
        emiGR_Ga += G*natoms;
        Y_GL += G*nL;
        f_GI += G*nI*2 + I_J[J];

        int s = s_J[J];
        int l = l_s[s];
        gpuDoubleComplex f1 = gpuCmulD(gpuCmul(emiGR_Ga[a_J[J]],
                                               imag_powers[l % 4]),
                                       f_Gs[s]);
        for (int m = 0; m < 2 * l + 1; m++) {
            gpuDoubleComplex f = gpuCmulD(f1, Y_GL[l * l + m]);
            f_GI[0] = f.x;
            f_GI[nI] = cc ? -f.y : f.y;
            f_GI++;
        }
    }
}

__global__ void pwlfc_expand_kernel_16(double* f_Gs,
                                       gpuDoubleComplex *emiGR_Ga,
                                       double *Y_GL,
                                       int* l_s,
                                       int* a_J,
                                       int* s_J,
                                       int* I_J,
                                       double* f_GI,
                                       int nG,
                                       int nJ,
                                       int nL,
                                       int nI,
                                       int natoms,
                                       int nsplines,
                                       bool cc)

{
    int G = threadIdx.x + blockIdx.x * blockDim.x;
    int J = threadIdx.y + blockIdx.y * blockDim.y;
    gpuDoubleComplex imag_powers[4] = {make_gpuDoubleComplex(1.0,0),
                                       make_gpuDoubleComplex(0.0,-1.0),
                                       make_gpuDoubleComplex(-1.0,0),
                                       make_gpuDoubleComplex(0, 1.0)};
    if ((G < nG) && (J < nJ))
    {
        f_Gs += G*nsplines;
        emiGR_Ga += G*natoms;
        Y_GL += G*nL;
        f_GI += (G*nI + I_J[J])*2;
        int s = s_J[J];
        int l = l_s[s];
        gpuDoubleComplex f1 = gpuCmulD(gpuCmul(emiGR_Ga[a_J[J]],
                                               imag_powers[l % 4]),
                                       f_Gs[s]);
        for (int m = 0; m < 2 * l + 1; m++) {
            gpuDoubleComplex f = gpuCmulD(f1, Y_GL[l * l + m]);
            *f_GI++ = f.x;
            *f_GI++ = cc ? -f.y : f.y;
        }
    }
}

extern "C"
void pwlfc_expand_gpu_launch_kernel(int itemsize,
                                    double* f_Gs,
                                    gpuDoubleComplex *emiGR_Ga,
                                    double *Y_GL,
                                    int* l_s,
                                    int* a_J,
                                    int* s_J,
                                    double* f_GI,
                                    int* I_J,
                                    int nG,
                                    int nJ,
                                    int nL,
                                    int nI,
                                    int natoms,
                                    int nsplines,
                                    bool cc)
{
    if (itemsize == 16)
    {
        gpuLaunchKernel(pwlfc_expand_kernel_16,
                        dim3((nG+15)/16, (nJ+15)/16),
                        dim3(16, 16),
                        0, 0,
                        f_Gs,
                        emiGR_Ga,
                        Y_GL,
                        l_s,
                        a_J,
                        s_J,
                        I_J,
                        f_GI,
                        nG,
                        nJ,
                        nL,
                        nI,
                        natoms,
                        nsplines,
                        cc);
    }
    else
    {
        gpuLaunchKernel(pwlfc_expand_kernel_8,
                        dim3((nG+15)/16, (nJ+15)/16),
                        dim3(16, 16),
                        0, 0,
                        f_Gs,
                        emiGR_Ga,
                        Y_GL,
                        l_s,
                        a_J,
                        s_J,
                        I_J,
                        f_GI,
                        nG,
                        nJ,
                        nL,
                        nI,
                        natoms,
                        nsplines,
                        cc);
    }
    //gpuDeviceSynchronize();
}
