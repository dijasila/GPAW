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
    if (R < nR)
    {
        double rho = 0.0;
        for (int b=0; b< nb; b++)
        {
            int idx = b * nR + R;
            rho += f_n[b] * (psit_nR[idx].x * psit_nR[idx].x + psit_nR[idx].y * psit_nR[idx].y);
        }
        rho_R[R] += rho;
    }
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

extern "C" void gpawDeviceSynchronize()
{
    gpuDeviceSynchronize();
}

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
void pw_insert_gpu_launch_kernel(
                             int nb,
                             int nG,
                             int nQ,
                             double* c_nG,
                             npy_int32* Q_G,
                             double scale,
                             double* tmp_nQ)
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

// outP_ani[a] = \sum_A H_aii[a] P_ani[a]
__global__ void dH_aii_times_P_ani_16(int nA, int nn, int nI, 
                                      npy_int32* ni_a, double* dH_aii_dev, 
                                      gpuDoubleComplex* P_ani_dev,
                                      gpuDoubleComplex* outP_ani_dev)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    if (n1 < nn) {
        double* dH_ii = dH_aii_dev;
        int I = 0;        
        for (int a=0; a< nA; a++)
        {
            int ni = ni_a[a];
            int Istart = I;
            for (int i=0; i< ni; i++)
            {
                gpuDoubleComplex* outP_ni = outP_ani_dev + n1 * nI + I;
                gpuDoubleComplex result = make_gpuDoubleComplex(0.0, 0.0);
                gpuDoubleComplex* P_ni = P_ani_dev + n1 * nI + Istart;
                for (int i2=0; i2 < ni; i2++)
                {
                   //printf("%d %d %d %d %g\n", n1, a, i, i2, dH_ii[i2 * ni + i]);
                   gpuDoubleComplex item = gpuCmulD(*P_ni, dH_ii[i2 * ni + i]);
                   result.x += item.x;
                   result.y += item.y;
                   P_ni++;
                }
                outP_ni->x = result.x;
                outP_ni->y = result.y;
                I++;
            }
            //P_ni += ni;
            //outP_ni += ni;
            dH_ii += ni * ni;
        }
    }
}



extern "C"
void dH_aii_times_P_ani_launch_kernel(int nA, int nn,
                                      int nI, npy_int32* ni_a, 
                                      double* dH_aii_dev, 
                                      gpuDoubleComplex* P_ani_dev,
                                      gpuDoubleComplex* outP_ani_dev)
{
    gpuLaunchKernel(dH_aii_times_P_ani_16,
                    dim3((nn+255)/256),
                    dim3(256),
                    0, 0,
                    nA, nn, nI, ni_a, dH_aii_dev,
                    P_ani_dev, outP_ani_dev);
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
