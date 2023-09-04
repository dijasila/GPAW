#include "../gpu.h"
#include "../gpu-complex.h"

__global__ void _pw_insert(int nG,
                           int nQ,
                           double complex* c_G,
                           npy_int32* Q_G,
                           double scale,
                           double complex* tmp_Q)
// Does the same as these two lines of Python:
//
//     tmp_Q[:] = 0.0
//     tmp_Q.ravel()[Q_G] = c_G * scale
{
    int Qstart = threadIdx.x + blockIdx.x * blockDim.x;
    int Qend = 
    int J = threadIdx.y + blockIdx.y * blockDim.y;
    int Q1 = 0;
    for (int G = 0; G < nG; G++) {
        int Q2 = Q_G[G];
        for (; Q1 < Q2; Q1++)
            tmp_Q[Q1] = 0.0;
        tmp_Q[Q1++] = c_G[G] * scale;
        }
    for (; Q1 < nQ; Q1++)
        tmp_Q[Q1] = 0.0;
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
