#include "hip_kernels.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>

#if defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__)
__device__ hipDoubleComplex operator*(hipDoubleComplex a, double b)
{
    return make_hipDoubleComplex(a.x*b, a.y*b);
}

#include "cuComplex.h"
__device__ hipDoubleComplex operator*(hipDoubleComplex p, hipDoubleComplex q) {
    return cuCmul(p, q);
}

#endif

__global__ void pwlfc_expand_kernel_8(double* f_Gs,
				       hipDoubleComplex *emiGR_Ga,
				       double *Y_GL,
				       uint32_t* l_s,
	                               uint32_t* a_J,
                                       uint32_t* s_J, 
                                       uint32_t* I_J, 
                                       double* f_GI,
                                       int nG,
                                       int nJ,
                                       int nL,
				       int nI,
                                       int natoms,
                                       int nsplines,
                                       bool cc)
{
    int G =threadIdx.x + blockIdx.x *blockDim.x;
    int J =threadIdx.y + blockIdx.y *blockDim.y;
    hipDoubleComplex imag_powers[4] = {make_hipDoubleComplex(1.0,0), 
	                               make_hipDoubleComplex(0.0,-1.0),
				       make_hipDoubleComplex(-1.0,0),
				       make_hipDoubleComplex(0, 1.0)};
    if ((G < nG) && (J < nJ))
    {
        f_Gs += G*nsplines;
        emiGR_Ga += G*natoms;
        Y_GL += G*nL;
        f_GI += G*nI*2+I_J[J];
	    
	int s = s_J[J];
	int l = l_s[s];
	hipDoubleComplex f1 = (emiGR_Ga[a_J[J]]* f_Gs[s] *
                               imag_powers[l % 4]);
	for (int m = 0; m < 2 * l + 1; m++) {
	    hipDoubleComplex f = f1 * Y_GL[l * l + m];
	    f_GI[0] = f.x;
	    f_GI[nI] = cc ? -f.y : f.y;
	    f_GI++;
	}
    }
}

__global__ void pwlfc_expand_kernel_16(double* f_Gs,
				       hipDoubleComplex *emiGR_Ga,
				       double *Y_GL,
				       uint32_t* l_s,
	                               uint32_t* a_J,
                                       uint32_t* s_J, 
                                       uint32_t* I_J, 
                                       double* f_GI,
                                       int nG,
                                       int nJ,
                                       int nL,
				       int nI,
                                       int natoms,
                                       int nsplines,
                                       bool cc)

{
    int G =threadIdx.x + blockIdx.x *blockDim.x;
    int J =threadIdx.y + blockIdx.y *blockDim.y;
    hipDoubleComplex imag_powers[4] = {make_hipDoubleComplex(1.0,0), 
	                               make_hipDoubleComplex(0.0,-1.0),
				       make_hipDoubleComplex(-1.0,0),
				       make_hipDoubleComplex(0, 1.0)};
    if ((G < nG) && (J < nJ))
    {
        f_Gs += G*nsplines;
        emiGR_Ga += G*natoms;
        Y_GL += G*nL;
        f_GI += (G*nI+I_J[J])*2;
	int s = s_J[J];
	int l = l_s[s];
	hipDoubleComplex f1 = (emiGR_Ga[a_J[J]] *
			     f_Gs[s] *
			     imag_powers[l % 4]);
	for (int m = 0; m < 2 * l + 1; m++) {
	    hipDoubleComplex f = f1 * Y_GL[l * l + m];
	    *f_GI++ = f.x;
	    *f_GI++ = cc ? -f.y : f.y;
	}
    }
}

extern "C" void pwlfc_expand_gpu_launch_kernel(int itemsize, 
		                               double* f_Gs,
		                               hipDoubleComplex *emiGR_Ga,
					       double *Y_GL,
					       uint32_t* l_s,
	                                       uint32_t* a_J,
                                               uint32_t* s_J, 
                                               double* f_GI,
					       uint32_t* I_J,
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
        hipLaunchKernelGGL(pwlfc_expand_kernel_16, dim3((nG+15)/16, (nJ+15)/16), dim3(16, 16), 0, 0, f_Gs, 
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
        hipLaunchKernelGGL(pwlfc_expand_kernel_8, dim3((nG+15)/16, (nJ+15)/16), dim3(16, 16), 0, 0, f_Gs, 
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
    //hipDeviceSynchronize();
}
