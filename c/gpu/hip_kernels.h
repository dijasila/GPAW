#ifdef GPAW_HIP_KERNELS_H
#define GPAW_HIP_KERNELS_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>

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
					       bool cc);

#endif
