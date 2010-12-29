#include<cuda.h>
#include<cublas.h>
#include<driver_types.h>
#include<cuda_runtime_api.h>

#include <stdio.h>
#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include "gpaw-cuda-int.h"

extern "C" {
#ifndef CUGPAWCOMPLEX
  void bmgs_translate_cuda(Tcuda* a, const int sizea[3], const int size[3],
			   const int start1[3], const int start2[3],
			   enum cudaMemcpyKind kind)
#else
    void bmgs_translate_cudaz(Tcuda* a, const int sizea[3], const int size[3],
			      const int start1[3], const int start2[3],
			      cuDoubleComplex phase,enum cudaMemcpyKind kind)
#endif
  {		
    const Tcuda* __restrict__ s = a + start1[2] + (start1[1] + start1[0] * sizea[1]) * sizea[2];
    Tcuda* __restrict__ d = a + start2[2] + (start2[1] + start2[0] * sizea[1]) * sizea[2];

    
    for (int i0 = 0; i0 < size[0]; i0++)
      {
	for (int i1 = 0; i1 < size[1]; i1++)
	  {
	    gpaw_cudaSafeCall(cudaMemcpy(d, s, size[2] * sizeof(Tcuda),kind));
#ifdef CUGPAWCOMPLEX
	    cublasZscal(size[2],phase,d,1);
	    gpaw_cublasSafeCall(cublasGetError());
#endif
	    s += sizea[2];
	    d += sizea[2];
	  }
	s += sizea[2] * (sizea[1] - size[1]);
	d += sizea[2] * (sizea[1] - size[1]);
      }
  }
}

#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "translate-cuda.cu"
#endif
