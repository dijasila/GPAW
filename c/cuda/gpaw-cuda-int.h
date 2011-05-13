#include<cuda.h>
#include<cuda_runtime_api.h>
#include<cuComplex.h>

#include"gpaw-cuda-common.h"

#undef Tcuda
#undef Zcuda
#undef MULTD
#undef MULDT
#undef ADD
#undef IADD
#undef MAKED
#undef MULTT

#ifndef CUGPAWCOMPLEX
#  define Tcuda double
#  define Zcuda(f) f
#  define MULTT(a,b) ((a)*(b))
#  define MULTD(a,b) ((a)*(b))
#  define MULDT(a,b) ((a)*(b))
#  define ADD(a,b)   ((a)+(b))
#  define IADD(a,b)  ((a)+=(b))
#  define MAKED(a)   (a)
#else
#  define Tcuda cuDoubleComplex
#  define Zcuda(f) f ## z
#  define MULTT(a,b) cuCmul((a),(b))
#  define MULTD(a,b) cuCmulD((a),(b))
#  define MULDT(b,a) MULTD((a),(b))
#  define ADD(a,b)   cuCadd((a),(b))
#  define IADD(a,b)  (a)=ADD((a),(b))
#  define MAKED(a)   make_cuDoubleComplex((a),0)
#endif


#ifndef GPAW_CUDA_INT_H
#define GPAW_CUDA_INT_H

#include <Python.h>

#define FD_BLOCK_X 16
#define FD_BLOCK_Xz (2*(FD_BLOCK_X))
#define FD_BLOCK_Y 8
#define FD_XDIV 4

#define FD_MAXJ      10
#define FD_MAXCOEFS  32

#define FD_ACACHE_X  ((FD_BLOCK_X)+16)
#define FD_ACACHE_Xz  ((FD_BLOCK_Xz)+2*16)


#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))


typedef struct
{
  int ncoefs;
  double* coefs;
  long* offsets;
  long n[3];
  long j[3];
} bmgsstencil;


__host__ __device__ static __inline__ cuDoubleComplex cuCmulD(cuDoubleComplex x,double y)
{
    cuDoubleComplex prod;
    prod = make_cuDoubleComplex ((cuCreal(x) * y),
                                 (cuCimag(x) * y));
    return prod;
}



#endif
