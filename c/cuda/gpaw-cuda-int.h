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
#undef CONJ
#undef REAL
#undef IMAG
#undef NEG

#ifndef CUGPAWCOMPLEX
#  define Tcuda double
#  define Zcuda(f) f
#  define MULTT(a,b) ((a)*(b))
#  define MULTD(a,b) ((a)*(b))
#  define MULDT(a,b) ((a)*(b))
#  define ADD(a,b)   ((a)+(b))
#  define IADD(a,b)  ((a)+=(b))
#  define MAKED(a)   (a)
#  define CONJ(a)    (a)
#  define REAL(a)    (a)
#  define IMAG(a)    (0)
#  define NEG(a)     (-(a))
#else
#  define Tcuda cuDoubleComplex
#  define Zcuda(f) f ## z
#  define MULTT(a,b) cuCmul((a),(b))
#  define MULTD(a,b) cuCmulD((a),(b))
#  define MULDT(b,a) MULTD((a),(b))
#  define ADD(a,b)   cuCadd((a),(b))
#  define IADD(a,b)  (a)=ADD((a),(b))
#  define MAKED(a)   make_cuDoubleComplex((a),0)
#  define CONJ(a)    cuConj((a))
#  define REAL(a)    cuCreal(a)
#  define IMAG(a)    cuCimag(a)
#  define NEG(a)     cuCneg(a)
#endif


#ifndef GPAW_CUDA_INT_H
#define GPAW_CUDA_INT_H

#include <Python.h>


#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif
#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

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

__host__ __device__ static __inline__ cuDoubleComplex cuCneg(cuDoubleComplex x)
{
    return make_cuDoubleComplex (-cuCreal(x), -cuCimag(x));
}



#endif
