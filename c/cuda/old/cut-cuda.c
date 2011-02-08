#include<cuda.h>
#include<driver_types.h>
#include<cuda_runtime_api.h>

#include <string.h>


extern "C" {
void bmgs_cut_cuda(const double* a, const int n[3], const int c[3],
                 double* b, const int m[3],enum cudaMemcpyKind kind)
{
  /*  a += c[2] + (c[1] + c[0] * n[1]) * n[2];
  for (int i0 = 0; i0 < m[0]; i0++)
    {
      for (int i1 = 0; i1 < m[1]; i1++)
        {
          cudaMemcpy(b, a, m[2] * sizeof(double),kind);
          a += n[2];
          b += m[2];
        }
      a += n[2] * (n[1] - m[1]);
    }
  */
  if (!(m[0] && m[2] && m[3])) return;

  cudaMemcpy3DParms myParms = {0};
  
  myParms.srcPtr=make_cudaPitchedPtr((void*)a, n[2]*sizeof(double), n[2], n[1] );
  
  myParms.dstPtr=make_cudaPitchedPtr((void*)b, m[2]*sizeof(double), m[2], m[1] );
  myParms.extent=make_cudaExtent(m[2]*sizeof(double),m[1],m[0]);
  myParms.srcPos=make_cudaPos(c[2]*sizeof(double),c[1],c[0]);
  
  myParms.kind=kind;
  gpaw_cudaSafeCall(cudaMemcpy3D(&myParms));
}

}
