/*  Copyright (C) 2003-2007  CAMP
 *  Please see the accompanying LICENSE file for further information. */

//#include "bmgs.h"
//#include <pthread.h>
//#include "../extensions.h"

#include <stdio.h>

#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include "gpaw-cuda-int.h"


#ifdef K
struct RST1DA{
  int thread_id;
  int nthds;
  const double* a;
  int n;
  int m;
  double* b;
};
#ifndef CUGPAWCOMPLEX
#define BLOCK_SIZEX 16
#define BLOCK_SIZEY 8

#define GPAW_MALLOC(T, n) (T*)(malloc((n) * sizeof(T)))

#endif


__global__ void RST1D_kernel(const Tcuda* a, int n, int m, Tcuda* b)
{
  a += K - 1;

  int j=blockIdx.x*BLOCK_SIZEX+threadIdx.x;
  if (j>=m) return;

  int i=blockIdx.y*BLOCK_SIZEY+threadIdx.y;
  if (i>=n) return;

  a += j * (n * 2 + K * 2 - 3) + i * 2;
  b += j + i * m ;
  
  if      (K == 2)
    b[0] = MULDT(0.5 , ADD(a[0] ,
			   MULDT(0.5 , ADD(a[1] , a[-1]))));
  
  else if (K == 4)
    b[0] = MULDT(0.5 , ADD(a[0] ,
		     ADD(MULDT( 0.5625 , ADD(a[1] , a[-1])),
			 MULDT(-0.0625 , ADD(a[3] , a[-3])))));
  
  else if (K == 6)
    b[0] = MULDT(0.5 , ADD(ADD(a[0] ,
			       MULDT( 0.58593750 , ADD(a[1] , a[-1]))) ,
			   ADD(MULDT(-0.09765625 , ADD(a[3] , a[-3])) ,
			       MULDT( 0.01171875 , ADD(a[5] , a[-5])))));
  
  else
    b[0] = MULDT(0.5 , ADD(a[0] ,
			   ADD(ADD(MULDT( 0.59814453125 , ADD(a[1] , a[-1])) ,
				   MULDT(-0.11962890625 , ADD(a[3] , a[-3]))) ,
			       ADD(MULDT( 0.02392578125 , ADD(a[5] , a[-5])) ,
				   MULDT(-0.00244140625 , ADD(a[7] , a[-7]))))));
  
}

void RST1D(const Tcuda* a, int n, int m, Tcuda* b){

  int gridy=(n+BLOCK_SIZEY-1)/BLOCK_SIZEY;

  int gridx=(m+BLOCK_SIZEX-1)/BLOCK_SIZEX;
  
  dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY); 
  dim3 dimGrid(gridx,gridy);    

  RST1D_kernel<<<dimGrid, dimBlock, 0>>>(a,n,m, b);
  
  gpaw_cudaSafeCall(cudaGetLastError());

}

#else
#  define K 2
#  define RST1D Zcuda(bmgs_restrict1D2)
#  define RST1DA Zcuda(bmgs_restrict1D2_args)
#  define RST1D_kernel Zcuda(bmgs_restrict1D2_kernel)
#  include "restrict-cuda.cu"
#  undef RST1D
#  undef RST1DA
#  undef RST1D_kernel
#  undef K
#  define K 4
#  define RST1D Zcuda(bmgs_restrict1D4)
#  define RST1DA Zcuda(bmgs_restrict1D4_args)
#  define RST1D_kernel Zcuda(bmgs_restrict1D4_kernel)
#  include "restrict-cuda.cu"
#  undef RST1D
#  undef RST1DA
#  undef RST1D_kernel
#  undef K
#  define K 6
#  define RST1D Zcuda(bmgs_restrict1D6)
#  define RST1DA Zcuda(bmgs_restrict1D6_args)
#  define RST1D_kernel Zcuda(bmgs_restrict1D6_kernel)
#  include "restrict-cuda.cu"
#  undef RST1D
#  undef RST1DA
#  undef RST1D_kernel
#  undef K
#  define K 8
#  define RST1D Zcuda(bmgs_restrict1D8)
#  define RST1DA Zcuda(bmgs_restrict1D8_args)
#  define RST1D_kernel Zcuda(bmgs_restrict1D8_kernel)
#  include "restrict-cuda.cu"
#  undef RST1D
#  undef RST1DA
#  undef RST1D_kernel
#  undef K

extern "C"{
  
  void Zcuda(bmgs_restrict_cuda_gpu)(int k, Tcuda* a, const int n[3], Tcuda* b, Tcuda* w)
  {
    void (*plg)(const Tcuda*, int, int, Tcuda*);
    
    if (k == 2)
      plg = Zcuda(bmgs_restrict1D2);
    else if (k == 4)
      plg = Zcuda(bmgs_restrict1D4);
    else if (k == 6)
      plg = Zcuda(bmgs_restrict1D6);
    else
      plg = Zcuda(bmgs_restrict1D8);
    
    int e = k * 2 - 3;
    plg(a, (n[2] - e) / 2, n[0] * n[1], w);
    plg(w, (n[1] - e) / 2, n[0] * (n[2] - e) / 2, a);
    plg(a, (n[0] - e) / 2, (n[1] - e) * (n[2] - e) / 4, b);
  }
}
#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "restrict-cuda.cu"

extern "C"{
  double bmgs_restrict_cuda_cpu(int k, double* a, const int n[3], double* b, double* w)
  {
    double *adev,*bdev,*wdev;
    size_t bsize;
    struct timeval  t0, t1; 
    double flops;
  
  
    bsize=n[0]*n[1]*n[2];
  
    gpaw_cudaSafeCall(cudaMalloc(&adev,sizeof(double)*bsize));
  
    gpaw_cudaSafeCall(cudaMalloc(&bdev,sizeof(double)*bsize));
    gpaw_cudaSafeCall(cudaMalloc(&wdev,sizeof(double)*bsize));
   
    gpaw_cudaSafeCall(cudaMemcpy(adev,a,sizeof(double)*bsize,
				 cudaMemcpyHostToDevice));
    gpaw_cudaSafeCall(cudaMemcpy(bdev,b,sizeof(double)*bsize,
				 cudaMemcpyHostToDevice));
    gpaw_cudaSafeCall(cudaMemcpy(wdev,w,sizeof(double)*bsize,
				 cudaMemcpyHostToDevice));
  
    gettimeofday(&t0,NULL);
    bmgs_restrict_cuda_gpu(k, adev, n, bdev, wdev);
  
  
    cudaThreadSynchronize();  

    gpaw_cudaSafeCall(cudaGetLastError());
    
    gettimeofday(&t1,NULL);
  
    gpaw_cudaSafeCall(cudaMemcpy(a,adev,sizeof(double)*bsize,
				 cudaMemcpyDeviceToHost));
    gpaw_cudaSafeCall(cudaMemcpy(b,bdev,sizeof(double)*bsize,
				 cudaMemcpyDeviceToHost));
    gpaw_cudaSafeCall(cudaMemcpy(w,wdev,sizeof(double)*bsize,
				 cudaMemcpyDeviceToHost));
  
  
    gpaw_cudaSafeCall(cudaFree(adev));
    gpaw_cudaSafeCall(cudaFree(bdev));
    gpaw_cudaSafeCall(cudaFree(wdev));
  
    flops=(t1.tv_sec*1.0+t1.tv_usec/1000000.0-t0.tv_sec*1.0-t0.tv_usec/1000000.0); 
  
    return flops;

  }

}
#endif
#endif
