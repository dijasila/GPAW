/*  Copyright (C) 2003-2007  CAMP
 *  Please see the accompanying LICENSE file for further information. */

#include <stdio.h>

#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include "gpaw-cuda-int.h"


#ifdef K

#ifndef CUGPAWCOMPLEX
#define BLOCK 16


#endif

#undef AC_X
#undef AC_Y
#undef ACK
#define ACK (2*(K-1))
#define AC_X (2*BLOCK+ACK)
#define AC_Y (BLOCK)

/*
__global__ void RST1D_kernel(const Tcuda* a, int n, int m, Tcuda* b)
{

  int j=blockIdx.x*BLOCK+threadIdx.x;
  if (j>=m) return;

  int i=blockIdx.y*BLOCK+threadIdx.y;
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
*/


__global__ void RST1D_kernel(const Tcuda* a, int n, int m, Tcuda* b,int ang,int bng, int blocks)
{
  __shared__ Tcuda ac[AC_Y*AC_X];
  Tcuda *acp;
  
  int jtid=threadIdx.x;
  int j=blockIdx.x*BLOCK;
  
  
  int itid=threadIdx.y;

  int yy=gridDim.y/blocks;
  int blocksi=blockIdx.y/yy;
  int ibl=blockIdx.y-yy*blocksi;

  int i=ibl*BLOCK;
  
  int sizex=(n * 2 + K * 2 - 3);

  int aind=(j+itid) * sizex + i * 2+jtid + K -1;

  a +=blocksi*ang+aind;
  b +=blocksi*bng+(j+jtid) + (i+itid) * m;
  
  acp=ac+AC_X*(itid)+jtid+ACK/2;
  if (aind<ang)
    acp[0]=a[0]; 
  if ((aind+BLOCK)<ang)  
    acp[BLOCK]=a[BLOCK];
  if  (jtid<ACK/2){
    if (aind-ACK/2<ang)
      acp[-ACK/2]=a[-ACK/2];
    if (aind+2*BLOCK<ang)
      acp[2*BLOCK]=a[2*BLOCK];
  }
  acp=ac+AC_X*(jtid)+2*itid+ACK/2;
  __syncthreads();
  
  if (((i+itid)<n) && ((j+jtid)<m)) {
    
    if      (K == 2)
      b[0] = MULDT(0.5 , ADD(acp[0] ,
			     MULDT(0.5 , ADD(acp[1] , acp[-1]))));
    
    else if (K == 4)
      b[0] = MULDT(0.5 , ADD(acp[0] ,
			     ADD(MULDT( 0.5625 , ADD(acp[1] , acp[-1])),
				 MULDT(-0.0625 , ADD(acp[3] , acp[-3])))));
    
    else if (K == 6)
      b[0] = MULDT(0.5 , ADD(ADD(acp[0] ,
				 MULDT( 0.58593750 , ADD(acp[1] , acp[-1]))) ,
			     ADD(MULDT(-0.09765625 , ADD(acp[3] , acp[-3])) ,
				 MULDT( 0.01171875 , ADD(acp[5] , acp[-5])))));
    
    else
      b[0] = MULDT(0.5 , 
		   ADD(acp[0] ,
		       ADD(ADD(MULDT( 0.59814453125 , ADD(acp[1] , acp[-1])) ,
			       MULDT(-0.11962890625 , ADD(acp[3] , acp[-3]))) ,
			   ADD(MULDT( 0.02392578125 , ADD(acp[5] , acp[-5])) ,
			       MULDT(-0.00244140625 , ADD(acp[7] , acp[-7]))))));
    
  }
}

void RST1D(const Tcuda* a, int n, int m, Tcuda* b,int ang,int bng, int blocks){
  

  
  int gridy=(n+BLOCK-1)/BLOCK;

  int gridx=(m+BLOCK-1)/BLOCK;
  
  dim3 dimBlock(BLOCK,BLOCK); 
  dim3 dimGrid(gridx,gridy*blocks);    

  RST1D_kernel<<<dimGrid, dimBlock, 0>>>(a,n,m, b, ang, bng, blocks);
  
  gpaw_cudaSafeCall(cudaGetLastError());

}

#else
#  define K 2
#  define RST1D Zcuda(bmgs_restrict1D2)
#  define RST1D_kernel Zcuda(bmgs_restrict1D2_kernel)
#  include "restrict-cuda.cu"
#  undef RST1D
#  undef RST1D_kernel
#  undef K
#  define K 4
#  define RST1D Zcuda(bmgs_restrict1D4)
#  define RST1D_kernel Zcuda(bmgs_restrict1D4_kernel)
#  include "restrict-cuda.cu"
#  undef RST1D
#  undef RST1D_kernel
#  undef K
#  define K 6
#  define RST1D Zcuda(bmgs_restrict1D6)
#  define RST1D_kernel Zcuda(bmgs_restrict1D6_kernel)
#  include "restrict-cuda.cu"
#  undef RST1D
#  undef RST1D_kernel
#  undef K
#  define K 8
#  define RST1D Zcuda(bmgs_restrict1D8)
#  define RST1D_kernel Zcuda(bmgs_restrict1D8_kernel)
#  include "restrict-cuda.cu"
#  undef RST1D
#  undef RST1D_kernel
#  undef K

extern "C"{
  
  void Zcuda(bmgs_restrict_cuda_gpu)(int k, Tcuda* a, const int n[3], 
				     Tcuda* b, const int nb[3], Tcuda* w,
				     int blocks)
  {
    void (*plg)(const Tcuda*, int, int, Tcuda*,int,int,int);
    int ang=n[0]*n[1]*n[2];
    int bng=nb[0]*nb[1]*nb[2];
    //printf("ang %d  bng %d\n",ang,bng);
    
    if (k == 2)
      plg = Zcuda(bmgs_restrict1D2);
    else if (k == 4)
      plg = Zcuda(bmgs_restrict1D4);
    else if (k == 6)
      plg = Zcuda(bmgs_restrict1D6);
    else
      plg = Zcuda(bmgs_restrict1D8);
    
    int e = k * 2 - 3;
    //    for (int i = 0; i < blocks; i++){
    plg(a, (n[2] - e) / 2, n[0] * n[1], w, ang, ang, blocks);
    plg(w, (n[1] - e) / 2, n[0] * (n[2] - e) / 2, a, ang, ang, blocks);
    plg(a, (n[0] - e) / 2, (n[1] - e) * (n[2] - e) / 4, b, ang, bng, blocks);

    /*   a+=n[0]*n[1]*n[2];
	 w+=n[0]*n[1]*n[2];
	 b+=nb[0]*nb[1]*nb[2];
	 }*/
  }
}
#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "restrict-cuda.cu"

extern "C"{
  double bmgs_restrict_cuda_cpu(int k, double* a, const int n[3], double* b, 
				double* w)
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
    bmgs_restrict_cuda_gpu(k, adev, n, bdev,n, wdev,1);
  
  
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
