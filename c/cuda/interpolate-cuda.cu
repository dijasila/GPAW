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
#define BLOCK_SIZEX BLOCK
#define BLOCK_SIZEY BLOCK
#endif

#undef AC_X
#undef AC_Y
#undef ACK
#define ACK (K)
#define AC_X (BLOCK+ACK)
#define AC_Y (BLOCK)

/*
__global__ void IP1D_kernel(const Tcuda* a, int n, int m, Tcuda* b, int skip0,
			    int skip1)
{
  a += K / 2 - 1;

  int j=blockIdx.x*BLOCK_SIZEX+threadIdx.x;
  if (j>=m) return;
  
  int i=blockIdx.y*BLOCK_SIZEY+threadIdx.y;
  if (i>=n) return;
  //  return;
  a += j * (K - 1 - skip1 + n)+i;
  b += j + 2 * m * i;
  
  

  if (skip0){
    b-=m; 
  }

  if (i>0 || !skip0)
    b[0] = a[0];
  

 
  if (i == n - 1 && skip1)
    b -= m;
  else
    {
      if (K == 2)
	b[m] = MULDT(0.5, ADD(a[0] , a[1]) );
      else if (K == 4)
	b[m] = ADD( MULDT( 0.5625 ,  ADD(a[ 0] , a[1])) ,
		    MULDT(-0.0625 ,  ADD(a[-1] , a[2])));
      else if (K == 6)
	b[m] = ADD(ADD( MULDT( 0.58593750 , ADD(a[ 0] , a[1])),
			MULDT(-0.09765625 , ADD(a[-1] , a[2]))) ,
		   MULDT(0.01171875 , ADD(a[-2] , a[3])));
      else
	b[m] = ADD( ADD( MULDT( 0.59814453125 , ADD(a[ 0] , a[1])) ,
			 MULDT(-0.11962890625 , ADD(a[-1] , a[2]))) ,
		    ADD ( MULDT( 0.02392578125 , ADD(a[-2] , a[3])) ,
			  MULDT(-0.00244140625 , ADD(a[-3] , a[4]))));
    }
  
}
*/
/*

__global__ void IP1D_kernel(const Tcuda* a, int n, int m, Tcuda* b, int skip0)
{

  int j=blockIdx.x*BLOCK+threadIdx.x;
  if (j>=m) return;
  
  int i=blockIdx.y*BLOCK+threadIdx.y;
  if (i>=n) return;
  a += j * (K - 1 + n)+i;
  b += j + 2 * m * i;
  
  
  if (i>0 || !skip0)
    b[0] = a[0];
  
  if (K == 2)
    b[m] = MULDT(0.5, ADD(a[0] , a[1]) );
  else if (K == 4)
    b[m] = ADD( MULDT( 0.5625 ,  ADD(a[ 0] , a[1])) ,
		MULDT(-0.0625 ,  ADD(a[-1] , a[2])));
  else if (K == 6)
    b[m] = ADD(ADD( MULDT( 0.58593750 , ADD(a[ 0] , a[1])),
		    MULDT(-0.09765625 , ADD(a[-1] , a[2]))) ,
	       MULDT(0.01171875 , ADD(a[-2] , a[3])));
  else
    b[m] = ADD( ADD( MULDT( 0.59814453125 , ADD(a[ 0] , a[1])) ,
		     MULDT(-0.11962890625 , ADD(a[-1] , a[2]))) ,
		ADD ( MULDT( 0.02392578125 , ADD(a[-2] , a[3])) ,
		      MULDT(-0.00244140625 , ADD(a[-3] , a[4]))));
  
  
}
*/

__global__ void IP1D_kernel(const Tcuda* a, int n, int m, Tcuda* b, int skip0, int skip1, int ang, int bng, int blocks)
{
  
  __shared__ Tcuda ac[AC_Y*AC_X];
  Tcuda *acp;
  
  int jtid=threadIdx.x;
  int j=blockIdx.x*BLOCK;

  int itid=threadIdx.y;
  int ibl=blockIdx.y/blocks;
  int blocksi=blockIdx.y-blocks*ibl;
  int i=ibl*BLOCK;

  a += blocksi*ang + (j+itid) * (K - 1 -skip1 + n)+(i+jtid);
  b += blocksi*bng + (j+jtid) + 2 * m * (i+itid);
    
  acp=ac+AC_X*(itid)+jtid+ACK/2-1;
  
  acp[0]=a[0];

  if  (jtid<ACK/2-1){
    acp[-ACK/2+1]=a[-ACK/2+1];
  }  
  if  (jtid<ACK/2){
    acp[BLOCK]=a[BLOCK];
  }
  acp=ac+AC_X*(jtid)+itid+ACK/2-1;
  __syncthreads();
    
  if (((i+itid)<n) && ((j+jtid)<m)) {

    if (!skip0 || (i+itid)>0 ) {
      b[0] = acp[0];
    }
    if (!(i == n - 1 && skip1)) {
      if (K == 2)
	b[m] = MULDT(0.5, ADD(acp[0] , acp[1]) );
      else if (K == 4)
	b[m] = ADD( MULDT( 0.5625 ,  ADD(acp[ 0] , acp[1])) ,
		    MULDT(-0.0625 ,  ADD(acp[-1] , acp[2])));
      else if (K == 6)
	b[m] = ADD(ADD( MULDT( 0.58593750 , ADD(acp[ 0] , acp[1])),
			MULDT(-0.09765625 , ADD(acp[-1] , acp[2]))) ,
		   MULDT(0.01171875 , ADD(acp[-2] , acp[3])));
      else
	b[m] = ADD( ADD( MULDT( 0.59814453125 , ADD(acp[ 0] , acp[1])) ,
			 MULDT(-0.11962890625 , ADD(acp[-1] , acp[2]))) ,
		    ADD ( MULDT( 0.02392578125 , ADD(acp[-2] , acp[3])) ,
			  MULDT(-0.00244140625 , ADD(acp[-3] , acp[4]))));
    }
  }
}


void IP1D(const Tcuda* a, int n, int m, Tcuda* b, int skip[2],int ang, int bng,
	  int blocks)
{
  
  a += K / 2 - 1;
  if (skip[0]) b-=m;
  
  int gridy=(n+BLOCK-1)/BLOCK;

  int gridx=(m+BLOCK-1)/BLOCK;
  
  dim3 dimBlock(BLOCK,BLOCK); 
  dim3 dimGrid(gridx,gridy*blocks);    


  IP1D_kernel<<<dimGrid, dimBlock, 0>>>(a,n,m, b,skip[0],skip[1],ang,bng,blocks);

  gpaw_cudaSafeCall(cudaGetLastError());
}
 
/*
void IP1D(const Tcuda* a, int n, int m, Tcuda* b, int skip[2])
{

  int gridy=(n+BLOCK_SIZEY-1)/BLOCK_SIZEY;

  int gridx=(m+BLOCK_SIZEX-1)/BLOCK_SIZEX;
  
  dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY); 
  dim3 dimGrid(gridx,gridy);    


  IP1D_kernel<<<dimGrid, dimBlock, 0>>>(a,n,m, b,skip[0],skip[1]);

  gpaw_cudaSafeCall(cudaGetLastError());
}
*/

#else
#  define K 2
#  define IP1D Zcuda(bmgs_interpolate1D2)
#  define IP1D_kernel Zcuda(bmgs_interpolate1D2_kernel)
#  include "interpolate-cuda.cu"
#  undef IP1D
#  undef IP1D_kernel
#  undef K
#  define K 4
#  define IP1D Zcuda(bmgs_interpolate1D4)
#  define IP1D_kernel Zcuda(bmgs_interpolate1D4_kernel)
#  include "interpolate-cuda.cu"
#  undef IP1D
#  undef IP1D_kernel
#  undef K
#  define K 6
#  define IP1D Zcuda(bmgs_interpolate1D6)
#  define IP1D_kernel Zcuda(bmgs_interpolate1D6_kernel)
#  include "interpolate-cuda.cu"
#  undef IP1D
#  undef IP1D_kernel
#  undef K
#  define K 8
#  define IP1D Zcuda(bmgs_interpolate1D8)
#  define IP1D_kernel Zcuda(bmgs_interpolate1D8_kernel)
#  include "interpolate-cuda.cu"
#  undef IP1D
#  undef IP1D_kernel
#  undef K

extern "C"{

  void Zcuda(bmgs_interpolate_cuda_gpu)(int k, int skip[3][2],
					const Tcuda* a, const int size[3], 
					Tcuda* b, const int sizeb[3], Tcuda* w,
					int blocks)
  {
    void (*ip)(const Tcuda*, int, int, Tcuda*, int[2],int,int,int);
    int ang=size[0]*size[1]*size[2];
    int bng=sizeb[0]*sizeb[1]*sizeb[2];
    if (k == 2)
      ip = Zcuda(bmgs_interpolate1D2);
    else if (k == 4)
      ip = Zcuda(bmgs_interpolate1D4);
    else if (k == 6)
      ip = Zcuda(bmgs_interpolate1D6);
    else
      ip = Zcuda(bmgs_interpolate1D8);
    int e = k - 1;
    //    for (int i = 0; i < blocks; i++){
      ip(a, size[2] - e + skip[2][1],
	 size[0] *
	 size[1],
	 b, skip[2],ang,bng,blocks);
      ip(b, size[1] - e + skip[1][1],
	 size[0] *
	 ((size[2] - e) * 2 - skip[2][0] + skip[2][1]),
	 w, skip[1],bng,bng,blocks);
      ip(w, size[0] - e + skip[0][1],
	 ((size[1] - e) * 2 - skip[1][0] + skip[1][1]) *
	 ((size[2] - e) * 2 - skip[2][0] + skip[2][1]),
	 b, skip[0],bng,bng,blocks);
      /*      a+=size[0]*size[1]*size[2];
      b+=sizeb[0]*sizeb[1]*sizeb[2];
      }*/
  }
}
#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "interpolate-cuda.cu"


extern "C"{
  double bmgs_interpolate_cuda_cpu(int k, int skip[3][2], const double* a, 
				   const int n[3], double* b, double* w)
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

    bmgs_interpolate_cuda_gpu(k,skip, adev, n, bdev,n, wdev,1);
  
  
    cudaThreadSynchronize();  

    gpaw_cudaSafeCall(cudaGetLastError());

    gettimeofday(&t1,NULL);
  
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
