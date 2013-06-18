/*  Copyright (C) 2003-2007  CAMP
 *  Please see the accompanying LICENSE file for further information. */

#include <stdio.h>

#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include "gpaw-cuda-int.h"


#ifndef CUGPAWCOMPLEX

#define BLOCK_X 32
#define BLOCK_Y 16
#define BCACHE_X ((BLOCK_X)+1)
#define BCACHE_Y ((BLOCK_Y)+1)
#define ACACHE_X ((BLOCK_X)/2+1)
#define ACACHE_Y ((BLOCK_Y)/2+1)

#endif



__global__ void Zcuda(interpolate_kernel)(const Tcuda* a, const int3 n,
					  Tcuda* b,const int3 b_n, 
					  const int3 skip0,const int3 skip1,
					  int xdiv,int blocks)
{
  int xx=gridDim.x/xdiv;
  int yy=gridDim.y/blocks;

  int xind=blockIdx.x/xx;

  int i2tid=threadIdx.x;
  int i2base=(blockIdx.x-xind*xx)*BLOCK_X;
  int i2=i2base+i2tid;

  int blocksi=blockIdx.y/yy;

  int i1tid=threadIdx.y;

  int i1base=(blockIdx.y-blocksi*yy)*BLOCK_Y;
  int i1=i1base+i1tid;

  __shared__ Tcuda bcache12[BCACHE_Y*BCACHE_X];

  Tcuda *bcache12p_2x;
  
  int xlen=(n.x+xdiv-1)/xdiv;
  int xstart=xind*xlen;
  int xend=MIN(xstart+xlen,n.x);

  if (xind<xdiv-1) xend++;

  xlen=xend-xstart;

  a+=n.x*n.y*n.z*blocksi+xstart*n.y*n.z+((i1base/2)+i1tid)*n.z+(i2base/2)+i2tid;
  
  if (skip0.y)
    i1--;
  if (skip0.z)
    i2--;

  b+=b_n.x*b_n.y*b_n.z*blocksi+2*xstart*b_n.y*b_n.z+i1*b_n.z+i2;
  
  if ((xind>0) && (skip0.x) )
    b-=b_n.y*b_n.z;
   
  bcache12p_2x=bcache12+BCACHE_X*(2*i1tid)+2*i2tid;

  if (i1tid<ACACHE_Y && i2tid<ACACHE_X)
    bcache12p_2x[0]=a[0];
  
  __syncthreads();         
  for (int i0=xstart+1; i0 < xend; i0++) { 
    Tcuda a_c;
    a+=n.y*n.z;
    if (i1tid<ACACHE_Y && i2tid<BLOCK_X/2)
      bcache12p_2x[1]=MULTD(ADD(bcache12p_2x[0],bcache12p_2x[2]),0.5);
    __syncthreads();         
    if (i1tid<BLOCK_Y/2) {
      bcache12p_2x[BCACHE_X*1-i2tid]=
	MULTD(ADD(bcache12p_2x[BCACHE_X*0-i2tid],
		  bcache12p_2x[BCACHE_X*2-i2tid]),0.5);
      if ((skip1.z) && (i2tid<1))
	bcache12p_2x[BCACHE_X*1-i2tid+BLOCK_X]=
	  MULTD(ADD(bcache12p_2x[BCACHE_X*0-i2tid+BLOCK_X],
		    bcache12p_2x[BCACHE_X*2-i2tid+BLOCK_X]),0.5);
    }
    __syncthreads();         

    if (i0>1 || !skip0.x) {
      if ((i1<b_n.y) && (i2<b_n.z) && (i1>=0) && (i2>=0))	
	b[0]=bcache12[BCACHE_X*(i1tid)+i2tid];
      b+=b_n.y*b_n.z;    
    }

    __syncthreads();         
    if (i1tid<ACACHE_Y && i2tid<ACACHE_X) {
      a_c=a[0];
      bcache12p_2x[0]=MULTD(ADD(bcache12p_2x[0],a_c),0.5);
    }
    __syncthreads();  
    if (i1tid<ACACHE_Y && i2tid<BLOCK_X/2)
      bcache12p_2x[1]=MULTD(ADD(bcache12p_2x[0],bcache12p_2x[2]),0.5);
    __syncthreads();         
    if (i1tid<BLOCK_Y/2) {
      bcache12p_2x[BCACHE_X*1-i2tid]=
	MULTD(ADD(bcache12p_2x[BCACHE_X*0-i2tid],
		  bcache12p_2x[BCACHE_X*2-i2tid]),0.5);
      if ((skip1.z) && (i2tid<1))
	bcache12p_2x[BCACHE_X*1-i2tid+BLOCK_X]=
	  MULTD(ADD(bcache12p_2x[BCACHE_X*0-i2tid+BLOCK_X],
		    bcache12p_2x[BCACHE_X*2-i2tid+BLOCK_X]),0.5);
    }
    __syncthreads();         
    if ((i1<b_n.y) && (i2<b_n.z) && (i1>=0) && (i2>=0))	
      b[0]=bcache12[BCACHE_X*(i1tid)+i2tid];
    __syncthreads();         
    if (i1tid<ACACHE_Y && i2tid<ACACHE_X) {
      bcache12p_2x[0]=a_c;
    }
    b+=b_n.y*b_n.z;
    __syncthreads();         
  }
  if (xend==n.x && skip1.x){
    if (i1tid<ACACHE_Y && i2tid<BLOCK_X/2)
      bcache12p_2x[1]=MULTD(ADD(bcache12p_2x[0],bcache12p_2x[2]),0.5);
    __syncthreads();         
    if (i1tid<BLOCK_Y/2){
      bcache12p_2x[BCACHE_X*1-i2tid]=
	MULTD(ADD(bcache12p_2x[BCACHE_X*0-i2tid],
		  bcache12p_2x[BCACHE_X*2-i2tid]),0.5);
      if ((skip1.z) && (i2tid<1))
	bcache12p_2x[BCACHE_X*1-i2tid+BLOCK_X]=
	  MULTD(ADD(bcache12p_2x[BCACHE_X*0-i2tid+BLOCK_X],
		    bcache12p_2x[BCACHE_X*2-i2tid+BLOCK_X]),0.5);
    }
    __syncthreads();         
    
    if (xend>1 || !skip0.x) {
      if ((i1<b_n.y) && (i2<b_n.z) && (i1>=0) && (i2>=0))	
	b[0]=bcache12[BCACHE_X*(i1tid)+i2tid];
      b+=b_n.y*b_n.z;    
    }
  }
}


extern "C"{
  void Zcuda(bmgs_interpolate_cuda_gpu)(int k, int skip[3][2],
					const Tcuda* a, const int size[3], 
					Tcuda* b, const int sizeb[3],
					int blocks)
  {
    if (k!=2) assert(0);
    //    int xdiv=MIN(MAX(size[0]/2,1),MAX((4+blocks-1)/blocks,1)); 
    int xdiv=1;

    int gridy=blocks*((sizeb[1]+skip[1][0]+BLOCK_Y-1)/BLOCK_Y);
    
    int gridx=xdiv*((sizeb[2]+skip[2][0]+BLOCK_X-1)/BLOCK_X);
    
    dim3 dimBlock(BLOCK_X,BLOCK_Y); 
    dim3 dimGrid(gridx,gridy);    
    int3 n={size[0],size[1],size[2]};
    int3 skip0={skip[0][0],skip[1][0],skip[2][0]};
    int3 skip1={skip[0][1],skip[1][1],skip[2][1]};
    int3 b_n={2*n.x-2-skip0.x+skip1.x,
	      2*n.y-2-skip0.y+skip1.y,
	      2*n.z-2-skip0.z+skip1.z};
    
        
    Zcuda(interpolate_kernel)<<<dimGrid, dimBlock, 0>>>(a,n, b,b_n,skip0,skip1,xdiv,blocks);
    gpaw_cudaSafeCall(cudaGetLastError());
  }
  
}
#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "interpolate-cuda.cu"


extern "C"{
  double bmgs_interpolate_cuda_cpu(int k, int skip[3][2], const double* a, 
				    const int n[3], double* b,int blocks)
  {
    double *adev,*bdev;
    size_t bsize,asize;
    struct timeval  t0, t1; 
    double flops;
    int b_n[3]={2*n[0]-2-skip[0][0]+skip[0][1],
		2*n[1]-2-skip[1][0]+skip[1][1],
		2*n[2]-2-skip[2][0]+skip[2][1]};
  
    asize=blocks*n[0]*n[1]*n[2];
    bsize=blocks*b_n[0]*b_n[1]*b_n[2];
  
    gpaw_cudaSafeCall(cudaMalloc(&adev,sizeof(double)*asize));
  
    gpaw_cudaSafeCall(cudaMalloc(&bdev,sizeof(double)*bsize));
    gpaw_cudaSafeCall(cudaMemcpy(adev,a,sizeof(double)*asize,
				 cudaMemcpyHostToDevice));
  
    gettimeofday(&t0,NULL);

    bmgs_interpolate_cuda_gpu(k,skip, adev, n, bdev,b_n,blocks);
  
  
    cudaThreadSynchronize();  

    gpaw_cudaSafeCall(cudaGetLastError());

    gettimeofday(&t1,NULL);
  
    gpaw_cudaSafeCall(cudaMemcpy(b,bdev,sizeof(double)*bsize,
				 cudaMemcpyDeviceToHost));
  
    gpaw_cudaSafeCall(cudaFree(adev));
    gpaw_cudaSafeCall(cudaFree(bdev));
  
    flops=(t1.tv_sec*1.0+t1.tv_usec/1000000.0-t0.tv_sec*1.0-t0.tv_usec/1000000.0); 
  
    return flops;

  }

}
#endif
