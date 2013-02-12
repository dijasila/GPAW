/*  Copyright (C) 2003-2007  CAMP
 *  Please see the accompanying LICENSE file for further information. */

#include <stdio.h>

#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include "gpaw-cuda-int.h"



#ifndef CUGPAWCOMPLEX
#define BLOCK_X 16
#define BLOCK_Y 8

#define ACACHE_X (2*(BLOCK_X)+1)
#define ACACHE_Y (2*(BLOCK_Y)+1)
#endif


__global__ void Zcuda(restrict_kernel)(const Tcuda* a, int3 n, Tcuda* b,int3 b_n,int xdiv,int blocks)
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

  int3 a_size={n.x*n.y*n.z,n.y*n.z,n.z};

  int3 b_size={b_n.x*b_n.y*b_n.z,b_n.y*b_n.z,b_n.z};

  __shared__ Tcuda acache12[ACACHE_Y*ACACHE_X];
  
  Tcuda *acache12p;
  Tcuda *acache12p_2x;

  Tcuda b_new;
  Tcuda b_old;
  
  int xlen=(b_n.x+xdiv-1)/xdiv;
  int xstart=xind*xlen;
  int xend=MIN(xstart+xlen,b_n.x);
  xlen=xend-xstart;

  a+=a_size.x*blocksi+2*xstart*a_size.y+(i1base*2+i1tid)*a_size.z+i2base*2+i2tid;  

  b+=b_size.x*blocksi+xstart*b_size.y+i1*b_size.z+i2;  
   
  acache12p=acache12+ACACHE_X*(i1tid)+i2tid;
  acache12p_2x=acache12+ACACHE_X*(2*i1tid)+2*i2tid;
  
  acache12p[0]=a[0];
  acache12p[BLOCK_X]=a[BLOCK_X];
  if  (i2tid<1) {
    acache12p[2*BLOCK_X]=a[2*BLOCK_X];
    acache12p[BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[BLOCK_Y*a_size.z+2*BLOCK_X];
  }
  acache12p[BLOCK_Y*ACACHE_X]=a[BLOCK_Y*a_size.z];
  acache12p[BLOCK_Y*ACACHE_X+BLOCK_X]=a[BLOCK_Y*a_size.z+BLOCK_X];
  if (i1tid<1) {
    acache12p[2*BLOCK_Y*ACACHE_X]=a[2*BLOCK_Y*a_size.z];
    acache12p[2*BLOCK_Y*ACACHE_X+BLOCK_X]=a[2*BLOCK_Y*a_size.z+BLOCK_X];
    if (i2tid<1)
      acache12p[2*BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[2*BLOCK_Y*a_size.z+2*BLOCK_X];
  }
  
  __syncthreads();        
  b_old=ADD(MULTD(acache12p_2x[ACACHE_X*1+1],0.0625),
	    ADD(MULTD(ADD(ADD(acache12p_2x[ACACHE_X*1+0],
			      acache12p_2x[ACACHE_X*1+2]),
			  ADD(acache12p_2x[ACACHE_X*0+1],
			      acache12p_2x[ACACHE_X*2+1])),0.03125),
		MULTD(ADD(ADD(acache12p_2x[ACACHE_X*0+0],
			      acache12p_2x[ACACHE_X*0+2]),
			  ADD(acache12p_2x[ACACHE_X*2+0],
			      acache12p_2x[ACACHE_X*2+2])),0.015625)));  

  __syncthreads();         
  for (int i0=xstart; i0 < xend; i0++) { 
    
    a+=a_size.y;
    acache12p[0]=a[0];
    acache12p[BLOCK_X]=a[BLOCK_X];
    if  (i2tid<1) {
      acache12p[2*BLOCK_X]=a[2*BLOCK_X];
      acache12p[BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[BLOCK_Y*a_size.z+2*BLOCK_X];
    }
    acache12p[BLOCK_Y*ACACHE_X]=a[BLOCK_Y*a_size.z];
    acache12p[BLOCK_Y*ACACHE_X+BLOCK_X]=a[BLOCK_Y*a_size.z+BLOCK_X];
    if (i1tid<1) {
      acache12p[2*BLOCK_Y*ACACHE_X]=a[2*BLOCK_Y*a_size.z];
      acache12p[2*BLOCK_Y*ACACHE_X+BLOCK_X]=a[2*BLOCK_Y*a_size.z+BLOCK_X];
      if (i2tid<1)
	acache12p[2*BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[2*BLOCK_Y*a_size.z+2*BLOCK_X];
    }
    
    __syncthreads();
    IADD(b_old,ADD(MULTD(acache12p_2x[ACACHE_X*1+1],0.125),
		   ADD(MULTD(ADD(ADD(acache12p_2x[ACACHE_X*1+0],
				     acache12p_2x[ACACHE_X*1+2]),
				 ADD(acache12p_2x[ACACHE_X*0+1],
				     acache12p_2x[ACACHE_X*2+1])),0.0625),
		       MULTD(ADD(ADD(acache12p_2x[ACACHE_X*0+0],
				     acache12p_2x[ACACHE_X*0+2]),
				 ADD(acache12p_2x[ACACHE_X*2+0],
				     acache12p_2x[ACACHE_X*2+2])),0.03125))));  
    
    __syncthreads();         
    a+=a_size.y;
    if (i0==b_n.x-1) {
      if (i1base*2+i1tid<n.y) {
	if (i2base*2+i2tid<n.z) {
	  acache12p[0]=a[0];
	  if (i2base*2+BLOCK_X+i2tid<n.z) {
	    acache12p[BLOCK_X]=a[BLOCK_X];
	    if  (i2tid<1) {
	      if (i2base*2+2*BLOCK_X+i2tid<n.z) 
		acache12p[2*BLOCK_X]=a[2*BLOCK_X];
	    }
	  }
	}
	
      }
      if (i1base*2+BLOCK_Y+i1tid<n.y) {
	if (i2base*2+i2tid<n.z) {
	  acache12p[BLOCK_Y*ACACHE_X]=a[BLOCK_Y*a_size.z];
	  if (i2base*2+BLOCK_X+i2tid<n.z) {
	    acache12p[BLOCK_Y*ACACHE_X+BLOCK_X]=a[BLOCK_Y*a_size.z+BLOCK_X];
	    if  (i2tid<1) {
	      if (i2base*2+2*BLOCK_X+i2tid<n.z) 
		acache12p[BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[BLOCK_Y*a_size.z+2*BLOCK_X];
	    }
	  }
	}
      }

      if (i1tid<1) {
	if (i1base*2+2*BLOCK_Y+i1tid<n.y) {
	  if (i2base*2+i2tid<n.z) {
	    acache12p[2*BLOCK_Y*ACACHE_X]=a[2*BLOCK_Y*a_size.z];
	    if (i2base*2+BLOCK_X+i2tid<n.z) {
	      acache12p[2*BLOCK_Y*ACACHE_X+BLOCK_X]=a[2*BLOCK_Y*a_size.z+BLOCK_X];
	      if (i2tid<1)
		if (i2base*2+2*BLOCK_X+i2tid<n.z) 
		  acache12p[2*BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[2*BLOCK_Y*a_size.z+2*BLOCK_X];
	    }
	  }
	}
      }
      
    } else {
      acache12p[0]=a[0];
      acache12p[BLOCK_X]=a[BLOCK_X];
      if  (i2tid<1) {
	acache12p[2*BLOCK_X]=a[2*BLOCK_X];
	acache12p[BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[BLOCK_Y*a_size.z+2*BLOCK_X];
      }
      acache12p[BLOCK_Y*ACACHE_X]=a[BLOCK_Y*a_size.z];
      acache12p[BLOCK_Y*ACACHE_X+BLOCK_X]=a[BLOCK_Y*a_size.z+BLOCK_X];
      if (i1tid<1) {
	acache12p[2*BLOCK_Y*ACACHE_X]=a[2*BLOCK_Y*a_size.z];
	acache12p[2*BLOCK_Y*ACACHE_X+BLOCK_X]=a[2*BLOCK_Y*a_size.z+BLOCK_X];
	if (i2tid<1)
	  acache12p[2*BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[2*BLOCK_Y*a_size.z+2*BLOCK_X];
      }      
    }
    __syncthreads();         
    b_new=ADD(MULTD(acache12p_2x[ACACHE_X*1+1],0.0625),
	      ADD(MULTD(ADD(ADD(acache12p_2x[ACACHE_X*1+0],
				acache12p_2x[ACACHE_X*1+2]),
			    ADD(acache12p_2x[ACACHE_X*0+1],
				acache12p_2x[ACACHE_X*2+1])),0.03125),
		  MULTD(ADD(ADD(acache12p_2x[ACACHE_X*0+0],
				acache12p_2x[ACACHE_X*0+2]),
			    ADD(acache12p_2x[ACACHE_X*2+0],
			    acache12p_2x[ACACHE_X*2+2])),0.015625)));
    
    if (i1<b_n.y && i2<b_n.z)
      b[0]=ADD(b_old,b_new);
    b_old=b_new;    
    __syncthreads();             
    b+=b_size.y;
  }
}




extern "C"{
  
  void Zcuda(bmgs_restrict_cuda_gpu)(int k,
				      const Tcuda* a, const int size[3], 
				      Tcuda* b, const int sizeb[3],
				      int blocks)
  {
    if (k!=2) assert(0);
    int xdiv=MIN(MAX(sizeb[0]/2,1),MAX((4+blocks-1)/blocks,1)); 
    
    int gridy=blocks*((sizeb[1]+BLOCK_Y-1)/BLOCK_Y);
    
    int gridx=xdiv*((sizeb[2]+BLOCK_X-1)/BLOCK_X);
    
    dim3 dimBlock(BLOCK_X,BLOCK_Y); 
    dim3 dimGrid(gridx,gridy);    
    int3 n={size[0],size[1],size[2]};
    int3 b_n={sizeb[0],sizeb[1],sizeb[2]};
    
    Zcuda(restrict_kernel)<<<dimGrid, dimBlock, 0>>>(a,n,b,b_n,xdiv,blocks);
    gpaw_cudaSafeCall(cudaGetLastError());
  }
  

}
#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "restrict-cuda.cu"

extern "C"{
  double bmgs_restrict_cuda_cpu(int k, double* a, const int n[3], double* b, int blocks)
  {
    double *adev,*bdev;
    size_t bsize,asize;
    struct timeval  t0, t1; 
    double flops;
    int   b_n[3]={(n[0]-1)/2,(n[1]-1)/2,(n[2]-1)/2};
  
    asize=n[0]*n[1]*n[2];
    bsize=b_n[0]*b_n[1]*b_n[2];
    
    gpaw_cudaSafeCall(cudaMalloc(&adev,sizeof(double)*asize*blocks));
    
    gpaw_cudaSafeCall(cudaMalloc(&bdev,sizeof(double)*bsize*blocks));
    
    gpaw_cudaSafeCall(cudaMemcpy(adev,a,sizeof(double)*asize*blocks,
				 cudaMemcpyHostToDevice));
  
    gettimeofday(&t0,NULL);
    bmgs_restrict_cuda_gpu(k, adev, n, bdev,b_n,blocks);
  
  
    cudaThreadSynchronize();  

    gpaw_cudaSafeCall(cudaGetLastError());
    
    gettimeofday(&t1,NULL);
  
    gpaw_cudaSafeCall(cudaMemcpy(b,bdev,sizeof(double)*bsize*blocks,
				 cudaMemcpyDeviceToHost));
  
    gpaw_cudaSafeCall(cudaFree(adev));
    gpaw_cudaSafeCall(cudaFree(bdev));
  
    flops=(t1.tv_sec*1.0+t1.tv_usec/1000000.0-t0.tv_sec*1.0-t0.tv_usec/1000000.0); 
  
    return flops;

  }

}
#endif

