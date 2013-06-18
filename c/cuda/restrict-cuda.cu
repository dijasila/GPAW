/*  Copyright (C) 2003-2007  CAMP
 *  Please see the accompanying LICENSE file for further information. */

#include <stdio.h>

#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include "gpaw-cuda-int.h"

#ifndef CUGPAWCOMPLEX
#define BLOCK_X_FERMI 16
#define BLOCK_Y_FERMI 4
#define BLOCK_X_KEPLER 32  
#define BLOCK_Y_KEPLER 8

#endif

#ifdef BLOCK_X

#define ACACHE_X  (2*BLOCK_X+1)
#define ACACHE_Y  (2*BLOCK_Y+1)

__global__ void RESTRICT_kernel(const Tcuda* a, const int3 n, 
				       Tcuda* b,const int3 b_n,int xdiv,
				       int blocks)
{
  int i2,i1;
  int i2_x2,i1_x2;
  int xlen;
  Tcuda *acache12p;
  Tcuda *acache12p_2x;  
  Tcuda b_old;
  __shared__ Tcuda Zcuda(acache12)[ACACHE_X*ACACHE_Y];  
  {
    int xx=gridDim.x/xdiv;    
    int xind=blockIdx.x/xx;    
    int base=(blockIdx.x-xind*xx)*BLOCK_X;
    i2=base+threadIdx.x;
    i2_x2=2*base+threadIdx.x;
    
    int yy=gridDim.y/blocks;
    int blocksi=blockIdx.y/yy;    
    base=(blockIdx.y-blocksi*yy)*BLOCK_Y;  
    i1=base+threadIdx.y;
    i1_x2=2*base+threadIdx.y;
    
    xlen=(b_n.x+xdiv-1)/xdiv;

    int xstart=xind*xlen;
    
    if ((b_n.x-xstart) < xlen)
      xlen=b_n.x-xstart;
    
    a+=n.x*n.y*n.z*blocksi+2*xstart*n.y*n.z+i1_x2*n.z+i2_x2;  
    b+=b_n.x*b_n.y*b_n.z*blocksi+xstart*b_n.y*b_n.z+i1*b_n.z+i2;     
  }
  acache12p=Zcuda(acache12)+ACACHE_X*(threadIdx.y)+threadIdx.x;
  acache12p_2x=Zcuda(acache12)+ACACHE_X*(2*threadIdx.y)+2*threadIdx.x;

  acache12p[0]=a[0];
  acache12p[BLOCK_X]=a[BLOCK_X];
  if  (threadIdx.x<1) {
    acache12p[2*BLOCK_X]=a[2*BLOCK_X];
    acache12p[BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[BLOCK_Y*n.z+2*BLOCK_X];
  }

  acache12p[BLOCK_Y*ACACHE_X+0]=a[BLOCK_Y*n.z];
  acache12p[BLOCK_Y*ACACHE_X+BLOCK_X]=a[BLOCK_Y*n.z+BLOCK_X];
  if (threadIdx.y<1) {
    acache12p[2*BLOCK_Y*ACACHE_X]=a[2*BLOCK_Y*n.z];
    acache12p[2*BLOCK_Y*ACACHE_X+BLOCK_X]=a[2*BLOCK_Y*n.z+BLOCK_X];
    if (threadIdx.x<1)
      acache12p[2*BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[2*BLOCK_Y*n.z+2*BLOCK_X];
  }

  __syncthreads();        
      
  b_old=ADD3(MULTD(acache12p_2x[ACACHE_X*1+1],0.0625),
	     MULTD(ADD4(acache12p_2x[ACACHE_X*1+0],
			acache12p_2x[ACACHE_X*1+2],
			acache12p_2x[ACACHE_X*0+1],
			acache12p_2x[ACACHE_X*2+1]),0.03125),
	     MULTD(ADD4(acache12p_2x[ACACHE_X*0+0],
			acache12p_2x[ACACHE_X*0+2],
			acache12p_2x[ACACHE_X*2+0],
			acache12p_2x[ACACHE_X*2+2]),0.015625));  
    /*
  b_old=MULTD(acache12p_2x[ACACHE_X*1+1],0.0625);
  IADD(b_old,MULTD(acache12p_2x[ACACHE_X*1+0], 0.03125));
  IADD(b_old,MULTD(acache12p_2x[ACACHE_X*1+2], 0.03125));
  IADD(b_old,MULTD(acache12p_2x[ACACHE_X*0+1], 0.03125));
  IADD(b_old,MULTD(acache12p_2x[ACACHE_X*2+1], 0.03125));
  IADD(b_old,MULTD(acache12p_2x[ACACHE_X*0+0], 0.015625));
  IADD(b_old,MULTD(acache12p_2x[ACACHE_X*0+2], 0.015625));
  IADD(b_old,MULTD(acache12p_2x[ACACHE_X*2+0], 0.015625));
  IADD(b_old,MULTD(acache12p_2x[ACACHE_X*2+2], 0.015625));
    */
  __syncthreads();         
  for (int i0=0; i0 < xlen; i0++) { 
    
    a+=n.y*n.z;
    acache12p[0]=a[0];
    acache12p[BLOCK_X]=a[BLOCK_X];
    if  (threadIdx.x<1) {
      acache12p[2*BLOCK_X]=a[2*BLOCK_X];
      acache12p[BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[BLOCK_Y*n.z+2*BLOCK_X];
    }
    acache12p[BLOCK_Y*ACACHE_X+0]=a[BLOCK_Y*n.z];
    acache12p[BLOCK_Y*ACACHE_X+BLOCK_X]=a[BLOCK_Y*n.z+BLOCK_X];
    if (threadIdx.y<1) {
      acache12p[2*BLOCK_Y*ACACHE_X]=a[2*BLOCK_Y*n.z];
      acache12p[2*BLOCK_Y*ACACHE_X+BLOCK_X]=a[2*BLOCK_Y*n.z+BLOCK_X];
      if (threadIdx.x<1)
	acache12p[2*BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[2*BLOCK_Y*n.z+2*BLOCK_X];
    }
    
    __syncthreads();
    IADD(b_old,ADD3(MULTD(acache12p_2x[ACACHE_X*1+1],0.125),
		    MULTD(ADD4(acache12p_2x[ACACHE_X*1+0],
			       acache12p_2x[ACACHE_X*1+2],
			       acache12p_2x[ACACHE_X*0+1],
			       acache12p_2x[ACACHE_X*2+1]),0.0625),
		    MULTD(ADD4(acache12p_2x[ACACHE_X*0+0],
			       acache12p_2x[ACACHE_X*0+2],
			       acache12p_2x[ACACHE_X*2+0],
			       acache12p_2x[ACACHE_X*2+2]),0.03125)));  
       /*
    IADD(b_old,MULTD(acache12p_2x[ACACHE_X*1+1], 0.125));
    IADD(b_old,MULTD(acache12p_2x[ACACHE_X*1+0], 0.0625));
    IADD(b_old,MULTD(acache12p_2x[ACACHE_X*1+2], 0.0625));
    IADD(b_old,MULTD(acache12p_2x[ACACHE_X*0+1], 0.0625));
    IADD(b_old,MULTD(acache12p_2x[ACACHE_X*2+1], 0.0625));
    IADD(b_old,MULTD(acache12p_2x[ACACHE_X*0+0], 0.03125));
    IADD(b_old,MULTD(acache12p_2x[ACACHE_X*0+2], 0.03125));
    IADD(b_old,MULTD(acache12p_2x[ACACHE_X*2+0], 0.03125));
    IADD(b_old,MULTD(acache12p_2x[ACACHE_X*2+2], 0.03125));
       */
    __syncthreads();         
    a+=n.y*n.z;
    if (i0 == b_n.x-1) {
      if (i1_x2 < n.y) {
	if (i2_x2 < n.z) {
	  acache12p[0]=a[0];
	  if (i2_x2+BLOCK_X < n.z) {
	    acache12p[BLOCK_X]=a[BLOCK_X];
	    if  (threadIdx.x<1) {
	      if (i2_x2+2*BLOCK_X < n.z) 
		acache12p[2*BLOCK_X]=a[2*BLOCK_X];
	    }
	  }
	}
	
      }
      if (i1_x2+BLOCK_Y < n.y) {
	if (i2_x2 < n.z) {
	  acache12p[BLOCK_Y*ACACHE_X+0]=a[BLOCK_Y*n.z];
	  if (i2_x2+BLOCK_X < n.z) {
	    acache12p[BLOCK_Y*ACACHE_X+BLOCK_X]=a[BLOCK_Y*n.z+BLOCK_X];
	    if  (threadIdx.x<1) {
	      if (i2_x2+2*BLOCK_X < n.z) 
		acache12p[BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[BLOCK_Y*n.z+2*BLOCK_X];
	    }
	  }
	}
      }

      if (threadIdx.y<1) {
	if (i1_x2+2*BLOCK_Y < n.y) {
	  if (i2_x2 < n.z) {
	    acache12p[2*BLOCK_Y*ACACHE_X]=a[2*BLOCK_Y*n.z];
	    if (i2_x2+BLOCK_X < n.z) {
	      acache12p[2*BLOCK_Y*ACACHE_X+BLOCK_X]=a[2*BLOCK_Y*n.z+BLOCK_X];
	      if (threadIdx.x<1)
		if (i2_x2+2*BLOCK_X < n.z) 
		  acache12p[2*BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[2*BLOCK_Y*n.z+2*BLOCK_X];
	    }
	  }
	}
      }
      
    } else {
      acache12p[0]=a[0];
      acache12p[BLOCK_X]=a[BLOCK_X];
      if  (threadIdx.x<1) {
	acache12p[2*BLOCK_X]=a[2*BLOCK_X];
	acache12p[BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[BLOCK_Y*n.z+2*BLOCK_X];
      }
      acache12p[BLOCK_Y*ACACHE_X+0]=a[BLOCK_Y*n.z];
      acache12p[BLOCK_Y*ACACHE_X+BLOCK_X]=a[BLOCK_Y*n.z+BLOCK_X];
      if (threadIdx.y<1) {
	acache12p[2*BLOCK_Y*ACACHE_X]=a[2*BLOCK_Y*n.z];
	acache12p[2*BLOCK_Y*ACACHE_X+BLOCK_X]=a[2*BLOCK_Y*n.z+BLOCK_X];
	if (threadIdx.x<1)
	  acache12p[2*BLOCK_Y*ACACHE_X+2*BLOCK_X]=a[2*BLOCK_Y*n.z+2*BLOCK_X];
      }      
    }
    __syncthreads();  
    
    Tcuda b_new=ADD3(MULTD(acache12p_2x[ACACHE_X*1+1],0.0625),
	       MULTD(ADD4(acache12p_2x[ACACHE_X*1+0],
			  acache12p_2x[ACACHE_X*1+2],
			  acache12p_2x[ACACHE_X*0+1],
			  acache12p_2x[ACACHE_X*2+1]),0.03125),
	       MULTD(ADD4(acache12p_2x[ACACHE_X*0+0],
			  acache12p_2x[ACACHE_X*0+2],
			  acache12p_2x[ACACHE_X*2+0],
			  acache12p_2x[ACACHE_X*2+2]),0.015625));
    /*
    Tcuda b_new=MULTD(acache12p_2x[ACACHE_X*1+1],0.0625);
    IADD(b_new,MULTD(acache12p_2x[ACACHE_X*1+0], 0.03125));
    IADD(b_new,MULTD(acache12p_2x[ACACHE_X*1+2], 0.03125));
    IADD(b_new,MULTD(acache12p_2x[ACACHE_X*0+1], 0.03125));
    IADD(b_new,MULTD(acache12p_2x[ACACHE_X*2+1], 0.03125));
    IADD(b_new,MULTD(acache12p_2x[ACACHE_X*0+0], 0.015625));
    IADD(b_new,MULTD(acache12p_2x[ACACHE_X*0+2], 0.015625));
    IADD(b_new,MULTD(acache12p_2x[ACACHE_X*2+0], 0.015625));
    IADD(b_new,MULTD(acache12p_2x[ACACHE_X*2+2], 0.015625));
    */
    if (i1<b_n.y && i2<b_n.z)
      b[0]=ADD(b_old,b_new);
    b_old=b_new;    
    __syncthreads();             
    b+=b_n.y*b_n.z;
  }
}

#else
#define BLOCK_X   (BLOCK_X_FERMI)
#define BLOCK_Y   (BLOCK_Y_FERMI)
#  define RESTRICT_kernel Zcuda(restrict_kernel_fermi)
#  include "restrict-cuda.cu"
#  undef RESTRICT_kernel
#undef BLOCK_X
#undef BLOCK_Y
#define BLOCK_X   (BLOCK_X_KEPLER)
#define BLOCK_Y   (BLOCK_Y_KEPLER)
#  define RESTRICT_kernel Zcuda(restrict_kernel_kepler)
#  include "restrict-cuda.cu"
#  undef RESTRICT_kernel
#undef BLOCK_X
#undef BLOCK_Y

extern "C"{
  
  void Zcuda(bmgs_restrict_cuda_gpu)(int k,
				      const Tcuda* a, const int size[3], 
				      Tcuda* b, const int sizeb[3],
				      int blocks)
  {
    if (k!=2) assert(0);

    dim3 dimBlock(1,1);

    switch (_gpaw_cuda_dev_prop.major)
      {
      case 0:
      case 1:
      case 2:
	dimBlock.x = BLOCK_X_FERMI;
	dimBlock.y = BLOCK_Y_FERMI;
	break;
      default:
	dimBlock.x = BLOCK_X_KEPLER;
	dimBlock.y = BLOCK_Y_KEPLER;
      }    
    
    int xdiv=MIN(MAX(sizeb[0]/2,1),MAX((4+blocks-1)/blocks,1)); 
    
    int gridy=blocks*((sizeb[1]+dimBlock.y-1)/dimBlock.y);
    
    int gridx=xdiv*((sizeb[2]+dimBlock.x-1)/dimBlock.x);
    
    dim3 dimGrid(gridx,gridy);    
    
    int3 n={size[0], size[1], size[2]};
    int3 b_n={sizeb[0], sizeb[1], sizeb[2]};
    
    switch (_gpaw_cuda_dev_prop.major)
      {
      case 0:
      case 1:
      case 2:
	Zcuda(restrict_kernel_fermi)<<<dimGrid, dimBlock, 0>>>
	  (a, n ,b, b_n, xdiv, blocks);
	break;
      default:
	Zcuda(restrict_kernel_kepler)<<<dimGrid, dimBlock, 0>>>
	  (a, n, b, b_n, xdiv, blocks);
    
      }    

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
#endif

