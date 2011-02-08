// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include <cuComplex.h>

#include "gpaw-cuda-int.h"

#ifndef MYJ
#ifndef CUGPAWCOMPLEX

__constant__ long c_offsets[FD_MAXCOEFS];
__constant__ double c_coefs[FD_MAXCOEFS];
__constant__ int c_offsets12[FD_MAXCOEFS];
__constant__ double c_coefs12[FD_MAXCOEFS];
__constant__ double c_coefs0[FD_MAXJ+1];

#endif
#endif


#ifdef MYJ
#undef  FD_ACACHE_Y
#define FD_ACACHE_Y  (FD_BLOCK_Y+MYJ)



#undef CACHE_LOOP
#define CACHE_LOOP(i0e,i1s,i1e,i2s,i2e)					\
  for (c=0;c<MYJ;c++)							\
    acache0[c]=acache0[c+1];						\
  acache0[MYJ]=(i0e) ? MAKED(0) : a[(MYJ/2)*sizeyz];			\
  acache12p[0]=acache0[MYJ/2];						\
  if  ((i2tid<MYJ/2)) {							\
    acache12p[-MYJ/2]=(i2s) ? MAKED(0) : a[-MYJ/2];			\
    acache12p[FD_BLOCK_X]=(i2e) ? MAKED(0) : a[FD_BLOCK_X];		\
  }									\
  if  (i1tid<MYJ/2){							\
    acache12p[-FD_ACACHE_X*MYJ/2]=(i1s) ? MAKED(0) : a[-sizez*MYJ/2];	\
    acache12p[FD_ACACHE_X*FD_BLOCK_Y]=(i1e) ? MAKED(0) : a[sizez*FD_BLOCK_Y]; \
  }									\
  __syncthreads();							\
  x = MAKED(0);								\
  

#undef APPLY_LOOP
#define APPLY_LOOP(i0e)							\
  for (c = 0; c < ncoefs12; c++)					\
    IADD(x , MULTD(acache12p[c_offsets12[c]] , c_coefs12[c]));		\
  for (c = 0; c < MYJ+1; c++)						\
    IADD(x , MULTD(acache0[c] , c_coefs0[c]));				\
  if (!(i0e))								\
    b[0] = x;								\
  b+=c_n.y*c_n.z;							\
  a+=sizeyz;								\
  __syncthreads();							\


__global__ void FD_kernel_bc(int ncoefs,int ncoefs12,int ncoefs0,
			     const Tcuda* a,Tcuda* b,const long3 c_n,
			     const long3 c_j,const int3 c_jb,int blocks)
{
  
  int i2bl=blockIdx.x/FD_XDIV;
  int xind=blockIdx.x-FD_XDIV*i2bl;
  int i2tid=threadIdx.x;
  int i2=i2bl*FD_BLOCK_X+i2tid;

  int i1bl=blockIdx.y/blocks;
  int blocksi=blockIdx.y-blocks*i1bl;

  int i1tid=threadIdx.y;
  int i1=i1bl*FD_BLOCK_Y+i1tid;

  __shared__ Tcuda acache12[FD_ACACHE_Y*FD_ACACHE_X];

  Tcuda acache0[MYJ+1];
  Tcuda x;
  int c;
  Tcuda *acache12p;
  int sizez=c_jb.z+c_n.z;  
  int sizeyz=c_j.y+c_n.y*sizez;
  
  int xlen=(c_n.x+FD_XDIV-1)/FD_XDIV;
  int xstart=xind*xlen;
  int xend=MIN(xstart+xlen,c_n.x);
  int xendbc=MIN(xend,c_n.x-MYJ/2);
  
  a+=(c_j.x+c_n.x*sizeyz)*blocksi;
  b+=(c_n.x*c_n.y*c_n.z)*blocksi;
  
  a+=xstart*sizeyz+i1*sizez+i2;
  b+=xstart*c_n.y*c_n.z+i1*c_n.z+i2;

  acache12p=acache12+FD_ACACHE_X*(i1tid+MYJ/2)+i2tid+MYJ/2;

  for (c=1;c<MYJ+1;c++)
    acache0[c]=MAKED(0);
  
  int borders=0;
  if (i1bl==0)                     borders|=(1 << 0);
  if (i1bl==(gridDim.y-1)/blocks)  borders|=(1 << 1);
  if (i2bl==0)                     borders|=(1 << 2);
  if (i2bl==(gridDim.x-1)/FD_XDIV) borders|=(1 << 3); 

  if ((i2<c_n.z)&&(i1<c_n.y)) {
    for (c=1;c<MYJ+1;c++){
      int xind2=(c-1-MYJ/2);
      if ((xind2+xstart>=0)&&(xind2+xstart<c_n.x))
	acache0[c]=a[xind2*(sizeyz)];
    }
  }
  
  switch(borders){    
  case 0x0:
    for (int i0=xstart; i0 < xendbc; i0++) {
      CACHE_LOOP(0,0,0,0,0);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(0);
    }
    for (int i0=xendbc; i0 < xend ; i0++) {
      CACHE_LOOP(1,0,0,0,0);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(0);
    }
    break;

  case 0x1:
    for (int i0=xstart; i0 < xendbc; i0++) {
      CACHE_LOOP(0,1,0,0,0);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(0);
    }
    for (int i0=xendbc; i0 < xend ; i0++) {
      CACHE_LOOP(1,1,0,0,0);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(0);
    }    
    break;
    
  case 0x2:
    for (int i0=xstart; i0 < xendbc; i0++) {
      CACHE_LOOP(i1>=c_n.y,0,1,0,0);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(i1>=c_n.y);
    }
    for (int i0=xendbc; i0 < xend ; i0++) {
      CACHE_LOOP(1,0,1,0,0);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(i1>=c_n.y);
    }    
    break;
  case 0x4:
    for (int i0=xstart; i0 < xendbc; i0++) {
      CACHE_LOOP(0,0,0,1,0);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(0);
    }
    for (int i0=xendbc; i0 < xend ; i0++) {
      CACHE_LOOP(1,0,0,1,0);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(0);
    }    
    break;
    
  case 0x5:
    for (int i0=xstart; i0 < xendbc; i0++) {
      CACHE_LOOP(0,1,0,1,0);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(0);
    }
    for (int i0=xendbc; i0 < xend ; i0++) {
      CACHE_LOOP(1,1,0,1,0);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(0);
    }    
    break;
  case 0x6:
    for (int i0=xstart; i0 < xendbc; i0++) {
      CACHE_LOOP(i1>=c_n.y,0,1,1,0);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(i1>=c_n.y);
    }
    for (int i0=xendbc; i0 < xend ; i0++) {
      CACHE_LOOP(1,0,1,1,0);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(i1>=c_n.y);
    }    
    break;
  case 0x8:    
    for (int i0=xstart; i0 < xendbc; i0++) {
      CACHE_LOOP(i2>=c_n.z,0,0,0,1);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(i2>=c_n.z);
    }
    for (int i0=xendbc; i0 < xend ; i0++) {
      CACHE_LOOP(1,0,0,0,1);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(i2>=c_n.z);
    }    
    break;
  case 0x9:    
    for (int i0=xstart; i0 < xendbc; i0++) {
      CACHE_LOOP(i2>=c_n.z,1,0,0,1);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(i2>=c_n.z);
    }
    for (int i0=xendbc; i0 < xend ; i0++) {
      CACHE_LOOP(1,1,0,0,1);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP(i2>=c_n.z);
    }    
    break;
  case 0xA:    
    for (int i0=xstart; i0 < xendbc; i0++) {
      CACHE_LOOP((i2>=c_n.z)||(i1>=c_n.y),0,1,0,1);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP((i2>=c_n.z)||(i1>=c_n.y));
    }
    for (int i0=xendbc; i0 < xend ; i0++) {
      CACHE_LOOP(1,0,1,0,1);
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP((i2>=c_n.z)||(i1>=c_n.y));
    }    
    break;
  default:
    for (int i0=xstart; i0 < xendbc; i0++) {
      CACHE_LOOP((i2>=c_n.z)||(i1>=c_n.y),
	      ((i1-MYJ/2)<0),((i1+FD_BLOCK_Y)>=c_n.y),
	      ((i2-MYJ/2)<0),((i2+FD_BLOCK_X)>=c_n.z));
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP((i2>=c_n.z)||(i1>=c_n.y));
    }
    for (int i0=xendbc; i0 < xend ; i0++) {
      CACHE_LOOP(1,
	      ((i1-MYJ/2)<0),((i1+FD_BLOCK_Y)>=c_n.y),
	      ((i2-MYJ/2)<0),((i2+FD_BLOCK_X)>=c_n.z));
#include"fd-cuda-pragmas.cu"
      APPLY_LOOP((i2>=c_n.z)||(i1>=c_n.y));
    }    
    break;
  }
}

__global__ void FD_kernel(int ncoefs,int ncoefs12,int ncoefs0,const Tcuda* a,
			  Tcuda* b,const long3 c_n,const long3 c_j,
			  const int3 c_jb,int blocks)
{
  
  int i2bl=blockIdx.x/FD_XDIV;
  int xind=blockIdx.x-FD_XDIV*i2bl;
  int i2tid=threadIdx.x;
  int i2=i2bl*FD_BLOCK_X+i2tid;

  int i1bl=blockIdx.y/blocks;
  int blocksi=blockIdx.y-blocks*i1bl;

  int i1tid=threadIdx.y;
  int i1=i1bl*FD_BLOCK_Y+i1tid;

  __shared__ Tcuda acache12[FD_ACACHE_Y*FD_ACACHE_X];

  Tcuda acache0[MYJ+1];
  Tcuda x;
  int c;
  Tcuda *acache12p;
  int sizez=c_jb.z+c_n.z;  
  int sizeyz=c_j.y+c_n.y*sizez;
  
  int xlen=(c_n.x+FD_XDIV-1)/FD_XDIV;
  int xstart=xind*xlen;
  int xend=MIN(xstart+xlen,c_n.x);
  
  a+=(c_j.x+c_n.x*sizeyz)*blocksi;
  b+=(c_n.x*c_n.y*c_n.z)*blocksi;
  
  a+=xstart*sizeyz+i1*sizez+i2;
  b+=xstart*c_n.y*c_n.z+i1*c_n.z+i2;

  acache12p=acache12+FD_ACACHE_X*(i1tid+MYJ/2)+i2tid+MYJ/2;
  for (c=1;c<MYJ+1;c++){
    acache0[c]=a[(c-1-MYJ/2)*(sizeyz)];
  }
  for (int i0=xstart; i0 < xend; i0++) {  
    for (c=0;c<MYJ;c++){
      acache0[c]=acache0[c+1];
    }
    acache0[MYJ]=a[(MYJ/2)*sizeyz];
    
    acache12p[0]=acache0[MYJ/2];
    if  (i2tid<MYJ/2){
      acache12p[-MYJ/2]=a[-MYJ/2];
      acache12p[FD_BLOCK_X]=a[FD_BLOCK_X];
    }
    if  (i1tid<MYJ/2){
      acache12p[-FD_ACACHE_X*MYJ/2]=a[-sizez*MYJ/2];
      acache12p[FD_ACACHE_X*FD_BLOCK_Y]=a[sizez*FD_BLOCK_Y];      
    }
    __syncthreads();         
    
    x = MAKED(0.0);
#include"fd-cuda-pragmas.cu"
    for (c = 0; c < ncoefs12; c++){
      IADD(x , MULTD(acache12p[c_offsets12[c]] , c_coefs12[c]));
    }	
    for (c = 0; c < MYJ+1; c++){	  
      IADD(x , MULTD(acache0[c] , c_coefs0[c]));
    }	    
    for (c = 0; c < ncoefs; c++){	  
      IADD(x , MULTD(a[c_offsets[c]] , c_coefs[c]));
    }	

    if ((i1<c_n.y) && (i2<c_n.z)) {
      b[0] = x;
    }

    b+=c_n.y*c_n.z;
    a+=sizeyz;
    __syncthreads();         
  }
  
}



#else
#define MYJ  2
#  define FD_kernel Zcuda(fd_kernel2)
#  define FD_kernel_bc Zcuda(fd_kernel2_bc)
#  include "fd-cuda.cu"
#  undef FD_kernel
#  undef FD_kernel_bc
#  undef MYJ
#define MYJ  4
#  define FD_kernel Zcuda(fd_kernel4)
#  define FD_kernel_bc Zcuda(fd_kernel4_bc)
#  include "fd-cuda.cu"
#  undef FD_kernel
#  undef FD_kernel_bc
#  undef MYJ
#define MYJ  6
#  define FD_kernel Zcuda(fd_kernel6)
#  define FD_kernel_bc Zcuda(fd_kernel6_bc)
#  include "fd-cuda.cu"
#  undef FD_kernel
#  undef FD_kernel_bc
#  undef MYJ
#define MYJ  8
#  define FD_kernel Zcuda(fd_kernel8)
#  define FD_kernel_bc Zcuda(fd_kernel8_bc)
#  include "fd-cuda.cu"
#  undef FD_kernel
#  undef FD_kernel_bc
#  undef MYJ
#define MYJ  10
#  define FD_kernel Zcuda(fd_kernel10)
#  define FD_kernel_bc Zcuda(fd_kernel10_bc)
#  include "fd-cuda.cu"
#  undef FD_kernel
#  undef FD_kernel_bc
#  undef MYJ


extern "C" {



  void Zcuda(bmgs_fd_cuda_gpu)(const bmgsstencil_gpu* s_gpu, 
			       const Tcuda* adev, Tcuda* bdev,int blocks)  
  {
    int3 jb;
    

    long3 hc_n;
    long3 hc_j;    
    hc_n.x=s_gpu->n[0];    hc_n.y=s_gpu->n[1];    hc_n.z=s_gpu->n[2];
    hc_j.x=s_gpu->j[0];    hc_j.y=s_gpu->j[1];    hc_j.z=s_gpu->j[2];

    //hc_j.x=0;    hc_j.y=0;    hc_j.z=0;

    jb.z=hc_j.z;
    jb.y=hc_j.y/(hc_j.z+hc_n.z);



    if (s_gpu->ncoefs>0){
      gpaw_cudaSafeCall(cudaMemcpyToSymbolAsync(c_offsets,s_gpu->offsets_gpu,
					   sizeof(long)*s_gpu->ncoefs,0,
					   cudaMemcpyDeviceToDevice,0));
      gpaw_cudaSafeCall(cudaMemcpyToSymbolAsync(c_coefs,s_gpu->coefs_gpu,
					   sizeof(double)*s_gpu->ncoefs,0,
					   cudaMemcpyDeviceToDevice,0));
    }
    gpaw_cudaSafeCall(cudaMemcpyToSymbolAsync(c_offsets12,s_gpu->offsets12_gpu,
					 sizeof(int)*s_gpu->ncoefs12,0,
					 cudaMemcpyDeviceToDevice,0));
    gpaw_cudaSafeCall(cudaMemcpyToSymbolAsync(c_coefs12,s_gpu->coefs12_gpu,
					 sizeof(double)*s_gpu->ncoefs12,0,
					 cudaMemcpyDeviceToDevice,0));
    /*gpaw_cudaSafeCall(cudaMemcpyToSymbolAsync(c_offsets0,s_gpu->offsets0_gpu,
					 sizeof(int)*s_gpu->ncoefs0,0,
					 cudaMemcpyDeviceToDevice,0));*/
    gpaw_cudaSafeCall(cudaMemcpyToSymbolAsync(c_coefs0,s_gpu->coefs0_gpu,
					 sizeof(double)*s_gpu->ncoefs0,0,
					 cudaMemcpyDeviceToDevice,0));


    int gridx=FD_XDIV*MAX((s_gpu->n[2]+FD_BLOCK_X-1)/FD_BLOCK_X,1);
    int gridy=blocks*MAX((s_gpu->n[1]+FD_BLOCK_Y-1)/FD_BLOCK_Y,1);
    
    dim3 dimBlock(FD_BLOCK_X,FD_BLOCK_Y); 
    dim3 dimGrid(gridx,gridy);    

    adev+=(hc_j.x+hc_j.y+hc_j.z)/2;


    if (s_gpu->ncoefs0<=3)
      Zcuda(fd_kernel2)<<<dimGrid, dimBlock, 0>>>(s_gpu->ncoefs,s_gpu->ncoefs12,s_gpu->ncoefs0,adev,bdev,hc_n,hc_j,jb,blocks);    
    else if (s_gpu->ncoefs0<=5)
      Zcuda(fd_kernel4)<<<dimGrid, dimBlock, 0>>>(s_gpu->ncoefs,s_gpu->ncoefs12,s_gpu->ncoefs0,adev,bdev,hc_n,hc_j,jb,blocks);    
    else if (s_gpu->ncoefs0<=7)
      Zcuda(fd_kernel6)<<<dimGrid, dimBlock, 0>>>(s_gpu->ncoefs,s_gpu->ncoefs12,s_gpu->ncoefs0,adev,bdev,hc_n,hc_j,jb,blocks);    
    else if (s_gpu->ncoefs0<=9)
      Zcuda(fd_kernel8)<<<dimGrid, dimBlock, 0>>>(s_gpu->ncoefs,s_gpu->ncoefs12,s_gpu->ncoefs0,adev,bdev,hc_n,hc_j,jb,blocks);    
    else if (s_gpu->ncoefs0<=11)
      Zcuda(fd_kernel10)<<<dimGrid, dimBlock, 0>>>(s_gpu->ncoefs,s_gpu->ncoefs12,s_gpu->ncoefs0,adev,bdev,hc_n,hc_j,jb,blocks);    
    gpaw_cudaSafeCall(cudaGetLastError());

  }



  void Zcuda(bmgs_fd_cuda_gpu_bc)(const bmgsstencil_gpu* s_gpu, 
				  const Tcuda* adev, Tcuda* bdev,int blocks)  
  {
    int3 jb;
    

    long3 hc_n;
    long3 hc_j;    
    hc_n.x=s_gpu->n[0];    hc_n.y=s_gpu->n[1];    hc_n.z=s_gpu->n[2];
    //     hc_j.x=s_gpu->j[0];    hc_j.y=s_gpu->j[1];    hc_j.z=s_gpu->j[2];

    hc_j.x=0;    hc_j.y=0;    hc_j.z=0;
    
    jb.z=hc_j.z;
    jb.y=hc_j.y/(hc_j.z+hc_n.z);


    if (s_gpu->ncoefs>0){
      gpaw_cudaSafeCall(cudaMemcpyToSymbolAsync(c_offsets,s_gpu->offsets_gpu,
					   sizeof(long)*s_gpu->ncoefs,0,
					   cudaMemcpyDeviceToDevice,0));
      gpaw_cudaSafeCall(cudaMemcpyToSymbolAsync(c_coefs,s_gpu->coefs_gpu,
					   sizeof(double)*s_gpu->ncoefs,0,
					   cudaMemcpyDeviceToDevice,0));
    }
    gpaw_cudaSafeCall(cudaMemcpyToSymbolAsync(c_offsets12,s_gpu->offsets12_gpu,
					 sizeof(int)*s_gpu->ncoefs12,0,
					 cudaMemcpyDeviceToDevice,0));
    gpaw_cudaSafeCall(cudaMemcpyToSymbolAsync(c_coefs12,s_gpu->coefs12_gpu,
					 sizeof(double)*s_gpu->ncoefs12,0,
					 cudaMemcpyDeviceToDevice,0));
    /*gpaw_cudaSafeCall(cudaMemcpyToSymbolAsync(c_offsets0,s_gpu->offsets0_gpu,
					 sizeof(int)*s_gpu->ncoefs0,0,
					 cudaMemcpyDeviceToDevice,0));*/
    gpaw_cudaSafeCall(cudaMemcpyToSymbolAsync(c_coefs0,s_gpu->coefs0_gpu,
					 sizeof(double)*s_gpu->ncoefs0,0,
					 cudaMemcpyDeviceToDevice,0));


    int gridx=FD_XDIV*MAX((s_gpu->n[2]+FD_BLOCK_X-1)/FD_BLOCK_X,1);
    int gridy=blocks*MAX((s_gpu->n[1]+FD_BLOCK_Y-1)/FD_BLOCK_Y,1);
    
    dim3 dimBlock(FD_BLOCK_X,FD_BLOCK_Y); 
    dim3 dimGrid(gridx,gridy);    

    adev+=(hc_j.x+hc_j.y+hc_j.z)/2;

    if (s_gpu->ncoefs0<=3)
      Zcuda(fd_kernel2_bc)<<<dimGrid, dimBlock, 0>>>(s_gpu->ncoefs,s_gpu->ncoefs12,s_gpu->ncoefs0,adev,bdev,hc_n,hc_j,jb,blocks);    
    else if (s_gpu->ncoefs0<=5)
      Zcuda(fd_kernel4_bc)<<<dimGrid, dimBlock, 0>>>(s_gpu->ncoefs,s_gpu->ncoefs12,s_gpu->ncoefs0,adev,bdev,hc_n,hc_j,jb,blocks);    
    else if (s_gpu->ncoefs0<=7)
      Zcuda(fd_kernel6_bc)<<<dimGrid, dimBlock, 0>>>(s_gpu->ncoefs,s_gpu->ncoefs12,s_gpu->ncoefs0,adev,bdev,hc_n,hc_j,jb,blocks);    
    else if (s_gpu->ncoefs0<=9)
      Zcuda(fd_kernel8_bc)<<<dimGrid, dimBlock, 0>>>(s_gpu->ncoefs,s_gpu->ncoefs12,s_gpu->ncoefs0,adev,bdev,hc_n,hc_j,jb,blocks);    
    else if (s_gpu->ncoefs0<=11)
      Zcuda(fd_kernel10_bc)<<<dimGrid, dimBlock, 0>>>(s_gpu->ncoefs,s_gpu->ncoefs12,s_gpu->ncoefs0,adev,bdev,hc_n,hc_j,jb,blocks);    
    gpaw_cudaSafeCall(cudaGetLastError());

  }

}

#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "fd-cuda.cu"

extern "C" {

  bmgsstencil_gpu bmgs_stencil_to_gpu(const bmgsstencil* s)  
  {
    bmgsstencil_gpu s_gpu;/*
={s->ncoefs,NULL,NULL,0,NULL,NULL,0,NULL,NULL,
			   {s->n[0],s->n[1],s->n[2]},
			   {s->j[0],s->j[1],s->j[2]}};
			  */
    long offsets[s->ncoefs];
    int  offsets12[s->ncoefs];
    int  offsets0[FD_MAXJ+1];
    double coefs[s->ncoefs],coefs12[s->ncoefs],coefs0[FD_MAXJ+1];
    long ncoefs=0,ncoefs12=0,ncoefs0=0;

    int n2=(s->n[2]+s->j[2]);
    int n1=s->j[1]+s->n[1]*n2;

    int jb[3];
    
    jb[2]=s->j[2];
    jb[1]=s->j[1]/n2;
    jb[0]=s->j[0]/n1;

        
    
    int cache_x=FD_ACACHE_X;


    s_gpu.n[0]=s->n[0];    s_gpu.n[1]=s->n[1];    s_gpu.n[2]=s->n[2];
    s_gpu.j[0]=s->j[0];    s_gpu.j[1]=s->j[1];    s_gpu.j[2]=s->j[2];
    
    memset(coefs0,0,sizeof(double)*(FD_MAXJ+1));
    memset(offsets0,0,sizeof(int)*(FD_MAXJ+1));
    /*
    fprintf(stdout,"%ld\t", s->ncoefs);
    for(int i = 0; i < s->ncoefs; ++i)
      fprintf(stdout,"(%lf %ld)\t", s->coefs[i], s->offsets[i]);
    fprintf(stdout,"\n%ld %ld %ld %ld %ld %ld\n",s->j[0],s->j[1],s->j[2],s->n[0],s->n[1],s->n[2]);
    */
    
    
    for(int i = 0; i < s->ncoefs; i++){
      int i0=s->offsets[i]/n1;
      int i1=(s->offsets[i]-i0*n1)/n2;
      int i2=(s->offsets[i]-i0*n1-i1*n2);
      if (i1==0 && i2==0 && abs(i0)<=jb[0]){
	int offset=FD_MAXJ/2+i0;
	offsets0[offset]=offset;
	coefs0[offset]=s->coefs[i];
	ncoefs0=MAX(ncoefs0,2*abs(i0)+1);	
      } else if (i0==0 && abs(i1)<=jb[1] && abs(i2)<=jb[2]){
	offsets12[ncoefs12]=i2+cache_x*i1;
	coefs12[ncoefs12]=s->coefs[i];
	ncoefs12++;	
      } else{
	offsets[ncoefs]=s->offsets[i];
	coefs[ncoefs]=s->coefs[i];
	ncoefs++;
      }
    }
    for(int i = 0; i < ncoefs0; i++){
      offsets0[i]=i;
      coefs0[i]=coefs0[i+(FD_MAXJ-ncoefs0+1)/2];
    }

    s_gpu.ncoefs=ncoefs;
    s_gpu.ncoefs12=ncoefs12;
    s_gpu.ncoefs0=ncoefs0;

    if (ncoefs>0){
      gpaw_cudaSafeCall(cudaMalloc(&(s_gpu.coefs_gpu),sizeof(double)*ncoefs));
      gpaw_cudaSafeCall(cudaMemcpy(s_gpu.coefs_gpu,coefs,sizeof(double)*ncoefs,
				   cudaMemcpyHostToDevice));
      
      gpaw_cudaSafeCall(cudaMalloc(&(s_gpu.offsets_gpu),sizeof(long)*ncoefs));
      gpaw_cudaSafeCall(cudaMemcpy(s_gpu.offsets_gpu,offsets,sizeof(long)*ncoefs,
				   cudaMemcpyHostToDevice));
    }
    gpaw_cudaSafeCall(cudaMalloc(&(s_gpu.coefs12_gpu),sizeof(double)*ncoefs12));
    gpaw_cudaSafeCall(cudaMemcpy(s_gpu.coefs12_gpu,coefs12,
				 sizeof(double)*ncoefs12,
				 cudaMemcpyHostToDevice));

    gpaw_cudaSafeCall(cudaMalloc(&(s_gpu.offsets12_gpu),sizeof(int)*ncoefs12));
    gpaw_cudaSafeCall(cudaMemcpy(s_gpu.offsets12_gpu,offsets12,
				 sizeof(int)*ncoefs12,cudaMemcpyHostToDevice));

    gpaw_cudaSafeCall(cudaMalloc(&(s_gpu.coefs0_gpu),sizeof(double)*ncoefs0));
    gpaw_cudaSafeCall(cudaMemcpy(s_gpu.coefs0_gpu,coefs0,
				 sizeof(double)*ncoefs0,
				 cudaMemcpyHostToDevice));

    gpaw_cudaSafeCall(cudaMalloc(&(s_gpu.offsets0_gpu),sizeof(int)*ncoefs0));
    gpaw_cudaSafeCall(cudaMemcpy(s_gpu.offsets0_gpu,offsets0,
				 sizeof(int)*ncoefs0,cudaMemcpyHostToDevice));

    return s_gpu;
  }

}


extern "C" {
  double bmgs_fd_cuda_cpu(const bmgsstencil* s, const double* a, double* b)
  {
  
    double *adev,*bdev;

    size_t asize,bsize;
    struct timeval  t0, t1; 
    double flops;
    bmgsstencil_gpu s_gpu=bmgs_stencil_to_gpu(s);


    asize=s->j[0]+s->n[0]*(s->j[1]+s->n[1]*(s->n[2]+s->j[2]));
    bsize=s->n[0]*s->n[1]*s->n[2];

    gpaw_cudaSafeCall(cudaGetLastError());
    gpaw_cudaSafeCall(cudaMalloc(&adev,sizeof(double)*asize));

    gpaw_cudaSafeCall(cudaMalloc(&bdev,sizeof(double)*bsize));

   
    gpaw_cudaSafeCall(cudaMemcpy(adev,a,sizeof(double)*asize,
				 cudaMemcpyHostToDevice));
    gpaw_cudaSafeCall(cudaGetLastError());
    gettimeofday(&t0,NULL);  
    bmgs_fd_cuda_gpu(&s_gpu, adev,bdev,1);
      
    
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

  double bmgs_fd_cuda_cpu_bc(const bmgsstencil* s, const double* a, double* b)
  {
  
    double *adev,*bdev;

    size_t asize,bsize;
    struct timeval  t0, t1; 
    double flops;
    bmgsstencil_gpu s_gpu=bmgs_stencil_to_gpu(s);


    bsize=s->n[0]*s->n[1]*s->n[2];

    gpaw_cudaSafeCall(cudaGetLastError());
    gpaw_cudaSafeCall(cudaMalloc(&adev,sizeof(double)*bsize));

    gpaw_cudaSafeCall(cudaMalloc(&bdev,sizeof(double)*bsize));

   
    gpaw_cudaSafeCall(cudaMemcpy(adev,a,sizeof(double)*bsize,
				 cudaMemcpyHostToDevice));
    gpaw_cudaSafeCall(cudaGetLastError());
    gettimeofday(&t0,NULL);  
    bmgs_fd_cuda_gpu_bc(&s_gpu, adev,bdev,1);
      
    
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
#endif
