// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

#include <sys/types.h>
#include <sys/time.h>

#include <cuComplex.h>

#include "gpaw-cuda-int.h"

#ifndef MYJ
#ifndef CUGPAWCOMPLEX
/*
__constant__ long c_offsets[FD_MAXCOEFS];
__constant__ double c_coefs[FD_MAXCOEFS];
__constant__ int c_offsets12[FD_MAXCOEFS];
__constant__ double c_coefs12[FD_MAXCOEFS];
__constant__ double c_coefs0[FD_MAXJ+1];
*/
#endif
#endif

#undef  ACACHE_X
#undef BLOCK_X
#undef BLOCK_Y_B
#undef BLOCK_X_B
#ifndef CUGPAWCOMPLEX
#define ACACHE_X  FD_ACACHE_X
#define BLOCK_X  FD_BLOCK_X
#define BLOCK_X_B   FD_BLOCK_X_B
#define BLOCK_Y_B   FD_BLOCK_Y_B

//XOPT 16 8
//YOPT 16 4
//ZOPT 8  16
#else
#define ACACHE_X  FD_ACACHE_Xz
#define BLOCK_X  FD_BLOCK_Xz
#define BLOCK_X_B  FD_BLOCK_X_Bz
#define BLOCK_Y_B   FD_BLOCK_Y_B
#endif

#ifdef MYJ
#undef  FD_ACACHE_Y
#define FD_ACACHE_Y  ((FD_BLOCK_Y)+(MYJ))

#undef MYJ_X
#ifndef CUGPAWCOMPLEX
#define MYJ_X MYJ
#else
#define MYJ_X (2*(MYJ))
#endif



__global__ void FD_kernel(const int ncoefs,const double *c_coefs,
			  const long *c_offsets,
			  const int ncoefs12,const double *c_coefs12,
			  const int *c_offsets12,const double *c_coefs0,
			  const double* a,
			  double* b,const long3 c_n,
			  const int3 c_jb,const int3 c_bjb,const int blocks)
{
  
  int xx=gridDim.x/FD_XDIV;
  int yy=gridDim.y/blocks;

  int xind=blockIdx.x/xx;
  int i2bl=blockIdx.x-xind*xx;


  int i2tid=threadIdx.x;
  int i2=i2bl*BLOCK_X+i2tid;

  int blocksi=blockIdx.y/yy;
  int i1bl=blockIdx.y-blocksi*yy;

  int i1tid=threadIdx.y;
  int i1=i1bl*FD_BLOCK_Y+i1tid;

  __shared__ double acache12[FD_ACACHE_Y*ACACHE_X];

  double acache0[MYJ+1];
  double *acache12p;
  int sizez=c_jb.z+c_n.z;  
  int sizeyz=(c_jb.y+c_n.y)*sizez;
  int sizebz=c_bjb.z+c_n.z;  
  int sizebyz=(c_bjb.y+c_n.y)*sizebz;
  
  int xlen=(c_n.x+FD_XDIV-1)/FD_XDIV;
  int xstart=xind*xlen;
  int xend=MIN(xstart+xlen,c_n.x);
  
  a+=(c_jb.x+c_n.x)*sizeyz*blocksi;
  b+=(c_n.x+c_bjb.x)*sizebyz*blocksi;
  
  a+=xstart*sizeyz+i1*sizez+i2;
  b+=xstart*sizebyz+i1*sizebz+i2;
  
  acache12p=acache12+ACACHE_X*(i1tid+MYJ/2)+i2tid+MYJ_X/2;
  for (int c=1;c<MYJ+1;c++){
    acache0[c]=a[(c-1-MYJ/2)*(sizeyz)];
  }
  for (int i0=xstart; i0 < xend; i0++) {  
    for (int c=0;c<MYJ;c++){
      acache0[c]=acache0[c+1];
    }   
    if ((i1<c_n.y+MYJ/2) && (i2<c_n.z+MYJ_X/2))
      acache0[MYJ]=a[(MYJ/2)*sizeyz];

    acache12p[0]=acache0[MYJ/2];
    if  (i2tid<MYJ_X/2){
      acache12p[-MYJ_X/2]=a[-MYJ_X/2];
      acache12p[BLOCK_X]=a[BLOCK_X];
    }
    if  (i1tid<MYJ/2){
      acache12p[-ACACHE_X*MYJ/2]=a[-sizez*MYJ/2];
      acache12p[ACACHE_X*FD_BLOCK_Y]=a[sizez*FD_BLOCK_Y];      
    }
    __syncthreads();         
    double x = 0.0;
#include"fd-cuda-pragmas.cu"
    for (int c = 0; c < ncoefs12; c++){
      x+=acache12p[c_offsets12[c]]*c_coefs12[c];
    }	
    for (int c = 0; c < MYJ+1; c++){
      x+=acache0[c]*c_coefs0[c];
    }	    
    for (int c = 0; c < ncoefs; c++){	  
      x+=a[c_offsets[c]]*c_coefs[c];
    }	

    if ((i1<c_n.y) && (i2<c_n.z)) {
      b[0] = x;
    }

    b+=sizebyz;
    a+=sizeyz;
    __syncthreads();         
  }
  
}


__global__ void FD_kernel_onlyb(const int ncoefs,const double *c_coefs,
				const long *c_offsets,
				const int ncoefs12,const double *c_coefs12,
				const int *c_offsets12,const double *c_coefs0,
				const double* a,
				double* b,const long3 c_n,
				const int3 c_jb,const int boundary,
				const int blocks)
{

  int xx=MAX((c_n.z+BLOCK_X_B-1)/BLOCK_X_B,1);
  int yy=MAX((c_n.y+BLOCK_Y_B-1)/BLOCK_Y_B,1);
  int ysiz=c_n.y;
  if ((boundary & GPAW_BOUNDARY_Y0) != 0) 
    ysiz-=BLOCK_Y_B;
  //ysiz-=c_jb.y/2;
  if ((boundary & GPAW_BOUNDARY_Y1) != 0) 
    ysiz-=BLOCK_Y_B;
  //ysiz-=c_jb.y/2;
  int yy2=MAX((ysiz+BLOCK_Y_B-1)/BLOCK_Y_B,0);

  int i2bl,i1bl;
  int xlen=c_n.x;
  int xind=0;
  int xstart=0;
  int i2pitch=0,i1pitch=0;
  int ymax=c_n.y,zmax=c_n.z,xmax=c_n.x;
  int xend,blockix;

  blockix=blockIdx.x;

  if ((boundary & GPAW_BOUNDARY_X0) != 0) {
    if ((blockix>=0) && (blockix<xx*yy)) {
      i1bl=blockix/xx;
      i2bl=blockix-i1bl*xx;
      xlen=c_jb.x/2;
      xstart=0;
    }
    blockix-=xx*yy;
  }
  if ((boundary & GPAW_BOUNDARY_X1) != 0) {
    if ((blockix>=0) && (blockix<xx*yy)) {
      i1bl=blockix/xx;
      i2bl=blockix-i1bl*xx;
      xlen=c_jb.x/2;
      xstart+=c_n.x-c_jb.x/2;
    }
    blockix-=xx*yy;    
  }
  if (blockix>=0){
    if ((boundary & GPAW_BOUNDARY_Y0) != 0) {
      if ((blockix>=0) && (blockix<FD_XDIV_B*xx)) {
	xind=blockix/xx;
	i2bl=blockix-xind*xx;
	i1bl=0;
	ymax=MIN(BLOCK_Y_B,ymax);
	//ymax=MIN(c_jb.y/2,ymax);
      }
      blockix-=FD_XDIV_B*xx;
    }
    if ((boundary & GPAW_BOUNDARY_Y1) != 0) {
      if ((blockix>=0) && (blockix<FD_XDIV_B*xx)) {
	xind=blockix/xx;
	i2bl=blockix-xind*xx;
	i1bl=0;
	//i1pitch=MAX(c_n.y-c_jb.y/2,0);
	i1pitch=MAX(c_n.y-BLOCK_Y_B,0);
      }
      blockix-=FD_XDIV_B*xx;
    }
    if ((boundary & GPAW_BOUNDARY_Z0) != 0) {
      if ((blockix>=0) && (blockix<FD_XDIV_B*yy2)) {
	xind=blockix/yy2;
	i2bl=0;
	zmax=MIN(BLOCK_X_B,zmax);
	//zmax=MIN(c_jb.z/2,zmax);
	i1bl=blockix-xind*yy2;
	if ((boundary & GPAW_BOUNDARY_Y0) != 0) 
	  i1pitch=BLOCK_Y_B;
	//i1pitch=c_jb.y/2;
	if ((boundary & GPAW_BOUNDARY_Y1) != 0) 
	  ymax=MAX(c_n.y-BLOCK_Y_B,0);
	//ymax=MAX(c_n.y-c_jb.y/2,0);	
      }
      blockix-=FD_XDIV_B*yy2;
    }
    if ((boundary & GPAW_BOUNDARY_Z1) != 0) {
      if ((blockix>=0) && (blockix<FD_XDIV_B*yy2)) {
	xind=blockix/yy2;
	i2bl=0;
	//i2pitch=MAX(c_n.z-c_jb.z/2,0);
	i2pitch=MAX(c_n.z-BLOCK_X_B,0);
	i1bl=blockix-xind*yy2;
	if ((boundary & GPAW_BOUNDARY_Y0) != 0) 
	  i1pitch=BLOCK_Y_B;
	//i1pitch=c_jb.y/2;
	if ((boundary & GPAW_BOUNDARY_Y1) != 0) 
	  ymax=MAX(c_n.y-BLOCK_Y_B,0);
	//ymax=MAX(c_n.y-c_jb.y/2,0);
      }
      blockix-=FD_XDIV_B*yy2;
    }
    if ((boundary & GPAW_BOUNDARY_X0) != 0) {
      xstart+=c_jb.x/2; 	
      xlen-=c_jb.x/2;
    }
    if ((boundary & GPAW_BOUNDARY_X1) != 0) {
      xlen-=c_jb.x/2;
      xmax-=c_jb.x/2;
    }
    xlen=(xlen+FD_XDIV_B-1)/FD_XDIV_B;
    xstart+=xind*xlen;        
  }
  xend=MIN(xstart+xlen,xmax);    
  if (blockix>=0){
    printf("Error!!\n");
    return;
  }
  

  int i2tid=threadIdx.x;
  int i2=i2pitch+i2bl*BLOCK_X_B+i2tid;

  int blocksi=blockIdx.y;

  int i1tid=threadIdx.y;
  int i1=i1pitch+i1bl*BLOCK_Y_B+i1tid;

  __shared__ double acache12[FD_ACACHE_Y*ACACHE_X];

  double acache0[MYJ+1];
  double *acache12p;
  int sizez=c_jb.z+c_n.z;  
  int sizeyz=(c_jb.y+c_n.y)*sizez;

  a+=((c_jb.x+c_n.x)*sizeyz)*blocksi;
  b+=(c_n.x*c_n.y*c_n.z)*blocksi;
  

  acache12p=acache12+ACACHE_X*(i1tid+MYJ/2)+i2tid+MYJ_X/2;

  a+=xstart*sizeyz+i1*sizez+i2;
  b+=xstart*c_n.y*c_n.z+i1*c_n.z+i2;
  for (int c=1;c<MYJ+1;c++){
    acache0[c]=a[(c-1-MYJ/2)*(sizeyz)];
  }
  
  for (int i0=xstart; i0 < xend; i0++) {  
    for (int c=0;c<MYJ;c++){
      acache0[c]=acache0[c+1];
    }   
    if ((i1<c_n.y+MYJ/2) && (i2<c_n.z+MYJ_X/2))
      acache0[MYJ]=a[(MYJ/2)*sizeyz];

    acache12p[0]=acache0[MYJ/2];
    if  (i2tid<MYJ_X/2){
      acache12p[-MYJ_X/2]=a[-MYJ_X/2];
      acache12p[BLOCK_X_B]=a[BLOCK_X_B];
    }
    if  (i1tid<MYJ/2){
      acache12p[-ACACHE_X*MYJ/2]=a[-sizez*MYJ/2];
      acache12p[ACACHE_X*BLOCK_Y_B]=a[sizez*BLOCK_Y_B];      
    }
    __syncthreads();         
    double x = 0.0;
#include"fd-cuda-pragmas.cu"
    for (int c = 0; c < ncoefs12; c++){
      x+=acache12p[c_offsets12[c]]*c_coefs12[c];
    }	
    for (int c = 0; c < MYJ+1; c++){
      x+=acache0[c]*c_coefs0[c];
    }	    
    for (int c = 0; c < ncoefs; c++){	  
      x+=a[c_offsets[c]]*c_coefs[c];
    }	            
    if  ((i1<ymax) && (i2<zmax)) {
      
      b[0] = x;
    }
    
    b+=c_n.y*c_n.z;
    a+=sizeyz;
    __syncthreads();         
  }
  
}




/*


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
*/


#else
#define MYJ  2
#  define FD_kernel Zcuda(fd_kernel2)
#  define FD_kernel_onlyb Zcuda(fd_kernel2_onlyb)
#  include "fd-cuda.cu"
#  undef FD_kernel
#  undef FD_kernel_onlyb
#  undef MYJ
#define MYJ  4
#  define FD_kernel Zcuda(fd_kernel4)
#  define FD_kernel_onlyb Zcuda(fd_kernel4_onlyb)
#  include "fd-cuda.cu"
#  undef FD_kernel
#  undef FD_kernel_onlyb
#  undef MYJ
#define MYJ  6
#  define FD_kernel Zcuda(fd_kernel6)
#  define FD_kernel_onlyb Zcuda(fd_kernel6_onlyb)
#  include "fd-cuda.cu"
#  undef FD_kernel
#  undef FD_kernel_onlyb
#  undef MYJ
#define MYJ  8
#  define FD_kernel Zcuda(fd_kernel8)
#  define FD_kernel_onlyb Zcuda(fd_kernel8_onlyb)
#  include "fd-cuda.cu"
#  undef FD_kernel
#  undef FD_kernel_onlyb
#  undef MYJ
#define MYJ  10
#  define FD_kernel Zcuda(fd_kernel10)
#  define FD_kernel_onlyb Zcuda(fd_kernel10_onlyb)
#  include "fd-cuda.cu"
#  undef FD_kernel
#  undef FD_kernel_onlyb
#  undef MYJ


extern "C" {



  bmgsstencil_gpu bmgs_stencil_to_gpu(const bmgsstencil* s);


  void Zcuda(bmgs_fd_cuda_gpu)(const bmgsstencil_gpu* s_gpu, 
			       const Tcuda* adev, Tcuda* bdev,
			       int boundary,int blocks,
			       cudaStream_t stream)  
  {
    int3 jb;
    int3 bjb;
    int3 hc_bj;
    

    long3 hc_n;
    long3 hc_j; 

    long* offsets_gpu;
    int* offsets12_gpu;




    hc_n.x=s_gpu->n[0];    hc_n.y=s_gpu->n[1];    hc_n.z=s_gpu->n[2];
    hc_j.x=s_gpu->j[0];    hc_j.y=s_gpu->j[1];    hc_j.z=s_gpu->j[2];


    bjb.x=0;    bjb.y=0;    bjb.z=0;
    hc_bj.x=0;    hc_bj.y=0;    hc_bj.y=0;

	
#ifdef CUGPAWCOMPLEX
    hc_n.z*=2;
    hc_j.x*=2;    hc_j.y*=2;    hc_j.z*=2;
    offsets_gpu=s_gpu->offsets_gpuz;
    offsets12_gpu=s_gpu->offsets12_gpuz;
#else
    offsets_gpu=s_gpu->offsets_gpu;
    offsets12_gpu=s_gpu->offsets12_gpu;
#endif

    jb.z=hc_j.z;
    jb.y=hc_j.y/(hc_j.z+hc_n.z);
    jb.x=hc_j.x/((hc_j.z+hc_n.z)*hc_n.y+hc_j.y);
    if ((boundary & GPAW_BOUNDARY_SKIP) != 0) {
      int3 jb1;
      int3 bjb1,bjb2;
      bjb1.x=0;    bjb1.y=0;    bjb1.z=0;
      bjb2.x=0;    bjb2.y=0;    bjb2.z=0;      
      jb1.z=jb.z/2;
      jb1.x=jb.x/2;
      jb1.y=jb.y/2;
      if ((boundary & GPAW_BOUNDARY_X0) != 0) {
	bjb1.x+=jb.x/2;
      }
      if ((boundary & GPAW_BOUNDARY_X1) != 0) {
	bjb2.x+=jb.x/2;
      }
      if ((boundary & GPAW_BOUNDARY_Y0) != 0) {
	bjb1.y+=BLOCK_Y_B;
	//bjb1.y+=jb.y/2;
      }
      if ((boundary & GPAW_BOUNDARY_Y1) != 0) {
	bjb2.y+=BLOCK_Y_B;
	//bjb2.y+=jb.y/2;
      }
      if ((boundary & GPAW_BOUNDARY_Z0) != 0) {
	bjb1.z+=BLOCK_X_B;
	//bjb1.z+=jb.z/2;
      }
      if ((boundary & GPAW_BOUNDARY_Z1) != 0) {
	bjb2.z+=BLOCK_X_B;
	//bjb2.z+=jb.z/2;
      }
      bjb.x=bjb1.x+bjb2.x;
      bjb.y=bjb1.y+bjb2.y;
      bjb.z=bjb1.z+bjb2.z;

      hc_n.x-=bjb.x;    hc_n.y-=bjb.y;    hc_n.z-=bjb.z;

      jb.x+= bjb.x;    jb.y+=bjb.y;    jb.z+=bjb.z;
      jb1.x+= bjb1.x;    jb1.y+=bjb1.y;    jb1.z+=bjb1.z;
      
      hc_bj.z=bjb.z;
      hc_bj.y=bjb.y*(hc_bj.z+hc_n.z);
      hc_bj.x=bjb.x*((hc_bj.z+hc_n.z)*hc_n.y+hc_bj.y);
      
      hc_j.z=jb.z;
      hc_j.y=jb.y*(hc_j.z+hc_n.z);
      hc_j.x=jb.x*((hc_j.z+hc_n.z)*hc_n.y+hc_j.y);
      
      bdev+=bjb1.z+bjb1.y*(hc_bj.z+hc_n.z)+
	bjb1.x*((hc_bj.z+hc_n.z)*hc_n.y+hc_bj.y);
      
      adev=(Tcuda*)((double*)adev+jb1.z+jb1.y*(hc_j.z+hc_n.z)+
		    jb1.x*((hc_j.z+hc_n.z)*hc_n.y+hc_j.y));
    }else {
      adev=(Tcuda*)((double*)adev +(hc_j.x+hc_j.y+hc_j.z)/2);
    }
    
    /*
    if (s_gpu->ncoefs>0){
      gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_offsets,offsets_gpu,
					   sizeof(long)*s_gpu->ncoefs,0,
					   cudaMemcpyDeviceToDevice));
      gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_coefs,s_gpu->coefs_gpu,
					   sizeof(double)*s_gpu->ncoefs,0,
					   cudaMemcpyDeviceToDevice,
					   stream));
    }
    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_offsets12,offsets12_gpu,
					 sizeof(int)*s_gpu->ncoefs12,0,
					 cudaMemcpyDeviceToDevice,
					 stream));
    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_coefs12,s_gpu->coefs12_gpu,
					 sizeof(double)*s_gpu->ncoefs12,0,
					 cudaMemcpyDeviceToDevice,
					 stream));
    */
    /*gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_offsets0,s_gpu->offsets0_gpu,
					 sizeof(int)*s_gpu->ncoefs0,0,
					 cudaMemcpyDeviceToDevice));*/
    /* gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_coefs0,s_gpu->coefs0_gpu,
					 sizeof(double)*s_gpu->ncoefs0,0,
					 cudaMemcpyDeviceToDevice,
    				 stream));
    */
    if ((hc_n.x<=0) || (hc_n.y<=0) || (hc_n.z<=0))
      return;

    dim3 dimBlock(1,1,1);
    dim3 dimGrid(1,1,1);
    if (((boundary & GPAW_BOUNDARY_NORMAL) != 0) ||
	((boundary & GPAW_BOUNDARY_SKIP) != 0)){
      dimGrid.x=FD_XDIV*MAX((hc_n.z+BLOCK_X-1)/BLOCK_X,1);
      dimGrid.y=blocks*MAX((hc_n.y+FD_BLOCK_Y-1)/FD_BLOCK_Y,1);
      dimBlock.x=BLOCK_X;
      dimBlock.y=FD_BLOCK_Y;
    } else if ((boundary & GPAW_BOUNDARY_ONLY) != 0) {
      int xx=MAX((hc_n.z+BLOCK_X_B-1)/BLOCK_X_B,1);
      int yy=MAX((hc_n.y+BLOCK_Y_B-1)/BLOCK_Y_B,1);
      int ysiz=hc_n.y;
      if ((boundary & GPAW_BOUNDARY_Y0) != 0) 
	ysiz-=BLOCK_Y_B;
      //ysiz-=jb.y/2;
      if ((boundary & GPAW_BOUNDARY_Y1) != 0) 
	ysiz-=BLOCK_Y_B;
      //ysiz-=jb.y/2;
      int yy2=MAX((ysiz+BLOCK_Y_B-1)/BLOCK_Y_B,0);
      dimGrid.x=0;
      if ((boundary & GPAW_BOUNDARY_X0) != 0) 
	dimGrid.x+=xx*yy;
      if ((boundary & GPAW_BOUNDARY_X1) != 0) 
	dimGrid.x+=xx*yy;
      if ((boundary & GPAW_BOUNDARY_Y0) != 0) 
	dimGrid.x+=FD_XDIV_B*xx;
      if ((boundary & GPAW_BOUNDARY_Y1) != 0) 
	dimGrid.x+=FD_XDIV_B*xx;
      if ((boundary & GPAW_BOUNDARY_Z0) != 0) 
	dimGrid.x+=FD_XDIV_B*yy2;
      if ((boundary & GPAW_BOUNDARY_Z1) != 0) 
	dimGrid.x+=FD_XDIV_B*yy2;
      dimGrid.y=blocks;
      dimBlock.x=BLOCK_X_B;
      dimBlock.y=BLOCK_Y_B;
    }

    
    if (((boundary & GPAW_BOUNDARY_NORMAL) != 0) ||
	((boundary & GPAW_BOUNDARY_SKIP) != 0)){
      if (s_gpu->ncoefs0<=3)
	Zcuda(fd_kernel2)<<<dimGrid, dimBlock, 0, stream>>>
	  (s_gpu->ncoefs,s_gpu->coefs_gpu,offsets_gpu,
	   s_gpu->ncoefs12,s_gpu->coefs12_gpu,offsets12_gpu,
	   s_gpu->coefs0_gpu,
	   (double*)adev,(double*)bdev, 
	   hc_n,jb,bjb,blocks);    
      else if (s_gpu->ncoefs0<=5)
	Zcuda(fd_kernel4)<<<dimGrid, dimBlock, 0, stream>>>
	  (s_gpu->ncoefs,s_gpu->coefs_gpu,offsets_gpu,
	   s_gpu->ncoefs12,s_gpu->coefs12_gpu,offsets12_gpu,
	   s_gpu->coefs0_gpu,
	   (double*)adev,(double*)bdev, 
	   hc_n,jb,bjb,blocks);    
      else if (s_gpu->ncoefs0<=7)
	Zcuda(fd_kernel6)<<<dimGrid, dimBlock, 0, stream>>>
	  (s_gpu->ncoefs,s_gpu->coefs_gpu,offsets_gpu,
	   s_gpu->ncoefs12,s_gpu->coefs12_gpu,offsets12_gpu,
	   s_gpu->coefs0_gpu,
	   (double*)adev,(double*)bdev, 
	   hc_n,jb,bjb,blocks);    
      else if (s_gpu->ncoefs0<=9)
	Zcuda(fd_kernel8)<<<dimGrid, dimBlock, 0, stream>>>
	  (s_gpu->ncoefs,s_gpu->coefs_gpu,offsets_gpu,
	   s_gpu->ncoefs12,s_gpu->coefs12_gpu,offsets12_gpu,
	   s_gpu->coefs0_gpu,
	   (double*)adev,(double*)bdev, 
	   hc_n,jb,bjb,blocks);    
      else if (s_gpu->ncoefs0<=11)
	Zcuda(fd_kernel10)<<<dimGrid, dimBlock, 0, stream>>>
	  (s_gpu->ncoefs,s_gpu->coefs_gpu,offsets_gpu,
	   s_gpu->ncoefs12,s_gpu->coefs12_gpu,offsets12_gpu,
	   s_gpu->coefs0_gpu,
	   (double*)adev,(double*)bdev, 
	   hc_n,jb,bjb,blocks); 
    } else if ((boundary & GPAW_BOUNDARY_ONLY) != 0) {
      if (s_gpu->ncoefs0<=3)
	Zcuda(fd_kernel2_onlyb)<<<dimGrid, dimBlock, 0, stream>>>
	  (s_gpu->ncoefs,s_gpu->coefs_gpu,offsets_gpu,
	   s_gpu->ncoefs12,s_gpu->coefs12_gpu,offsets12_gpu,
	   s_gpu->coefs0_gpu,
	   (double*)adev,(double*)bdev, 
	   hc_n,jb,boundary,blocks);    
      else if (s_gpu->ncoefs0<=5)
	Zcuda(fd_kernel4_onlyb)<<<dimGrid, dimBlock, 0, stream>>>
	  (s_gpu->ncoefs,s_gpu->coefs_gpu,offsets_gpu,
	   s_gpu->ncoefs12,s_gpu->coefs12_gpu,offsets12_gpu,
	   s_gpu->coefs0_gpu,
	   (double*)adev,(double*)bdev, 
	   hc_n,jb,boundary,blocks);    
      else if (s_gpu->ncoefs0<=7)
	Zcuda(fd_kernel6_onlyb)<<<dimGrid, dimBlock, 0, stream>>>
	  (s_gpu->ncoefs,s_gpu->coefs_gpu,offsets_gpu,
	   s_gpu->ncoefs12,s_gpu->coefs12_gpu,offsets12_gpu,
	   s_gpu->coefs0_gpu,
	   (double*)adev,(double*)bdev, 
	   hc_n,jb,boundary,blocks);    
      else if (s_gpu->ncoefs0<=9)
	Zcuda(fd_kernel8_onlyb)<<<dimGrid, dimBlock, 0, stream>>>
	  (s_gpu->ncoefs,s_gpu->coefs_gpu,offsets_gpu,
	   s_gpu->ncoefs12,s_gpu->coefs12_gpu,offsets12_gpu,
	   s_gpu->coefs0_gpu,
	   (double*)adev,(double*)bdev, 
	   hc_n,jb,boundary,blocks);    
      else if (s_gpu->ncoefs0<=11)
	Zcuda(fd_kernel10_onlyb)<<<dimGrid, dimBlock, 0, stream>>>
	  (s_gpu->ncoefs,s_gpu->coefs_gpu,offsets_gpu,
	   s_gpu->ncoefs12,s_gpu->coefs12_gpu,offsets12_gpu,
	   s_gpu->coefs0_gpu,
	   (double*)adev,(double*)bdev, 
	   hc_n,jb,boundary,blocks); 
    }

    gpaw_cudaSafeCall(cudaGetLastError());    
  }
  



  double Zcuda(bmgs_fd_cuda_cpu)(const bmgsstencil* s, const Tcuda* a, 
				 Tcuda* b,int boundary)
  {
  
    Tcuda *adev,*bdev;

    size_t asize,bsize;
    struct timeval  t0, t1; 
    double flops;
    bmgsstencil_gpu s_gpu=bmgs_stencil_to_gpu(s);


    asize=s->j[0]+s->n[0]*(s->j[1]+s->n[1]*(s->n[2]+s->j[2]));
    bsize=s->n[0]*s->n[1]*s->n[2];

    gpaw_cudaSafeCall(cudaGetLastError());
    gpaw_cudaSafeCall(cudaMalloc(&adev,sizeof(Tcuda)*asize));

    gpaw_cudaSafeCall(cudaMalloc(&bdev,sizeof(Tcuda)*bsize));

   
    gpaw_cudaSafeCall(cudaMemcpy(adev,a,sizeof(Tcuda)*asize,
				 cudaMemcpyHostToDevice));
    gpaw_cudaSafeCall(cudaGetLastError());
    gettimeofday(&t0,NULL);  
    Zcuda(bmgs_fd_cuda_gpu)(&s_gpu, adev,bdev,boundary,1,0);
      
    
    cudaThreadSynchronize(); 
    gpaw_cudaSafeCall(cudaGetLastError());

    gettimeofday(&t1,NULL);
    gpaw_cudaSafeCall(cudaMemcpy(b,bdev,sizeof(Tcuda)*bsize,
				 cudaMemcpyDeviceToHost));
    
    gpaw_cudaSafeCall(cudaFree(adev));
    gpaw_cudaSafeCall(cudaFree(bdev));

    flops=(t1.tv_sec*1.0+t1.tv_usec/1000000.0-t0.tv_sec*1.0-t0.tv_usec/1000000.0); 

    return flops;
  
  
  }
}

#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "fd-cuda.cu"

extern "C" {

  int bmgs_fd_boundary_test(const bmgsstencil_gpu* s)
  {
    int3 jb;

    long3 hc_n;
    long3 hc_j; 

    hc_n.x=s->n[0];    hc_n.y=s->n[1];    hc_n.z=s->n[2];
    hc_j.x=s->j[0];    hc_j.y=s->j[1];    hc_j.z=s->j[2];
    jb.z=hc_j.z;
    jb.y=hc_j.y/(hc_j.z+hc_n.z);
    jb.x=hc_j.x/((hc_j.z+hc_n.z)*hc_n.y+hc_j.y);
    if (hc_n.x<(jb.x+1) || hc_n.y<(3*FD_BLOCK_Y_B) || 
	hc_n.z<(3*FD_BLOCK_X_B))
      return 0;

    return 1;        
  }

  bmgsstencil_gpu bmgs_stencil_to_gpu(const bmgsstencil* s)  
  {
    bmgsstencil_gpu s_gpu;/*
={s->ncoefs,NULL,NULL,0,NULL,NULL,0,NULL,NULL,
			   {s->n[0],s->n[1],s->n[2]},
			   {s->j[0],s->j[1],s->j[2]}};
			  */
    long offsets[s->ncoefs];
    int  offsets12[s->ncoefs];
    long offsetsz[s->ncoefs];
    int  offsets12z[s->ncoefs];
    int  offsets0[FD_MAXJ+1];
    double coefs[s->ncoefs],coefs12[s->ncoefs],coefs0[FD_MAXJ+1];
    long ncoefs=0,ncoefs12=0,ncoefs0=0;

    int n2=(s->n[2]+s->j[2]);
    int n1=s->j[1]+s->n[1]*n2;

    int jb[3];
    
    jb[2]=s->j[2];
    jb[1]=s->j[1]/n2;
    jb[0]=s->j[0]/n1;

        
    s_gpu.n[0]=s->n[0];    s_gpu.n[1]=s->n[1];    s_gpu.n[2]=s->n[2];
    s_gpu.j[0]=s->j[0];    s_gpu.j[1]=s->j[1];    s_gpu.j[2]=s->j[2];
    
    memset(coefs0,0,sizeof(double)*(FD_MAXJ+1));
    memset(offsets0,0,sizeof(int)*(FD_MAXJ+1));
    
    
    for(int i = 0; i < s->ncoefs; i++){
      int offpoint=s->offsets[i]+(s->j[0]+s->j[1]+s->j[2])/2;
      int i0=offpoint/n1;
      int i1=(offpoint-i0*n1)/n2;
      int i2=(offpoint-i0*n1-i1*n2);
      i0-=jb[0]/2;
      i1-=jb[1]/2;
      i2-=jb[2]/2;
      if (i1==0 && i2==0 && abs(i0)<=jb[0]/2){
	int offset=FD_MAXJ/2+i0;
	if (fabs(s->coefs[i]) > DBL_EPSILON){
	  offsets0[offset]=offset;
	  coefs0[offset]=s->coefs[i];
	  ncoefs0=MAX(ncoefs0,2*abs(i0)+1);	
	}
      } else if (i0==0 && abs(i1)<=jb[1]/2 && abs(i2)<=jb[2]/2 && (i1==0 || i2==0)){
	offsets12[ncoefs12]=i2+FD_ACACHE_X*i1;
	offsets12z[ncoefs12]=2*i2+FD_ACACHE_Xz*i1;
      
	coefs12[ncoefs12]=s->coefs[i];
	ncoefs12++;	
      } else{
	offsets[ncoefs]=s->offsets[i];
	offsetsz[ncoefs]=2*s->offsets[i];
	coefs[ncoefs]=s->coefs[i];
	ncoefs++;
      }
    }
    ncoefs0=jb[0]+1;
    for(int i = 0; i < ncoefs0; i++){
      offsets0[i]=i;
      coefs0[i]=coefs0[i+(FD_MAXJ-ncoefs0+1)/2];
    }
    /*
    fprintf(stdout,"ncoefs %d\t", ncoefs);
    for(int i = 0; i < ncoefs; ++i)
      fprintf(stdout,"(%lf %d)\t", coefs[i], offsets[i]);
    fprintf(stdout,"\n");
    fprintf(stdout,"ncoefs0 %d\t", ncoefs0);
    for(int i = 0; i < ncoefs0; ++i)
      fprintf(stdout,"(%lf %d)\t", coefs0[i], offsets0[i]);
    fprintf(stdout,"\n");
    fprintf(stdout,"ncoefs12 %d\t", ncoefs12);
    for(int i = 0; i < ncoefs12; ++i)
      fprintf(stdout,"(%lf %d)\t", coefs12[i], offsets12[i]);
    fprintf(stdout,"\n");
    */
    s_gpu.ncoefs=ncoefs;
    s_gpu.ncoefs12=ncoefs12;
    s_gpu.ncoefs0=ncoefs0;

    s_gpu.coef_relax=s->coefs[0];

    if (ncoefs>0){
      GPAW_CUDAMALLOC(&(s_gpu.coefs_gpu),double,ncoefs);
      GPAW_CUDAMEMCPY(s_gpu.coefs_gpu,coefs,double,ncoefs, 
		      cudaMemcpyHostToDevice);
      
      GPAW_CUDAMALLOC(&(s_gpu.offsets_gpu),long,ncoefs);
      GPAW_CUDAMEMCPY(s_gpu.offsets_gpu,offsets,long,ncoefs,
		      cudaMemcpyHostToDevice);

      GPAW_CUDAMALLOC(&(s_gpu.offsets_gpuz),long,ncoefs);
      GPAW_CUDAMEMCPY(s_gpu.offsets_gpuz,offsetsz,long,ncoefs,
		      cudaMemcpyHostToDevice);
    }
    GPAW_CUDAMALLOC(&(s_gpu.coefs12_gpu),double,ncoefs12);
    GPAW_CUDAMEMCPY(s_gpu.coefs12_gpu,coefs12,double,ncoefs12,
		    cudaMemcpyHostToDevice);

    GPAW_CUDAMALLOC(&(s_gpu.offsets12_gpu),int,ncoefs12);
    GPAW_CUDAMEMCPY(s_gpu.offsets12_gpu,offsets12,int,ncoefs12,
		    cudaMemcpyHostToDevice);

    GPAW_CUDAMALLOC(&(s_gpu.offsets12_gpuz),int,ncoefs12);
    GPAW_CUDAMEMCPY(s_gpu.offsets12_gpuz,offsets12z,int,ncoefs12,
		    cudaMemcpyHostToDevice);

    GPAW_CUDAMALLOC(&(s_gpu.coefs0_gpu),double,ncoefs0);
    GPAW_CUDAMEMCPY(s_gpu.coefs0_gpu,coefs0,double,ncoefs0,
		    cudaMemcpyHostToDevice);

    GPAW_CUDAMALLOC(&(s_gpu.offsets0_gpu),int,ncoefs0);
    GPAW_CUDAMEMCPY(s_gpu.offsets0_gpu,offsets0,int,ncoefs0,
		    cudaMemcpyHostToDevice);

    return s_gpu;
  }

}



#endif
#endif
