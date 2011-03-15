#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include "gpaw-cuda-int.h"


#ifndef MYJ

__constant__ long c_offsets[FD_MAXCOEFS];
__constant__ double c_coefs[FD_MAXCOEFS];
__constant__ int c_offsets12[FD_MAXCOEFS];
__constant__ double c_coefs12[FD_MAXCOEFS];
__constant__ double c_coefs0[FD_MAXJ+1];

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
  for (c = 0; c < MYJ/2; c++)						\
    IADD(x , MULTD(acache0[c] , c_coefs0[c]));				\
  for (c = MYJ/2+1; c < MYJ+1; c++)					\
    IADD(x , MULTD(acache0[c] , c_coefs0[c]));				\
  if (!(i0e))								\
    b[0] = (1.0 - w) * b[0] + w * (src[0] - x)/c_coefs0[MYJ/2];		\
  b+=c_n.y*c_n.z;							\
  src+=c_n.y*c_n.z;							\
  a+=sizeyz;								\
  __syncthreads();							\


/*
__global__ void RELAX_kernel_bc(const int relax_method,const int ncoefs,
				const int ncoefs12,double* a,double* b,
				const double* src,const long3  c_n,
				const long3 c_j,const int3 c_jb,const double w)
{

  int i2bl=blockIdx.x/FD_XDIV;
  int xind=blockIdx.x-FD_XDIV*i2bl;
  
  int i2tid=threadIdx.x;
  int i2=i2bl*FD_BLOCK_X+i2tid;

  int i1bl=blockIdx.y;
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

  a+=xstart*sizeyz+i1*sizez+i2;
  b+=xstart*c_n.y*c_n.z+i1*c_n.z+i2;
  src+=xstart*c_n.y*c_n.z+i1*c_n.z+i2;
  

  acache12p=acache12+FD_ACACHE_X*(i1tid+MYJ/2)+i2tid+MYJ/2;
  
	
  if (relax_method == 1)
    {			
    // Weighted Gauss-Seidel relaxation for the equation "operator" b = src
    // a contains the temporary array holding also the boundary values. 
      
      // Coefficient needed multiple times later
      //      const double coef = 1.0/c_coefs[0];
      
      // The number of steps in each direction
      //  long nstep[3] = {c_n.x, c_n.y, c_n.z};
      
      //  a += (c_j.x + c_j.y + c_j.z) / 2;
      
//
      return;

    }
  else
    {
      // Weighted Jacobi relaxation for the equation "operator" b = src
      //	 a contains the temporariry array holding also the boundary values. 
      
      for (c=1;c<MYJ+1;c++)
	acache0[c]=MAKED(0);
      
      int borders=0;
      if (i1bl==0)                     borders|=(1 << 0);
      if (i1bl==(gridDim.y-1))         borders|=(1 << 1);
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
  
}
*/


__global__ void RELAX_kernel(const int relax_method,double coef_relax,
			     const int ncoefs,
			     const int ncoefs12,double* a,double* b,
			     const double* src,const long3  c_n,
			     const long3 c_j,const int3 c_jb,const double w)
{

  int i2bl=blockIdx.x/FD_XDIV;
  int xind=blockIdx.x-FD_XDIV*i2bl;
  
  int i2tid=threadIdx.x;
  int i2=i2bl*FD_BLOCK_X+i2tid;

  int i1tid=threadIdx.y;
  int i1=blockIdx.y*FD_BLOCK_Y+i1tid;

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

  a+=xstart*sizeyz+i1*sizez+i2;
  b+=xstart*c_n.y*c_n.z+i1*c_n.z+i2;
  src+=xstart*c_n.y*c_n.z+i1*c_n.z+i2;
  

  acache12p=acache12+FD_ACACHE_X*(i1tid+MYJ/2)+i2tid+MYJ/2;
  
	
  if (relax_method == 1)
    {			
      /* Weighted Gauss-Seidel relaxation for the equation "operator" b = src
	 a contains the temporary array holding also the boundary values. */
      
      // Coefficient needed multiple times later
      //      const double coef = 1.0/c_coefs[0];
      
      // The number of steps in each direction
      //  long nstep[3] = {c_n.x, c_n.y, c_n.z};
      
      //  a += (c_j.x + c_j.y + c_j.z) / 2;
      
      /*NOT WORKIN ATM*/
      return;
      /*      for (i2=0; i2 < c_n.z; i2+=BLOCK_SIZEX) {    
	if ((i2+threadIdx.x<c_n.z)  && (i1+threadIdx.y<c_n.y)){    
	  aa=a+i2;
	  x = 0.0;
	  for (c = 1; c < ncoefs; c++)
	    x += aa[c_offsets[c]] * c_coefs[c];
	  x = (src[i2] - x) * coef;
	  b[i2] = x;
	  *aa = x;
	}
	}*/
      /*
	for (int i0 = 0; i0 < c_n.x; i0++)
        {
      
	for (int i1 = 0; i1 < c_n.y; i1++)
	{

	for (int i2 = 0; i2 < c_n.z; i2++)
	{
	x = 0.0;
	for (int c = 1; c < ncoefs; c++)
	x += a[c_offsets[c]] * c_coefs[c];
	x = (*src - x) * coef;
	*b++ = x;
	*a++ = x;
	src++;
	}
	a += c_j.z;
	}
	a += c_j.y;
	}
      */

    }
  else
    {
      /* Weighted Jacobi relaxation for the equation "operator" b = src
	 a contains the temporariry array holding also the boundary values. */

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
#if   MYJ==2
#pragma unroll 4
#elif MYJ==4
#pragma unroll 8
#elif MYJ==6
#pragma unroll 12
#elif MYJ==8
#pragma unroll 16
#elif MYJ==10
#pragma unroll 20
#endif
	for (c = 0; c < ncoefs12; c++){
	  IADD(x , MULTD(acache12p[c_offsets12[c]] , c_coefs12[c]));
	}	
	for (c = 0; c < MYJ/2; c++){	  
	  IADD(x , MULTD(acache0[c] , c_coefs0[c]));
	}	    
	for (c = MYJ/2+1; c < MYJ+1; c++){	  
	  IADD(x , MULTD(acache0[c] , c_coefs0[c]));
	}	    
	for (c = 0; c < ncoefs; c++){	  
	  IADD(x , MULTD(a[c_offsets[c]] , c_coefs[c]));
	}	
	
	if ((i1<c_n.y) && (i2<c_n.z)) {
	  b[0] = (1.0 - w) * b[0] + w * (src[0] - x)/coef_relax;
	  
	}
	b+=c_n.y*c_n.z;
	src+=c_n.y*c_n.z;
	a+=sizeyz;
	__syncthreads();         	
      }
      
    }

}


#else
#define MYJ  2
#  define RELAX_kernel relax_kernel2
#  define RELAX_kernel_bc relax_kernel2_bc
#  include "relax-cuda.cu"
#  undef RELAX_kernel
#  undef RELAX_kernel_bc
#  undef MYJ
#define MYJ  4
#  define RELAX_kernel relax_kernel4
#  define RELAX_kernel_bc relax_kernel4_bc
#  include "relax-cuda.cu"
#  undef RELAX_kernel
#  undef RELAX_kernel_bc
#  undef MYJ
#define MYJ  6
#  define RELAX_kernel relax_kernel6
#  define RELAX_kernel_bc relax_kernel6_bc
#  include "relax-cuda.cu"
#  undef RELAX_kernel
#  undef RELAX_kernel_bc
#  undef MYJ
#define MYJ  8
#  define RELAX_kernel relax_kernel8
#  define RELAX_kernel_bc relax_kernel8_bc
#  include "relax-cuda.cu"
#  undef RELAX_kernel
#  undef RELAX_kernel_bc
#  undef MYJ
#define MYJ  10
#  define RELAX_kernel relax_kernel10
#  define RELAX_kernel_bc relax_kernel10_bc
#  include "relax-cuda.cu"
#  undef RELAX_kernel
#  undef RELAX_kernel_bc
#  undef MYJ




extern "C" {


  bmgsstencil_gpu bmgs_stencil_to_gpu(const bmgsstencil* s);

  void bmgs_relax_cuda_gpu(const int relax_method,
			   const bmgsstencil_gpu* s_gpu, double* adev, 
			   double* bdev,const double* src, const double w)
  {
    int3 jb;
    
    jb.z=s_gpu->j[2];
    jb.y=s_gpu->j[1]/(s_gpu->j[2]+s_gpu->n[2]);


    long3 hc_n;
    long3 hc_j;    
    hc_n.x=s_gpu->n[0];    hc_n.y=s_gpu->n[1];    hc_n.z=s_gpu->n[2];
    hc_j.x=s_gpu->j[0];    hc_j.y=s_gpu->j[1];    hc_j.z=s_gpu->j[2];
    
    if (s_gpu->ncoefs>0){
      gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_offsets,s_gpu->offsets_gpu,
					   sizeof(long)*s_gpu->ncoefs,0,
					   cudaMemcpyDeviceToDevice));
      gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_coefs,s_gpu->coefs_gpu,
					   sizeof(double)*s_gpu->ncoefs,0,
					   cudaMemcpyDeviceToDevice));
    }
    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_offsets12,s_gpu->offsets12_gpu,
					  sizeof(int)*s_gpu->ncoefs12,0,
					  cudaMemcpyDeviceToDevice));
    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_coefs12,s_gpu->coefs12_gpu,
					 sizeof(double)*s_gpu->ncoefs12,0,
					 cudaMemcpyDeviceToDevice));

    /*    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_offsets0,s_gpu->offsets0_gpu,
					 sizeof(int)*s_gpu->ncoefs0,0,
					 cudaMemcpyDeviceToDevice));*/
    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_coefs0,s_gpu->coefs0_gpu,
					 sizeof(double)*s_gpu->ncoefs0,0,
					 cudaMemcpyDeviceToDevice));

    adev+=(s_gpu->j[0]+s_gpu->j[1]+s_gpu->j[2])/2;
    
    int gridx=FD_XDIV*MAX((s_gpu->n[2]+FD_BLOCK_X-1)/FD_BLOCK_X,1);
    int gridy=MAX((s_gpu->n[1]+FD_BLOCK_Y-1)/FD_BLOCK_Y,1);
    
    dim3 dimBlock(FD_BLOCK_X,FD_BLOCK_Y); 
    dim3 dimGrid(gridx,gridy);    
    if (s_gpu->ncoefs0<=3)
      relax_kernel2<<<dimGrid, dimBlock, 0>>>
	(relax_method,s_gpu->coef_relax,s_gpu->ncoefs,s_gpu->ncoefs12,
	 adev,bdev,src,hc_n,hc_j, jb,w);    
    else if (s_gpu->ncoefs0<=5)
      relax_kernel4<<<dimGrid, dimBlock, 0>>>
	(relax_method,s_gpu->coef_relax,s_gpu->ncoefs,s_gpu->ncoefs12,adev,bdev,src,hc_n,hc_j,
	 jb,w);    
    else if (s_gpu->ncoefs0<=7)
      relax_kernel6<<<dimGrid, dimBlock, 0>>>
	(relax_method,s_gpu->coef_relax,s_gpu->ncoefs,s_gpu->ncoefs12,adev,bdev,src,hc_n,hc_j,
	 jb,w);    
    else if (s_gpu->ncoefs0<=9)
      relax_kernel8<<<dimGrid, dimBlock, 0>>>
	(relax_method,s_gpu->coef_relax,s_gpu->ncoefs,s_gpu->ncoefs12,adev,bdev,src,hc_n,hc_j,
	 jb,w);    
    else if (s_gpu->ncoefs0<=11)
      relax_kernel10<<<dimGrid, dimBlock, 0>>>
	(relax_method,s_gpu->coef_relax,s_gpu->ncoefs,s_gpu->ncoefs12,adev,bdev,src,hc_n,hc_j,
	 jb,w);    
    
    gpaw_cudaSafeCall(cudaGetLastError());
  }
  /*
  void bmgs_relax_cuda_gpu_bc(const int relax_method,
			      const bmgsstencil_gpu* s_gpu, double* adev, 
			      double* bdev,const double* src, const double w)
  {
    int3 jb;
    long3 hc_n;
    long3 hc_j;    

    hc_n.x=s_gpu->n[0];    hc_n.y=s_gpu->n[1];    hc_n.z=s_gpu->n[2];

    hc_j.x=0;    hc_j.y=0;    hc_j.z=0;
    
    jb.z=hc_j.z;
    jb.y=hc_j.y/(hc_j.z+hc_n.z);

    if (s_gpu->ncoefs>0){
      gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_offsets,s_gpu->offsets_gpu,
						sizeof(long)*s_gpu->ncoefs,0,
						cudaMemcpyDeviceToDevice));
      gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_coefs,s_gpu->coefs_gpu,
						sizeof(double)*s_gpu->ncoefs,0,
						cudaMemcpyDeviceToDevice));
    }
    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_offsets12,s_gpu->offsets12_gpu,
					      sizeof(int)*s_gpu->ncoefs12,0,
					      cudaMemcpyDeviceToDevice));
    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_coefs12,s_gpu->coefs12_gpu,
					      sizeof(double)*s_gpu->ncoefs12,0,
					      cudaMemcpyDeviceToDevice));
    
    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_coefs0,s_gpu->coefs0_gpu,
					      sizeof(double)*s_gpu->ncoefs0,0,
					      cudaMemcpyDeviceToDevice));
    
    adev+=(hc_j.x+hc_j.y+hc_j.z)/2;
    
    int gridx=FD_XDIV*MAX((s_gpu->n[2]+FD_BLOCK_X-1)/FD_BLOCK_X,1);
    int gridy=MAX((s_gpu->n[1]+FD_BLOCK_Y-1)/FD_BLOCK_Y,1);
    
    dim3 dimBlock(FD_BLOCK_X,FD_BLOCK_Y); 
    dim3 dimGrid(gridx,gridy);    
    if (s_gpu->ncoefs0<=3)
      relax_kernel2_bc<<<dimGrid, dimBlock, 0>>>
	(relax_method,s_gpu->coef_relax,s_gpu->ncoefs,s_gpu->ncoefs12,adev,bdev,src,hc_n,hc_j,
	 jb,w);    
    else if (s_gpu->ncoefs0<=5)
      relax_kernel4_bc<<<dimGrid, dimBlock, 0>>>
	(relax_method,s_gpu->coef_relax,s_gpu->ncoefs,s_gpu->ncoefs12,adev,bdev,src,hc_n,hc_j,
	 jb,w);    
    else if (s_gpu->ncoefs0<=7)
      relax_kernel6_bc<<<dimGrid, dimBlock, 0>>>
	(relax_method,s_gpu->coef_relax,s_gpu->ncoefs,s_gpu->ncoefs12,adev,bdev,src,hc_n,hc_j,
	 jb,w);    
    else if (s_gpu->ncoefs0<=9)
      relax_kernel8_bc<<<dimGrid, dimBlock, 0>>>
	(relax_method,s_gpu->coef_relax,s_gpu->ncoefs,s_gpu->ncoefs12,adev,bdev,src,hc_n,hc_j,
	 jb,w);    
    else if (s_gpu->ncoefs0<=11)
      relax_kernel10_bc<<<dimGrid, dimBlock, 0>>>
	(relax_method,s_gpu->coef_relax,s_gpu->ncoefs,s_gpu->ncoefs12,adev,bdev,src,hc_n,hc_j,
	 jb,w);    
    
    gpaw_cudaSafeCall(cudaGetLastError());
  }
  */
  double bmgs_relax_cuda_cpu(const int relax_method, const bmgsstencil* s,
			     double* a, double* b,const double* src, 
			     const double w)
  {
    double *adev,*bdev,*srcdev;
    size_t asize,bsize;
    struct timeval  t0, t1; 
    double flops;
    bmgsstencil_gpu s_gpu=bmgs_stencil_to_gpu(s);
    
    asize=s->j[0]+s->n[0]*(s->j[1]+s->n[1]*(s->n[2]+s->j[2]));
    bsize=s->n[0]*s->n[1]*s->n[2];

    gpaw_cudaSafeCall(cudaMalloc(&adev,sizeof(double)*asize));
   
    gpaw_cudaSafeCall(cudaMalloc(&bdev,sizeof(double)*bsize));
    gpaw_cudaSafeCall(cudaMalloc(&srcdev,sizeof(double)*bsize));
   
    gpaw_cudaSafeCall(cudaMemcpy(adev,a,sizeof(double)*asize,
				 cudaMemcpyHostToDevice));
    gpaw_cudaSafeCall(cudaMemcpy(bdev,b,sizeof(double)*bsize,
				 cudaMemcpyHostToDevice));
    gpaw_cudaSafeCall(cudaMemcpy(srcdev,src,sizeof(double)*bsize,
				 cudaMemcpyHostToDevice));
   
    gettimeofday(&t0,NULL);
    bmgs_relax_cuda_gpu(relax_method, &s_gpu, adev, bdev,srcdev, w);

    cudaThreadSynchronize();  
    gpaw_cudaSafeCall(cudaGetLastError());

    gettimeofday(&t1,NULL);

    gpaw_cudaSafeCall(cudaMemcpy(b,bdev,sizeof(double)*bsize,
				 cudaMemcpyDeviceToHost));
    
    gpaw_cudaSafeCall(cudaFree(adev));
    gpaw_cudaSafeCall(cudaFree(bdev));
    gpaw_cudaSafeCall(cudaFree(srcdev));

    flops=(t1.tv_sec*1.0+t1.tv_usec/1000000.0-t0.tv_sec*1.0-t0.tv_usec/1000000.0); 
   
    return flops;
  
  }
  /*
  double bmgs_relax_cuda_cpu_bc(const int relax_method, const bmgsstencil* s,
				double* a, double* b,const double* src, 
				const double w)
  {
    double *adev,*bdev,*srcdev;
    size_t bsize;
    struct timeval  t0, t1; 
    double flops;
    bmgsstencil_gpu s_gpu=bmgs_stencil_to_gpu(s);
    
    bsize=s->n[0]*s->n[1]*s->n[2];

    gpaw_cudaSafeCall(cudaMalloc(&adev,sizeof(double)*bsize));
   
    gpaw_cudaSafeCall(cudaMalloc(&bdev,sizeof(double)*bsize));
    gpaw_cudaSafeCall(cudaMalloc(&srcdev,sizeof(double)*bsize));
   
    gpaw_cudaSafeCall(cudaMemcpy(adev,a,sizeof(double)*bsize,
				 cudaMemcpyHostToDevice));
    gpaw_cudaSafeCall(cudaMemcpy(bdev,b,sizeof(double)*bsize,
				 cudaMemcpyHostToDevice));
    gpaw_cudaSafeCall(cudaMemcpy(srcdev,src,sizeof(double)*bsize,
				 cudaMemcpyHostToDevice));
   
    gettimeofday(&t0,NULL);
    bmgs_relax_cuda_gpu_bc(relax_method, &s_gpu, adev, bdev,srcdev, w);

    cudaThreadSynchronize();  
    gpaw_cudaSafeCall(cudaGetLastError());

    gettimeofday(&t1,NULL);

    gpaw_cudaSafeCall(cudaMemcpy(b,bdev,sizeof(double)*bsize,
				 cudaMemcpyDeviceToHost));
    
    gpaw_cudaSafeCall(cudaFree(adev));
    gpaw_cudaSafeCall(cudaFree(bdev));
    gpaw_cudaSafeCall(cudaFree(srcdev));

    flops=(t1.tv_sec*1.0+t1.tv_usec/1000000.0-t0.tv_sec*1.0-t0.tv_usec/1000000.0); 
   
    return flops;
  
  }
  */
}
#endif
