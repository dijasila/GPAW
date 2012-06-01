#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include "gpaw-cuda-int.h"


#ifndef MYJ

#define BLOCK_X 16
#define BLOCK_X_B BLOCK_X
#define BLOCK_Y 8
#define BLOCK_Y_B BLOCK_Y

#endif

#ifdef MYJ	
#undef  ACACHE_Y 
#undef  ACACHE_X 
#define ACACHE_Y  ((BLOCK_Y)+MYJ*2)
#define ACACHE_X  ((BLOCK_X)+MYJ*2)


__global__ void RELAX_kernel(const int relax_method,const double coef_relax,
			     const int ncoefs,const double *c_coefs,
			     const long *c_offsets,
			     const double *c_coefs0,
			     const double *c_coefs1,
			     const double *c_coefs2,
			     const double* a,double* b,
			     const double* src,const long3  c_n,
			     const int3 a_size,const int3 b_size,
			     const double w,const int xdiv)
{
  int xx=gridDim.x/xdiv;

  int xind=blockIdx.x/xx;
  
  int i2tid=threadIdx.x;
  int i2=(blockIdx.x-xind*xx)*BLOCK_X+i2tid;

  int i1tid=threadIdx.y;
  int i1=blockIdx.y*BLOCK_Y+i1tid;

  __shared__ double s_coefs0[MYJ*2+1];
  __shared__ double s_coefs1[MYJ*2];
  __shared__ double s_coefs2[MYJ*2];
  __shared__ double acache12[ACACHE_Y*ACACHE_X];

  double acache0[MYJ];
  double acache0t[MYJ+1];

  double *acache12p;

  int xlen=(c_n.x+xdiv-1)/xdiv;
  int xstart=xind*xlen;

  if ((c_n.x-xstart) < xlen)
    xlen=c_n.x-xstart;

  a+=xstart*a_size.y+i1*a_size.z+i2;
  b+=xstart*b_size.y+i1*b_size.z+i2;
  src+=xstart*b_size.y+i1*b_size.z+i2;


  acache12p=acache12+ACACHE_X*(i1tid+MYJ)+i2tid+MYJ;
  
  if (i2tid<=MYJ*2)
    s_coefs0[i2tid]=c_coefs0[i2tid];
  if (i2tid<MYJ*2){
    s_coefs1[i2tid]=c_coefs1[i2tid];
    s_coefs2[i2tid]=c_coefs2[i2tid];
  }  
  __syncthreads();
  
	
  if (relax_method == 1)
    {			
      /* Weighted Gauss-Seidel relaxation for the equation "operator" b = src
	 a contains the temporary array holding also the boundary values. */
      
      // Coefficient needed multiple times later
      //      const double coef = 1.0/c_coefs[0];
      
      /*NOT WORKIN ATM*/
      return;
    }
  else
    {
      /* Weighted Jacobi relaxation for the equation "operator" b = src
	 a contains the temporariry array holding also the boundary values. */

      for (int c=0;c<MYJ;c++){
	if ((i1<c_n.y) && (i2<c_n.z)) 
	  acache0[c]=a[(c-MYJ)*(a_size.y)];
      }

      for (int i0=0; i0 < xlen; i0++) {  
	if (i1<c_n.y+MYJ) {
	  acache12p[-MYJ]=a[-MYJ];
	  if  ((i2tid<MYJ*2) && (i2<c_n.z+MYJ-BLOCK_X+MYJ))
	    acache12p[BLOCK_X-MYJ]=a[BLOCK_X-MYJ];
	}
	if  (i1tid<MYJ) {
	  acache12p[-ACACHE_X*MYJ]=a[-a_size.z*MYJ];
	  if  (i1<c_n.y+MYJ-BLOCK_Y)
	    acache12p[ACACHE_X*BLOCK_Y]=a[a_size.z*BLOCK_Y];      
	}
	__syncthreads();         
	
	acache0t[0]=0.0;
	
	for (int c = 0; c < MYJ; c++)
	  acache0t[0]+=acache12p[ACACHE_X*(c-MYJ)]*s_coefs1[c];
	for (int c = 0; c < MYJ; c++)
	  acache0t[0]+=acache12p[c-MYJ]*s_coefs2[c];
	for (int c = 0; c < MYJ; c++)
	  acache0t[0]+=acache12p[(c+1)]*s_coefs2[c+MYJ];        
	for (int c = 0; c < MYJ; c++)
	  acache0t[0]+=acache12p[ACACHE_X*(c+1)]*s_coefs1[c+MYJ];    
	for (int c = 0; c < MYJ; c++)
	  acache0t[0]+=acache0[c]*s_coefs0[c];    
	
	for (int c = 0; c < MYJ; c++)
	  acache0t[c+1]+= acache12p[0]*s_coefs0[c+1+MYJ];
	for (int c = 0; c < ncoefs; c++)
	  acache0t[0]+=a[c_offsets[c]]*c_coefs[c];
	
	if (i0>=MYJ) {
	  if ((i1<c_n.y) && (i2<c_n.z)) {
	    b[0] = (1.0 - w) * b[0] + w * (src[0] - acache0t[MYJ])/coef_relax;
	  }
	  b+=b_size.y;
	  src+=b_size.y;
	}    
	
	for (int c=0;c<MYJ-1;c++){
	  acache0[c]=acache0[c+1];
	}   
	acache0[MYJ-1]= acache12p[0];
	
	for (int c=MYJ;c>0;c--){
	  acache0t[c]=acache0t[c-1];
	}   
	a+=a_size.y;
	__syncthreads();  
	
      }
#pragma unroll  
      for (int i0=0; i0 < MYJ; i0++) { 
	if ((i1<c_n.y) && (i2<c_n.z)) 
	  acache0[0]=a[0];
	
	if (i0 < 1)
	  acache0t[1-i0]+=acache0[0]*s_coefs0[1+MYJ];
#if MYJ >= 2
	if (i0 < 2)
	  acache0t[2-i0]+=acache0[0]*s_coefs0[2+MYJ];
#endif
#if MYJ >= 3
	if (i0 < 3)
	  acache0t[3-i0]+=acache0[0]*s_coefs0[3+MYJ];
#endif
#if MYJ >= 4
	if (i0 < 4)
	  acache0t[4-i0]+=acache0[0]*s_coefs0[4+MYJ];
#endif
#if MYJ >= 5
	if (i0 < 5)
	  acache0t[5-i0]+=acache0[0]*s_coefs0[5+MYJ];
#endif
	
	if (i0+xlen>=MYJ) {
	  if ((i1<c_n.y) && (i2<c_n.z)) {
	    b[0] = (1.0 - w) * b[0] + 
	      w * (src[0] - acache0t[MYJ-i0])/coef_relax;
	  }
	  b+=b_size.y;
	  src+=b_size.y;
	}    
	
	a+=a_size.y;   
      }  
      
    }
  
}



__global__ void RELAX_kernel_onlyb(const int relax_method,
				   const double coef_relax,
				   const int ncoefs,const double *c_coefs,
				   const long *c_offsets,
				   const double *c_coefs0,
				   const double *c_coefs1,
				   const double *c_coefs2,
				   const double* a,double* b,
				   const double* src,const long3  c_n,
				   const int3 c_jb,const int boundary,
				   const double w,const int xdiv)
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
  int blockix;

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
      if ((blockix>=0) && (blockix<xdiv*xx)) {
	xind=blockix/xx;
	i2bl=blockix-xind*xx;
	i1bl=0;
	ymax=MIN(BLOCK_Y_B,ymax);
	//ymax=MIN(c_jb.y/2,ymax);
      }
      blockix-=xdiv*xx;
    }
    if ((boundary & GPAW_BOUNDARY_Y1) != 0) {
      if ((blockix>=0) && (blockix<xdiv*xx)) {
	xind=blockix/xx;
	i2bl=blockix-xind*xx;
	i1bl=0;
	//i1pitch=MAX(c_n.y-c_jb.y/2,0);
	i1pitch=MAX(c_n.y-BLOCK_Y_B,0);
      }
      blockix-=xdiv*xx;
    }
    if ((boundary & GPAW_BOUNDARY_Z0) != 0) {
      if ((blockix>=0) && (blockix<xdiv*yy2)) {
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
      blockix-=xdiv*yy2;
    }
    if ((boundary & GPAW_BOUNDARY_Z1) != 0) {
      if ((blockix>=0) && (blockix<xdiv*yy2)) {
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
      blockix-=xdiv*yy2;
    }
    if ((boundary & GPAW_BOUNDARY_X0) != 0) {
      xstart+=c_jb.x/2; 	
      xlen-=c_jb.x/2;
    }
    if ((boundary & GPAW_BOUNDARY_X1) != 0) {
      xlen-=c_jb.x/2;
      xmax-=c_jb.x/2;
    }
    xlen=(xlen+xdiv-1)/xdiv;
    xstart+=xind*xlen;        
  }

  int i2tid=threadIdx.x;
  int i2=i2pitch+i2bl*BLOCK_X_B+i2tid;

  int i1tid=threadIdx.y;
  int i1=i1pitch+i1bl*BLOCK_Y_B+i1tid;

  __shared__ double s_coefs0[MYJ*2+1];
  __shared__ double s_coefs1[MYJ*2];
  __shared__ double s_coefs2[MYJ*2];
  __shared__ double acache12[ACACHE_Y*ACACHE_X];

  double acache0[MYJ];
  double acache0t[MYJ+1];
  double *acache12p;
  int sizez=c_jb.z+c_n.z;  
  int sizeyz=(c_jb.y+c_n.y)*sizez;

  if ((xmax-xstart) < xlen)
    xlen=xmax-xstart;

  acache12p=acache12+ACACHE_X*(i1tid+MYJ)+i2tid+MYJ;
  
  if (i2tid<=MYJ*2)
    s_coefs0[i2tid]=c_coefs0[i2tid];
  if (i2tid<MYJ*2){
    s_coefs1[i2tid]=c_coefs1[i2tid];
    s_coefs2[i2tid]=c_coefs2[i2tid];
  }
  __syncthreads();
  
  a+=xstart*sizeyz+i1*sizez+i2;
  b+=xstart*c_n.y*c_n.z+i1*c_n.z+i2;
  src+=xstart*c_n.y*c_n.z+i1*c_n.z+i2;
	
  if (relax_method == 1)
    {			
      /* Weighted Gauss-Seidel relaxation for the equation "operator" b = src
	 a contains the temporary array holding also the boundary values. */
      
      // Coefficient needed multiple times later
      //      const double coef = 1.0/c_coefs[0];
      
      /*NOT WORKIN ATM*/
      return;
    }
  else    {
    /* Weighted Jacobi relaxation for the equation "operator" b = src
       a contains the temporariry array holding also the boundary values. */
    
    for (int c=0;c<MYJ;c++){
      if ((i1<ymax) && (i2<zmax)) 
	acache0[c]=a[(c-MYJ)*(sizeyz)];
    }
    
    for (int i0=0; i0 < xlen; i0++) {  
      if (i1<ymax+MYJ) {
	acache12p[-MYJ]=a[-MYJ];
	if  ((i2tid<MYJ*2) && (i2<zmax+MYJ-BLOCK_X_B+MYJ))
	  acache12p[BLOCK_X_B-MYJ]=a[BLOCK_X_B-MYJ];
      }
      if  (i1tid<MYJ) {
	acache12p[-ACACHE_X*MYJ]=a[-sizez*MYJ];
	if  (i1<ymax+MYJ-BLOCK_Y_B)
	  acache12p[ACACHE_X*BLOCK_Y_B]=a[sizez*BLOCK_Y_B];      
      }
      __syncthreads();         
      
      acache0t[0]=0.0;
      
      for (int c = 0; c < MYJ; c++)
	acache0t[0]+=acache12p[ACACHE_X*(c-MYJ)]*s_coefs1[c];
      for (int c = 0; c < MYJ; c++)
	acache0t[0]+=acache12p[c-MYJ]*s_coefs2[c];
      for (int c = 0; c < MYJ; c++)
	acache0t[0]+=acache12p[(c+1)]*s_coefs2[c+MYJ];        
      for (int c = 0; c < MYJ; c++)
	acache0t[0]+=acache12p[ACACHE_X*(c+1)]*s_coefs1[c+MYJ];    
      for (int c = 0; c < MYJ; c++)
	acache0t[0]+=acache0[c]*s_coefs0[c];    
      
      //acache0t[0]+=acache12p[0]*s_coefs0[MYJ];
    
      for (int c = 0; c < MYJ; c++)
	acache0t[c+1]+= acache12p[0]*s_coefs0[c+1+MYJ];

      for (int c = 0; c < ncoefs; c++)
	acache0t[0]+=a[c_offsets[c]]*c_coefs[c];
      
      if (i0>=MYJ) {
	if ((i1<ymax) && (i2<zmax)) {
	  b[0] = (1.0 - w) * b[0] + w * (src[0] - acache0t[MYJ])/coef_relax;
	}
	b+=c_n.y*c_n.z;
	src+=c_n.y*c_n.z;
      }    
      
      for (int c=0;c<MYJ-1;c++){
	acache0[c]=acache0[c+1];
      }   
      acache0[MYJ-1]= acache12p[0];
      
      for (int c=MYJ;c>0;c--){
	acache0t[c]=acache0t[c-1];
      }   
      a+=sizeyz;
      __syncthreads(); 
    }
#pragma unroll  
    for (int i0=0; i0 < MYJ; i0++) { 
      if ((i1<c_n.y) && (i2<c_n.z)) 
	acache0[0]=a[0];
      
      if (i0 < 1)
	acache0t[1-i0]+=acache0[0]*s_coefs0[1+MYJ];
#if MYJ >= 2
      if (i0 < 2)
	acache0t[2-i0]+=acache0[0]*s_coefs0[2+MYJ];
#endif
#if MYJ >= 3
      if (i0 < 3)
	acache0t[3-i0]+=acache0[0]*s_coefs0[3+MYJ];
#endif
#if MYJ >= 4
      if (i0 < 4)
	acache0t[4-i0]+=acache0[0]*s_coefs0[4+MYJ];
#endif
#if MYJ >= 5
      if (i0 < 5)
	acache0t[5-i0]+=acache0[0]*s_coefs0[5+MYJ];
#endif
      if (i0+xlen>=MYJ) {
	if ((i1<ymax) && (i2<zmax)) {
	  b[0] = (1.0 - w) * b[0] + w * (src[0] - acache0t[MYJ-i0])/coef_relax;
	  
	}
	b+=c_n.y*c_n.z;
	src+=c_n.y*c_n.z;
      }	
      a+=sizeyz;   
    }  
  }  
  
}


#else
#define MYJ  (2/2)
#  define RELAX_kernel relax_kernel2
#  define RELAX_kernel_onlyb relax_kernel2_onlyb
#  include "relax-cuda.cu"
#  undef RELAX_kernel
#  undef RELAX_kernel_onlyb
#  undef MYJ
#define MYJ  (4/2)
#  define RELAX_kernel relax_kernel4
#  define RELAX_kernel_onlyb relax_kernel4_onlyb
#  include "relax-cuda.cu"
#  undef RELAX_kernel
#  undef RELAX_kernel_onlyb
#  undef MYJ
#define MYJ  (6/2)
#  define RELAX_kernel relax_kernel6
#  define RELAX_kernel_onlyb relax_kernel6_onlyb
#  include "relax-cuda.cu"
#  undef RELAX_kernel
#  undef RELAX_kernel_onlyb
#  undef MYJ
#define MYJ  (8/2)
#  define RELAX_kernel relax_kernel8
#  define RELAX_kernel_onlyb relax_kernel8_onlyb
#  include "relax-cuda.cu"
#  undef RELAX_kernel
#  undef RELAX_kernel_onlyb
#  undef MYJ
#define MYJ  (10/2)
#  define RELAX_kernel relax_kernel10
#  define RELAX_kernel_onlyb relax_kernel10_onlyb
#  include "relax-cuda.cu"
#  undef RELAX_kernel
#  undef RELAX_kernel_onlyb
#  undef MYJ




extern "C" {


  bmgsstencil_gpu bmgs_stencil_to_gpu(const bmgsstencil* s);
  int bmgs_fd_boundary_test(const bmgsstencil_gpu* s,int boundary);


  void bmgs_relax_cuda_gpu(const int relax_method,
			   const bmgsstencil_gpu* s_gpu, double* adev, 
			   double* bdev,const double* src, const double w,
			   int boundary,cudaStream_t stream)
  {
    int3 jb;
    int3 bjb;
    int3 hc_bj;
    
    //jb.z=s_gpu->j[2];
    //jb.y=s_gpu->j[1]/(s_gpu->j[2]+s_gpu->n[2]);


    long3 hc_n;
    long3 hc_j;   

    if ((boundary & GPAW_BOUNDARY_SKIP) != 0) {
      if  (!bmgs_fd_boundary_test(s_gpu,boundary))
	return;
      
    } else if ((boundary & GPAW_BOUNDARY_ONLY) != 0) {
      if  (!bmgs_fd_boundary_test(s_gpu,boundary)){
	boundary&=~GPAW_BOUNDARY_ONLY;
	boundary|=GPAW_BOUNDARY_NORMAL;
      }
    }
    
 
    hc_n.x=s_gpu->n[0];    hc_n.y=s_gpu->n[1];    hc_n.z=s_gpu->n[2];
    hc_j.x=s_gpu->j[0];    hc_j.y=s_gpu->j[1];    hc_j.z=s_gpu->j[2];

    bjb.x=0;    bjb.y=0;    bjb.z=0;
    hc_bj.x=0;    hc_bj.y=0;    hc_bj.z=0;
    

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
      src+=bjb1.z+bjb1.y*(hc_bj.z+hc_n.z)+
	bjb1.x*((hc_bj.z+hc_n.z)*hc_n.y+hc_bj.y);
      
      adev=(Tcuda*)((double*)adev+jb1.z+jb1.y*(hc_j.z+hc_n.z)+
		    jb1.x*((hc_j.z+hc_n.z)*hc_n.y+hc_j.y));

    }else {
      adev=(Tcuda*)((double*)adev +(hc_j.x+hc_j.y+hc_j.z)/2);
    }
    if ((hc_n.x<=0) || (hc_n.y<=0) || (hc_n.z<=0))
      return;
    
    dim3 dimBlock(1,1,1);
    dim3 dimGrid(1,1,1);
    int xdiv=MIN(hc_n.z,4);
    if (((boundary & GPAW_BOUNDARY_NORMAL) != 0) ||
	((boundary & GPAW_BOUNDARY_SKIP) != 0)){
      dimGrid.x=xdiv*MAX((hc_n.z+BLOCK_X-1)/BLOCK_X,1);
      dimGrid.y=MAX((hc_n.y+BLOCK_Y-1)/BLOCK_Y,1);
      dimBlock.x=BLOCK_X;
      dimBlock.y=BLOCK_Y;
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
	dimGrid.x+=xdiv*xx;
      if ((boundary & GPAW_BOUNDARY_Y1) != 0) 
	dimGrid.x+=xdiv*xx;
      if ((boundary & GPAW_BOUNDARY_Z0) != 0) 
	dimGrid.x+=xdiv*yy2;
      if ((boundary & GPAW_BOUNDARY_Z1) != 0) 
	dimGrid.x+=xdiv*yy2;
      dimGrid.y=1;
      dimBlock.x=BLOCK_X_B;
      dimBlock.y=BLOCK_Y_B;

    }

    int3 sizea;
    sizea.z=hc_j.z+hc_n.z;
    sizea.y=sizea.z*hc_n.y+hc_j.y;
    sizea.x= sizea.y*hc_n.x+hc_j.x;

    int3 sizeb;
    sizeb.z=hc_bj.z+hc_n.z;
    sizeb.y=sizeb.z*hc_n.y+hc_bj.y;
    sizeb.x= sizeb.y*hc_n.x+hc_bj.x;    

    if (((boundary & GPAW_BOUNDARY_NORMAL) != 0) ||
	((boundary & GPAW_BOUNDARY_SKIP) != 0)){
      switch(s_gpu->ncoefs0) 
	{
	case 3:	  
	  relax_kernel2<<<dimGrid, dimBlock, 0, stream>>>
	    (relax_method,s_gpu->coef_relax,
	     s_gpu->ncoefs,s_gpu->coefs_gpu,s_gpu->offsets_gpu,	   
	     s_gpu->coefs0_gpu,s_gpu->coefs1_gpu,s_gpu->coefs2_gpu,
	     adev,bdev,src,hc_n,sizea,sizeb,w,xdiv);    
	  break;
	case 5:
	  relax_kernel4<<<dimGrid, dimBlock, 0, stream>>>
	    (relax_method,s_gpu->coef_relax,
	     s_gpu->ncoefs,s_gpu->coefs_gpu,s_gpu->offsets_gpu,
	     s_gpu->coefs0_gpu,s_gpu->coefs1_gpu,s_gpu->coefs2_gpu,
	     adev,bdev,src,hc_n,sizea,sizeb,w,xdiv);    
	  break;
	case 7:
	  relax_kernel6<<<dimGrid, dimBlock, 0, stream>>>
	    (relax_method,s_gpu->coef_relax,
	     s_gpu->ncoefs,s_gpu->coefs_gpu,s_gpu->offsets_gpu,
	     s_gpu->coefs0_gpu,s_gpu->coefs1_gpu,s_gpu->coefs2_gpu,
	     adev,bdev,src,hc_n,sizea,sizeb,w,xdiv);    
	  break;
	case 9:
	  relax_kernel8<<<dimGrid, dimBlock, 0, stream>>>
	    (relax_method,s_gpu->coef_relax,
	     s_gpu->ncoefs,s_gpu->coefs_gpu,s_gpu->offsets_gpu,
	     s_gpu->coefs0_gpu,s_gpu->coefs1_gpu,s_gpu->coefs2_gpu,
	     adev,bdev,src,hc_n,sizea,sizeb,w,xdiv);    
	  break;
	case 11:
	  relax_kernel10<<<dimGrid, dimBlock, 0, stream>>>
	    (relax_method,s_gpu->coef_relax,
	     s_gpu->ncoefs,s_gpu->coefs_gpu,s_gpu->offsets_gpu,
	     s_gpu->coefs0_gpu,s_gpu->coefs1_gpu,s_gpu->coefs2_gpu,
	     adev,bdev,src,hc_n,sizea,sizeb,w,xdiv);    
	  break;
	default:
	  assert(0);
	}	  
    } else if ((boundary & GPAW_BOUNDARY_ONLY) != 0) {
      switch(s_gpu->ncoefs0) 
	{
	case 3:
	  relax_kernel2_onlyb<<<dimGrid, dimBlock, 0, stream>>>
	    (relax_method,s_gpu->coef_relax,
	     s_gpu->ncoefs,s_gpu->coefs_gpu,s_gpu->offsets_gpu,
	     s_gpu->coefs0_gpu,s_gpu->coefs1_gpu,s_gpu->coefs2_gpu,
	     adev,bdev,src,hc_n,jb,boundary,w,xdiv);    
	  break;
	case 5:
	  relax_kernel4_onlyb<<<dimGrid, dimBlock, 0, stream>>>
	    (relax_method,s_gpu->coef_relax,
	     s_gpu->ncoefs,s_gpu->coefs_gpu,s_gpu->offsets_gpu,
	     s_gpu->coefs0_gpu,s_gpu->coefs1_gpu,s_gpu->coefs2_gpu,
	     adev,bdev,src,hc_n,jb,boundary,w,xdiv);    
	  break;
	case 7:
	  relax_kernel6_onlyb<<<dimGrid, dimBlock, 0, stream>>>
	    (relax_method,s_gpu->coef_relax,
	     s_gpu->ncoefs,s_gpu->coefs_gpu,s_gpu->offsets_gpu,
	     s_gpu->coefs0_gpu,s_gpu->coefs1_gpu,s_gpu->coefs2_gpu,
	     adev,bdev,src,hc_n,jb,boundary,w,xdiv);    
	  break;
	case 9:
	  relax_kernel8_onlyb<<<dimGrid, dimBlock, 0, stream>>>
	    (relax_method,s_gpu->coef_relax,
	     s_gpu->ncoefs,s_gpu->coefs_gpu,s_gpu->offsets_gpu,
	     s_gpu->coefs0_gpu,s_gpu->coefs1_gpu,s_gpu->coefs2_gpu,
	     adev,bdev,src,hc_n,jb,boundary,w,xdiv);    
	  break;
	case 11:
	  relax_kernel10_onlyb<<<dimGrid, dimBlock, 0, stream>>>
	    (relax_method,s_gpu->coef_relax,
	     s_gpu->ncoefs,s_gpu->coefs_gpu,s_gpu->offsets_gpu,
	     s_gpu->coefs0_gpu,s_gpu->coefs1_gpu,s_gpu->coefs2_gpu,
	     adev,bdev,src,hc_n,jb,boundary,w,xdiv); 
	  break;
	default:
	  assert(0);	  
	}
    }
    gpaw_cudaSafeCall(cudaGetLastError());
  }


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
    bmgs_relax_cuda_gpu(relax_method, &s_gpu, adev, bdev,srcdev, w,
			GPAW_BOUNDARY_NORMAL,0);
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
  
}
#endif
