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


#ifndef CUGPAWCOMPLEX




#define BLOCK_X 16
#define BLOCK_Y 8
#define MAXCOEFS  24
#define MAXJ      8

#define ACACHE_X  (BLOCK_X+MAXJ)    
#define ACACHE_Y  (BLOCK_Y+MAXJ)


__constant__ long c_offsets[MAXCOEFS];
__constant__ double c_coefs[MAXCOEFS];
__constant__ int c_offsets1[MAXCOEFS];
__constant__ double c_coefs1[MAXCOEFS];


#endif


__global__ void Zcuda(bmgs_fd_cuda_kernel)(int ncoefs,int ncoefs1,const Tcuda* a,Tcuda* b,const long4 c_n,const long4 c_j,const int4 c_jb)
{
  int i0=blockIdx.x;
  
  int i1=blockIdx.y*blockDim.y;


  __shared__ Tcuda acache[ACACHE_Y][ACACHE_X];
  int i2,c;
  Tcuda x;
  const Tcuda *aa,*aacache2;
  Tcuda *aacache/*,*aaa*/;
  int sizez=c_jb.z+c_n.z;

  
  a+=i0*(c_j.y+c_n.y*sizez)+(i1+threadIdx.y)*sizez+threadIdx.x;
  b+=i0*c_n.y*c_n.z+(i1+threadIdx.y)*c_n.z+threadIdx.x;
  aacache2=&acache[0][0]+ACACHE_X*(threadIdx.y+c_jb.y/2)+threadIdx.x+c_jb.z/2;

  aacache=&acache[0][0]+ACACHE_X*(threadIdx.y)+threadIdx.x;    

  aa=a-c_j.z/2-c_j.y/2;
  aacache[0]=*(aa);
  if (BLOCK_Y>=MAXJ){
    if (threadIdx.y<c_jb.y)
      aacache[ACACHE_X*BLOCK_Y]=aa[BLOCK_Y*sizez]; 
  }else{
    if (threadIdx.y<c_jb.y/2){
      aa+=BLOCK_Y*sizez;     
      aacache[ACACHE_X*BLOCK_Y]=*(aa); 
      aa+=(c_jb.y/2)*sizez;
      aacache[ACACHE_X*(BLOCK_Y+c_jb.y/2)]=*(aa);
    }
  }
  aa=a-c_j.z/2-c_j.y/2+BLOCK_X;
  if  (threadIdx.x<c_jb.z){
    aacache[BLOCK_X]=*(aa);
    if (BLOCK_Y>=MAXJ){
      if (threadIdx.y<c_jb.y)
	aacache[ACACHE_X*BLOCK_Y+BLOCK_X]=aa[BLOCK_Y*sizez]; 
    } else {
      if (threadIdx.y<c_jb.y/2){
	aa+=BLOCK_Y*sizez;      
	aacache[ACACHE_X*BLOCK_Y+BLOCK_X]=*(aa); 
	aa+=(c_jb.y/2)*sizez;
	aacache[ACACHE_X*(BLOCK_Y+c_jb.y/2)+BLOCK_X]=*(aa);
      }
    }
  }

  __syncthreads();            

  if ((threadIdx.x<c_n.z)  && (i1+threadIdx.y<c_n.y)){    
    x = MAKED(0.0);	      
#pragma unroll 13
    for (c = 0; c < ncoefs1; c++){
      IADD(x,MULTD(aacache2[c_offsets1[c]] , c_coefs1[c]));
    }	

#pragma unroll 6
    for (c = 0; c < ncoefs; c++){	  
      IADD(x, MULTD(a[c_offsets[c]] , c_coefs[c]));
    }	
    
    *b = x;
  }

  for (i2=BLOCK_X; i2 < c_n.z; i2+=BLOCK_X) {    
    __syncthreads();  	  
    if (threadIdx.x<c_j.z){      
      aacache[0]=aacache[BLOCK_X];
      if (BLOCK_Y>=MAXJ){
	if (threadIdx.y<c_jb.y)
	  aacache[ACACHE_X*BLOCK_Y]=aacache[ACACHE_X*BLOCK_Y+BLOCK_X];
      } else {
	if (threadIdx.y<c_jb.y/2){
	  aacache[ACACHE_X*BLOCK_Y]=aacache[ACACHE_X*BLOCK_Y+BLOCK_X];
	  aacache[ACACHE_X*(BLOCK_Y+c_jb.y/2)]=aacache[ACACHE_X*(BLOCK_Y+c_jb.y/2)+BLOCK_X];
	}
      }
    }
    __syncthreads();     
    a+=BLOCK_X;
    aa=a-c_j.z/2-c_j.y/2+c_j.z; 
    aacache[c_j.z]=*(aa);

    if (BLOCK_Y>=MAXJ){
      if (threadIdx.y<c_jb.y)
	aacache[ACACHE_X*BLOCK_Y+c_j.z]=aa[BLOCK_Y*sizez];
    }else{
      if (threadIdx.y<c_jb.y/2){
	aa+=BLOCK_Y*sizez;
	aacache[ACACHE_X*BLOCK_Y+c_j.z]=*(aa);
	aa+=(c_jb.y/2)*sizez;
	aacache[ACACHE_X*(BLOCK_Y+c_jb.y/2)+c_j.z]=*(aa);
      }
    }
    __syncthreads();         

    if ((i2+threadIdx.x<c_n.z)  && (i1+threadIdx.y<c_n.y)){    
      
      x = MAKED(0.0);	      
#pragma unroll 13
      for (c = 0; c < ncoefs1; c++){
	IADD(x , MULTD(aacache2[c_offsets1[c]] , c_coefs1[c]));
      }	
#pragma unroll 6
      for (c = 0; c < ncoefs; c++){	  
	IADD(x , MULTD(a[c_offsets[c]] , c_coefs[c]));
      }	
      
      b[i2] = x;
    }

  }
  
}



extern "C" {



  void Zcuda(bmgs_fd_cuda_gpu)(const bmgsstencil_gpu* s_gpu, const Tcuda* adev, Tcuda* bdev)  
  {
    int4 jb;
    
    jb.z=s_gpu->j[2];
    jb.y=s_gpu->j[1]/(s_gpu->j[2]+s_gpu->n[2]);

    long4 hc_n;
    long4 hc_j;    
    hc_n.x=s_gpu->n[0];    hc_n.y=s_gpu->n[1];    hc_n.z=s_gpu->n[2];
    hc_j.x=s_gpu->j[0];    hc_j.y=s_gpu->j[1];    hc_j.z=s_gpu->j[2];

    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_offsets,s_gpu->offsets_gpu,
					 sizeof(long)*s_gpu->ncoefs,0,
					 cudaMemcpyDeviceToDevice));
    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_coefs,s_gpu->coefs_gpu,
					 sizeof(double)*s_gpu->ncoefs,0,
					 cudaMemcpyDeviceToDevice));
    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_offsets1,s_gpu->offsets1_gpu,
					 sizeof(int)*s_gpu->ncoefs1,0,
					 cudaMemcpyDeviceToDevice));
    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_coefs1,s_gpu->coefs1_gpu,
					 sizeof(double)*s_gpu->ncoefs1,0,
					 cudaMemcpyDeviceToDevice));
    
    int gridy=(s_gpu->n[1]+BLOCK_Y-1)/BLOCK_Y;

    dim3 dimBlock(BLOCK_X,BLOCK_Y); 
    dim3 dimGrid(s_gpu->n[0],gridy);    

   
    Zcuda(bmgs_fd_cuda_kernel)<<<dimGrid, dimBlock, 0>>>(s_gpu->ncoefs,s_gpu->ncoefs1,adev+(s_gpu->j[0]+s_gpu->j[1]+s_gpu->j[2])/2,bdev,hc_n,hc_j,jb);    


    gpaw_cudaSafeCall(cudaGetLastError());

  }

}

#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "fd-cuda.cu"

__global__ void bmgs_fd_cuda_kernel2(int ncoefs,int ncoefs1,const double* a,double* b,const long4 c_n,const long4 c_j,const int4 c_jb)
{

  //  return;
  int ty=(c_n.y+BLOCK_Y-1)/BLOCK_Y;
  int i0=blockIdx.y/ty;
  
  if (i0>=c_n.x) return;  
  int i1=(blockIdx.y-i0*ty)*blockDim.y;

  if (i1>=c_n.y) return;

  int i2=(blockIdx.x)*BLOCK_X+threadIdx.x; 
  // if (i1+threadIdx.y>=c_n.y) return;

  __shared__ double acache[ACACHE_Y][ACACHE_X];
  int c/*,xind,yind,xsize,ysize,i0c*/;
  double x;
  const double *aa;
  double *aacache/*,*aaa*/;
  /*double *bb,*bbb*/;
  //  int blodimy=MIN(blockDim.y,c_n.y-i1);  
  //int blodimy=blockDim.y;
  
  a+=i0*(c_j.y+c_n.y*(c_j.z+c_n.z))+(i1+threadIdx.y)*(c_j.z+c_n.z)+i2;
  b+=i0*c_n.y*c_n.z+(i1+threadIdx.y)*c_n.z+i2;

  aacache=&acache[0][0]+ACACHE_X*(threadIdx.y)+threadIdx.x+c_jb.z/2;

  //  aa=a-c_j.z/2-c_j.y/2;
  aa=a-c_j.y/2;

  aacache[0]=aa[0];
  if (threadIdx.y<c_jb.y){
    aacache[ACACHE_X*BLOCK_Y]=aa[BLOCK_Y*(c_jb.z+c_n.z)];
  }
  aacache+=ACACHE_X*(c_jb.y/2);
  if  (threadIdx.x<c_jb.z/2){
    //    aa=a-c_j.z/2;
    aacache[-c_jb.z/2]=a[-c_j.z/2];
    aacache[BLOCK_X]=a[BLOCK_X]; 
  }
  __syncthreads();            
  if ((i1+threadIdx.y<c_n.y) && (i2<c_n.z)){    

    x = 0.0;	      
#pragma unroll 13
    for (c = 0; c < ncoefs1; c++){
      x += aacache[c_offsets1[c]] * c_coefs1[c];
    }	
#pragma unroll 6
    for (c = 0; c < ncoefs; c++){	  
      x += a[c_offsets[c]] * c_coefs[c];
    }	
    
    *b = x;
  }
  
}

extern "C" {

  bmgsstencil_gpu bmgs_stencil_to_gpu(const bmgsstencil* s)  
  {
    bmgsstencil_gpu s_gpu={s->ncoefs,NULL,NULL,0,NULL,NULL,
			   {s->n[0],s->n[1],s->n[2]},
			   {s->j[0],s->j[1],s->j[2]}};
    
    long offsets[s->ncoefs];
    int  offsets1[s->ncoefs];
    double coefs[s->ncoefs],coefs1[s->ncoefs];
    long ncoefs=0,ncoefs1=0;

    for(int i = 0; i < s->ncoefs; i++){
      if (abs(s->offsets[i])<=(s->j[2]/2)){
	offsets1[ncoefs1]=s->offsets[i];
	coefs1[ncoefs1]=s->coefs[i];
	ncoefs1++;
      } else if ((abs(s->offsets[i])<=(s->j[1]/2))){
	offsets1[ncoefs1]=s->offsets[i]*ACACHE_X/(s->j[2]+s->n[2]);
	coefs1[ncoefs1]=s->coefs[i];
	ncoefs1++;
      }
      else{
	offsets[ncoefs]=s->offsets[i];
	coefs[ncoefs]=s->coefs[i];
	ncoefs++;
      }
      
    }
    s_gpu.ncoefs=ncoefs;
    s_gpu.ncoefs1=ncoefs1;
    //    printf("ncoefs %d %d %d\n",s->ncoefs,ncoefs,ncoefs1);


    gpaw_cudaSafeCall(cudaMalloc(&(s_gpu.coefs_gpu),sizeof(double)*ncoefs));
    gpaw_cudaSafeCall(cudaMemcpy(s_gpu.coefs_gpu,coefs,sizeof(double)*ncoefs,
				 cudaMemcpyHostToDevice));
    
    gpaw_cudaSafeCall(cudaMalloc(&(s_gpu.offsets_gpu),sizeof(long)*ncoefs));
    gpaw_cudaSafeCall(cudaMemcpy(s_gpu.offsets_gpu,offsets,sizeof(long)*ncoefs,
				 cudaMemcpyHostToDevice));

    gpaw_cudaSafeCall(cudaMalloc(&(s_gpu.coefs1_gpu),sizeof(double)*ncoefs1));
    gpaw_cudaSafeCall(cudaMemcpy(s_gpu.coefs1_gpu,coefs1,
				 sizeof(double)*ncoefs1,
				 cudaMemcpyHostToDevice));

    gpaw_cudaSafeCall(cudaMalloc(&(s_gpu.offsets1_gpu),sizeof(int)*ncoefs1));
    gpaw_cudaSafeCall(cudaMemcpy(s_gpu.offsets1_gpu,offsets1,
				 sizeof(int)*ncoefs1,cudaMemcpyHostToDevice));
    
    return s_gpu;
  }

}


extern "C" {
  double bmgs_fd_cuda_cpu(const bmgsstencil* s, const double* a, double* b)
  {
  
    double *adev,*bdev;//,*adevp,*bdevp,*bp;

    size_t asize,bsize;//,asizep,bsizep;
    struct timeval  t0, t1; 
    double flops;
    bmgsstencil_gpu s_gpu=bmgs_stencil_to_gpu(s);

    asize=s->j[0]+s->n[0]*(s->j[1]+s->n[1]*(s->n[2]+s->j[2]));
    bsize=s->n[0]*s->n[1]*s->n[2];


    gpaw_cudaSafeCall(cudaMalloc(&adev,sizeof(double)*asize));


    gpaw_cudaSafeCall(cudaMalloc(&bdev,sizeof(double)*bsize));

   
    gpaw_cudaSafeCall(cudaMemcpy(adev,a,sizeof(double)*asize,cudaMemcpyHostToDevice));

    gettimeofday(&t0,NULL);  
    
    bmgs_fd_cuda_gpu(&s_gpu, adev,bdev);
      
    
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
