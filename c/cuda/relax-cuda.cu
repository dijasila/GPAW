#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include "gpaw-cuda-int.h"



#define BLOCK_SIZEX 16
#define BLOCK_SIZEY 8
#define MAXCOEFS  24
#define MAXJ      8

#define ACACHE_SIZEX  (BLOCK_SIZEX+MAXJ)    
#define ACACHE_SIZEY  (BLOCK_SIZEY+MAXJ)


//__constant__ long3 c_n;
//__constant__ long3 c_j;
__constant__ long c_offsets[MAXCOEFS];
__constant__ double c_coefs[MAXCOEFS];
__constant__ int c_offsets1[MAXCOEFS];
__constant__ double c_coefs1[MAXCOEFS];
//__constant__ int c_jb[3];
//__constant__ int3 c_jb;
//__constant__ int  c_ncoefs;
	


__global__ void bmgs_relax_cuda_kernel(const int relax_method,const int ncoefs,const int ncoefs1,double* a,double* b,const double* src,const long3  c_n,const long3 c_j,const int3 c_jb,const double w)
{

  
  int i0=blockIdx.x;
  int i1=blockIdx.y*blockDim.y;
  

  __shared__ double acache[ACACHE_SIZEY][ACACHE_SIZEX];
  int i2,c/*,xind,yind,xsize,ysize,i0c*/;
  double x;
  double *aa,*aacache,*aacache2;
  int sizez=c_jb.z+c_n.z;
  
  a+=i0*(c_j.y+c_n.y*sizez)+(i1+threadIdx.y)*sizez+threadIdx.x;
  b+=i0*c_n.y*c_n.z+(i1+threadIdx.y)*c_n.z+threadIdx.x;
  src+=i0*c_n.y*c_n.z+(i1+threadIdx.y)*c_n.z+threadIdx.x;

  aacache2=&acache[0][0]+ACACHE_SIZEX*(threadIdx.y+c_jb.y/2)+threadIdx.x+c_jb.z/2;
  
  aacache=&acache[0][0]+ACACHE_SIZEX*(threadIdx.y)+threadIdx.x;    
  
  
	
  if (relax_method == 1)
    {			
      /* Weighted Gauss-Seidel relaxation for the equation "operator" b = src
	 a contains the temporary array holding also the boundary values. */
      
      // Coefficient needed multiple times later
      const double coef = 1.0/c_coefs[0];
      
      // The number of steps in each direction
      //  long nstep[3] = {c_n.x, c_n.y, c_n.z};
      
      //  a += (c_j.x + c_j.y + c_j.z) / 2;
      
      /*NOT WORKIN ATM*/
      return;
      for (i2=0; i2 < c_n.z; i2+=BLOCK_SIZEX) {    
	if ((i2+threadIdx.x<c_n.z)  && (i1+threadIdx.y<c_n.y)){    
	  aa=a+i2;
	  x = 0.0;
	  for (c = 1; c < ncoefs; c++)
	    x += aa[c_offsets[c]] * c_coefs[c];
	  x = (src[i2] - x) * coef;
	  b[i2] = x;
	  *aa = x;
	}
      }
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

      
      aa=a-c_j.z/2-c_j.y/2;
      aacache[0]=*(aa);
      if (BLOCK_SIZEY>=MAXJ){
	if (threadIdx.y<c_jb.y)
	  aacache[ACACHE_SIZEX*BLOCK_SIZEY]=aa[BLOCK_SIZEY*sizez]; 
      }else{
	if (threadIdx.y<c_jb.y/2){
	  aa+=BLOCK_SIZEY*sizez;     
	  aacache[ACACHE_SIZEX*BLOCK_SIZEY]=*(aa); 
	  aa+=(c_jb.y/2)*sizez;
	  aacache[ACACHE_SIZEX*(BLOCK_SIZEY+c_jb.y/2)]=*(aa);
	}
      }
      aa=a-c_j.z/2-c_j.y/2+BLOCK_SIZEX;
      if  (threadIdx.x<c_jb.z){
	aacache[BLOCK_SIZEX]=*(aa);
	if (BLOCK_SIZEY>=MAXJ){
	  if (threadIdx.y<c_jb.y)
	    aacache[ACACHE_SIZEX*BLOCK_SIZEY+BLOCK_SIZEX]=aa[BLOCK_SIZEY*sizez]; 
	} else {
	  if (threadIdx.y<c_jb.y/2){
	    aa+=BLOCK_SIZEY*sizez;      
	    aacache[ACACHE_SIZEX*BLOCK_SIZEY+BLOCK_SIZEX]=*(aa); 
	    aa+=(c_jb.y/2)*sizez;
	    aacache[ACACHE_SIZEX*(BLOCK_SIZEY+c_jb.y/2)+BLOCK_SIZEX]=*(aa);
	  }
	}
      }
      
      __syncthreads();            
      if ((threadIdx.x<c_n.z)  && (i1+threadIdx.y<c_n.y)){    
	x = 0.0;	      
#pragma unroll 5
	for (c = 1; c < ncoefs1; c++){
	  x += aacache2[c_offsets1[c]] * c_coefs1[c];
	}	
#pragma unroll 2
	for (c = 0; c < ncoefs; c++){	  
	  x += a[c_offsets[c]] * c_coefs[c];
	}	
	*b = (1.0 - w) * *b + w * (*src - x)/c_coefs1[0];
      }

   
      for (i2=BLOCK_SIZEX; i2 < c_n.z; i2+=BLOCK_SIZEX) {    
	__syncthreads();

	if (threadIdx.x<c_j.z){      
	  aacache[0]=aacache[BLOCK_SIZEX];
	  if (BLOCK_SIZEY>=MAXJ){
	    if (threadIdx.y<c_jb.y)
	      aacache[ACACHE_SIZEX*BLOCK_SIZEY]=aacache[ACACHE_SIZEX*BLOCK_SIZEY+BLOCK_SIZEX];
	  } else {
	    if (threadIdx.y<c_jb.y/2){
	      aacache[ACACHE_SIZEX*BLOCK_SIZEY]=aacache[ACACHE_SIZEX*BLOCK_SIZEY+BLOCK_SIZEX];
	      aacache[ACACHE_SIZEX*(BLOCK_SIZEY+c_jb.y/2)]=aacache[ACACHE_SIZEX*(BLOCK_SIZEY+c_jb.y/2)+BLOCK_SIZEX];
	    }
	  }
	}
	__syncthreads();   
	a+=BLOCK_SIZEX;
	aa=a-c_j.z/2-c_j.y/2+c_j.z; 
	aacache[c_j.z]=*(aa);
	
	if (BLOCK_SIZEY>=MAXJ){
	  if (threadIdx.y<c_jb.y)
	    aacache[ACACHE_SIZEX*BLOCK_SIZEY+c_j.z]=aa[BLOCK_SIZEY*sizez];
	}else{
	  if (threadIdx.y<c_jb.y/2){
	    aa+=BLOCK_SIZEY*sizez;
	    aacache[ACACHE_SIZEX*BLOCK_SIZEY+c_j.z]=*(aa);
	    aa+=(c_jb.y/2)*sizez;
	    aacache[ACACHE_SIZEX*(BLOCK_SIZEY+c_jb.y/2)+c_j.z]=*(aa);
	  }
	}

	__syncthreads();         
	if ((i2+threadIdx.x<c_n.z)  && (i1+threadIdx.y<c_n.y)){    
	  
	  x = 0.0;	      
#pragma unroll 5
	  for (c = 1; c < ncoefs1; c++){
	    x += aacache2[c_offsets1[c]] * c_coefs1[c];
	  }	
#pragma unroll 2
	  for (c = 0; c < ncoefs; c++){	  
	    x += a[c_offsets[c]] * c_coefs[c];
	  }	
	  b[i2] = (1.0 - w) * b[i2] + w * (src[i2] - x)/c_coefs1[0];
	}
      }
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
	temp = (1.0 - w) * *b + w * (*src - x)/c_coefs[0];
	*b++ = temp;
	a++;
	src++;
	}
	a += c_j.z;
	}
	a += c_j.y;
	}
      */
    }

}

extern "C" {


  bmgsstencil_gpu bmgs_stencil_to_gpu(const bmgsstencil* s);

  void bmgs_relax_cuda_gpu(const int relax_method, const bmgsstencil_gpu* s_gpu, double* adev, double* bdev,const double* src, const double w)
  {
    int3 jb;
    
    jb.z=s_gpu->j[2];
    jb.y=s_gpu->j[1]/(s_gpu->j[2]+s_gpu->n[2]);


    long3 hc_n;
    long3 hc_j;    
    hc_n.x=s_gpu->n[0];    hc_n.y=s_gpu->n[1];    hc_n.z=s_gpu->n[2];
    hc_j.x=s_gpu->j[0];    hc_j.y=s_gpu->j[1];    hc_j.z=s_gpu->j[2];


    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_offsets,s_gpu->offsets_gpu,
					 sizeof(long)*s_gpu->ncoefs,0,
					 cudaMemcpyDeviceToDevice));
    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_coefs,s_gpu->coefs_gpu,
					 sizeof(double)*s_gpu->ncoefs,0,
					 cudaMemcpyDeviceToDevice));
    gpaw_cudaSafeCall( cudaMemcpyToSymbol(c_offsets1,s_gpu->offsets1_gpu,
					  sizeof(int)*s_gpu->ncoefs1,0,
					  cudaMemcpyDeviceToDevice));
    gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_coefs1,s_gpu->coefs1_gpu,
					 sizeof(double)*s_gpu->ncoefs1,0,
					 cudaMemcpyDeviceToDevice));

    int gridy=(s_gpu->n[1]+BLOCK_SIZEY-1)/BLOCK_SIZEY;

    dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY); 
    dim3 dimGrid(s_gpu->n[0],gridy);    

    bmgs_relax_cuda_kernel<<<dimGrid, dimBlock, 0>>>(relax_method,s_gpu->ncoefs,s_gpu->ncoefs1,adev+(s_gpu->j[0]+s_gpu->j[1]+s_gpu->j[2])/2,bdev,src,hc_n,hc_j,jb,w);    
    
    gpaw_cudaSafeCall(cudaGetLastError());
  }

  double bmgs_relax_cuda_cpu(const int relax_method, const bmgsstencil* s, double* a, double* b,const double* src, const double w)
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

    gpaw_cudaSafeCall(cudaMemcpy(a,adev,sizeof(double)*asize,
				 cudaMemcpyDeviceToHost));
    gpaw_cudaSafeCall(cudaMemcpy(b,bdev,sizeof(double)*bsize,
				 cudaMemcpyDeviceToHost));
    
    gpaw_cudaSafeCall(cudaFree(adev));
    gpaw_cudaSafeCall(cudaFree(bdev));
    gpaw_cudaSafeCall(cudaFree(srcdev));

    flops=(t1.tv_sec*1.0+t1.tv_usec/1000000.0-t0.tv_sec*1.0-t0.tv_usec/1000000.0); 
   
    return flops;
  
  }

}
