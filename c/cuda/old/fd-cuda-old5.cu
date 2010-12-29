// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include <cuComplex.h>

#include "bmgs-cuda.h"



#ifndef BMGSCOMPLEX

#define BLOCK_X 16
#define BLOCK_Y 8
#define MAXCOEFS  24
#define MAXJ      8

#define ACACHE_X  (BLOCK_X+MAXJ)    
#define ACACHE_Y  (BLOCK_Y+MAXJ)


//#define MAX(a,b) ((a)>(b)?(a):(b))
//#define MIN(a,b) ((a)<(b)?(a):(b))

//__constant__ long3 c_n;
//__constant__ long3 c_j;
__constant__ long c_offsets[MAXCOEFS];
__constant__ double c_coefs[MAXCOEFS];
__constant__ int c_offsets1[MAXCOEFS];
__constant__ double c_coefs1[MAXCOEFS];
//__constant__ int3 c_jb;

#endif


__global__ void Zcuda(bmgs_fd_cuda_kernel)(int ncoefs,int ncoefs1,const Tcuda* a,Tcuda* b,const long4 c_n,const long4 c_j,const int4 c_jb)
{
  int i0=blockIdx.x;
  
  //if (i0>=c_n.x) return;  
  int i1=blockIdx.y*blockDim.y;

  //if (i1>=c_n.y) return;  
  // if (i1+threadIdx.y>=c_n.y) return;

  __shared__ Tcuda acache[ACACHE_Y][ACACHE_X];
  int i2,c/*,xind,yind,xsize,ysize,i0c*/;
  Tcuda x;
  const Tcuda *aa,*aacache2;
  Tcuda *aacache/*,*aaa*/;
  int sizez=c_jb.z+c_n.z;
  /*double *bb,*bbb*/;
  //  int blodimy=MIN(blockDim.y,c_n.y-i1);  
  //int blodimy=blockDim.y;
  
  a+=i0*(c_j.y+c_n.y*sizez)+(i1+threadIdx.y)*sizez+threadIdx.x;
  b+=i0*c_n.y*c_n.z+(i1+threadIdx.y)*c_n.z+threadIdx.x;
  aacache2=&acache[0][0]+ACACHE_X*(threadIdx.y+c_jb.y/2)+threadIdx.x+c_jb.z/2;

  aacache=&acache[0][0]+ACACHE_X*(threadIdx.y)+threadIdx.x;    

  aa=a-c_j.z/2-c_j.y/2;
  //#pragma unroll 3
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
  /*
  for (c=threadIdx.y;c<c_jb.y+BLOCK_Y;c+=BLOCK_Y){
    acache[c][threadIdx.x]=*(aa);
    aa+=BLOCK_Y*(c_jb.z+c_n.z);      
  }
  */
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
    /*
#pragma unroll 3
    for (c=threadIdx.y;c<c_jb.y+BLOCK_Y;c+=BLOCK_Y){
      acache[c][threadIdx.x+BLOCK_X]=*(aa);
      aa+=BLOCK_Y*(c_jb.z+c_n.z);      
    }
    */
  }
  __syncthreads();            
  if ((threadIdx.x<c_n.z)  && (i1+threadIdx.y<c_n.y)){    
    x = MAKED(0.0);	      
#pragma unroll 13
    for (c = 0; c < ncoefs1; c++){
      IADD(x,MULCD(aacache2[c_offsets1[c]] , c_coefs1[c]));
    }	
#pragma unroll 6
    for (c = 0; c < ncoefs; c++){	  
      IADD(x, MULCD(a[c_offsets[c]] , c_coefs[c]));
    }	
    
    *b = x;
  }

  for (i2=BLOCK_X; i2 < c_n.z; i2+=BLOCK_X) {    
    /*aa=a+i2-c_j.z/2-c_j.y/2;
#pragma unroll 3
    for (i=threadIdx.y;i<c_jb.y+BLOCK_Y;i+=BLOCK_Y){
      acache[i][threadIdx.x]=*(aa);
      aa+=BLOCK_Y*(c_jb.z+c_n.z);      
    }
    
    if  (threadIdx.x<c_jb.z){
      aa=a+i2-c_j.z/2-c_j.y/2+BLOCK_X;
#pragma unroll 3
      for (i=threadIdx.y;i<c_jb.y+BLOCK_Y;i+=BLOCK_Y){
	acache[i][threadIdx.x+BLOCK_X]=*(aa);
	aa+=BLOCK_Y*(c_jb.z+c_n.z);      
      }
    }
    __syncthreads();
    */
    __syncthreads();  	  
    if (threadIdx.x<c_j.z){      
    //    if (threadIdx.x<MAXJ){      
      //#pragma unroll 3
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
      /*      for (c=threadIdx.y;c<c_jb.y+BLOCK_Y;c+=BLOCK_Y){	
	acache[c][threadIdx.x]=acache[c][threadIdx.x+BLOCK_X];
	}*/
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
    /*#pragma unroll 2
    for (c=threadIdx.y;c<c_jb.y;c+=BLOCK_Y){
      aa+=BLOCK_Y*(c_j.z+c_n.z);
      acache[c+BLOCK_Y][threadIdx.x+c_j.z]=*(aa);
    }
    */
    __syncthreads();         

    if ((i2+threadIdx.x<c_n.z)  && (i1+threadIdx.y<c_n.y)){    
      
      x = MAKED(0.0);	      
#pragma unroll 13
      for (c = 0; c < ncoefs1; c++){
	IADD(x , MULCD(aacache2[c_offsets1[c]] , c_coefs1[c]));
      }	
      //      aa=a+i2;
#pragma unroll 6
      for (c = 0; c < ncoefs; c++){	  
	IADD(x , MULCD(a[c_offsets[c]] , c_coefs[c]));
      }	
      
      b[i2] = x;
    }

  }
  
}



extern "C" {



  void Zcuda(bmgs_fd_cuda_gpu)(const bmgsstencil_gpu* s_gpu, const Tcuda* adev, Tcuda* bdev)  
  {
    //    size_t asize,bsize;
    
    /*    long offsets[s->ncoefs];
    int  offsets1[s->ncoefs];
    double coefs[s->ncoefs],coefs1[s->ncoefs];*/
    int4 jb;
    //    long ncoefs=0,ncoefs1=0;
    //    bmgsstencil_gpu s_gpu=bmgs_stencil_to_gpu(s);


    /*    asize=s->j[0]+s->n[0]*(s->j[1]+s->n[1]*(s->n[2]+s->j[2]));
	  bsize=s->n[0]*s->n[1]*s->n[2];*/
    
    jb.z=s_gpu->j[2];
    jb.y=s_gpu->j[1]/(s_gpu->j[2]+s_gpu->n[2]);

    /*    for(int i = 0; i < s->ncoefs; i++){
      if (abs(s->offsets[i])<=(s->j[2]/2)){
	offsets1[ncoefs1]=s->offsets[i];
	coefs1[ncoefs1]=s->coefs[i];
	ncoefs1++;
      } else if (abs(s->offsets[i])<=(s->j[1]/2)){
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
    */
    long4 hc_n;
    long4 hc_j;    
    hc_n.x=s_gpu->n[0];    hc_n.y=s_gpu->n[1];    hc_n.z=s_gpu->n[2];
    hc_j.x=s_gpu->j[0];    hc_j.y=s_gpu->j[1];    hc_j.z=s_gpu->j[2];
    /*    
    cudaMemcpyToSymbol(c_offsets,offsets,sizeof(long)*ncoefs);
    cudaMemcpyToSymbol(c_coefs,coefs,sizeof(double)*ncoefs);
    cudaMemcpyToSymbol(c_offsets1,offsets1,sizeof(int)*ncoefs1);
    cudaMemcpyToSymbol(c_coefs1,coefs1,sizeof(double)*ncoefs1);
    */
    cudaMemcpyToSymbol(c_offsets,s_gpu->offsets_gpu,
		       sizeof(long)*s_gpu->ncoefs,0,cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_coefs,s_gpu->coefs_gpu,
		       sizeof(double)*s_gpu->ncoefs,0,cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_offsets1,s_gpu->offsets1_gpu,
		       sizeof(int)*s_gpu->ncoefs1,0,cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_coefs1,s_gpu->coefs1_gpu,
		       sizeof(double)*s_gpu->ncoefs1,0,cudaMemcpyDeviceToDevice);
    
    int gridy=(s_gpu->n[1]+BLOCK_Y-1)/BLOCK_Y;
    //int gridx=(s_gpu->n[2]+BLOCK_X-1)/BLOCK_X;
    //    gridy=(s_gpu->n[1]%BLOCK_Y) ? gridy+1 : gridy;

    dim3 dimBlock(BLOCK_X,BLOCK_Y); 
    //dim3 dimGrid(gridx,gridy*s_gpu->n[0]);    
    dim3 dimGrid(s_gpu->n[0],gridy);    

    //   fprintf(stdout,"array: %d x %d x %d\t",s_gpu->n[0],s_gpu->n[1],s_gpu->n[2]);
    //    fprintf(stdout,"block: %d x %d\t grid: %d x %d\n",BLOCK_X,BLOCK_Y,gridx,gridy*s_gpu->n[0]);
    
    
    Zcuda(bmgs_fd_cuda_kernel)<<<dimGrid, dimBlock, 0>>>(s_gpu->ncoefs,s_gpu->ncoefs1,adev+(s_gpu->j[0]+s_gpu->j[1]+s_gpu->j[2])/2,bdev,hc_n,hc_j,jb);    
    
    
    
  }

  /*  void bmgs_fd_cuda_gpu(const bmgsstencil* s, const double* adev, double* bdev)  
  {
    bmgsstencil_gpu s_gpu=bmgs_stencil_to_gpu(s);
    bmgs_fd_cuda_gpu2(&s_gpu,adev,bdev); 
  }
  */
  
}

#ifndef BMGSCOMPLEX
#define BMGSCOMPLEX
#include "fd-cuda-old5.cu"

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


    cudaMalloc(&(s_gpu.coefs_gpu),sizeof(double)*ncoefs);
    cudaMemcpy(s_gpu.coefs_gpu,coefs,sizeof(double)*ncoefs,cudaMemcpyHostToDevice);
    
    cudaMalloc(&(s_gpu.offsets_gpu),sizeof(long)*ncoefs);
    cudaMemcpy(s_gpu.offsets_gpu,offsets,sizeof(long)*ncoefs,cudaMemcpyHostToDevice);

    cudaMalloc(&(s_gpu.coefs1_gpu),sizeof(double)*ncoefs1);
    cudaMemcpy(s_gpu.coefs1_gpu,coefs1,sizeof(double)*ncoefs1,cudaMemcpyHostToDevice);

    cudaMalloc(&(s_gpu.offsets1_gpu),sizeof(int)*ncoefs1);
    cudaMemcpy(s_gpu.offsets1_gpu,offsets1,sizeof(int)*ncoefs1,cudaMemcpyHostToDevice);
    
    return s_gpu;
  }

}


extern "C" {
  double bmgs_fd_cuda_cpu(const bmgsstencil* s, const double* a, double* b)
  {
    //    bmgsstencil stemp=*s,*sdev;
  
    double *adev,*bdev;//,*adevp,*bdevp,*bp;
    //    const double *ap;
    //cudaPitchedPtr bdevp;

    size_t asize,bsize;//,asizep,bsizep;
    struct timeval  t0, t1; 
    double flops;
    bmgsstencil_gpu s_gpu=bmgs_stencil_to_gpu(s);

    //cudaExtent bext;
    //  long nsmall[3]={1,1,BLOCK_X};
    //double h[3]={1.0, 1.0, 1.0};
    /*
    long offsets[s->ncoefs];
    int  offsets1[s->ncoefs];
    double coefs[s->ncoefs],coefs1[s->ncoefs];
    int3 jb;
    long ncoefs=0,ncoefs1=0;

    */
    asize=s->j[0]+s->n[0]*(s->j[1]+s->n[1]*(s->n[2]+s->j[2]));
    bsize=s->n[0]*s->n[1]*s->n[2];
    /* 
    jb.z=s->j[2];
    jb.y=s->j[1]/(s->j[2]+s->n[2]);

    for(int i = 0; i < s->ncoefs; i++){
      if (abs(s->offsets[i])<=(s->j[2]/2)){
	offsets1[ncoefs1]=s->offsets[i];
	coefs1[ncoefs1]=s->coefs[i];
	ncoefs1++;
      } else if (abs(s->offsets[i])<=(s->j[1]/2)){
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
    */
    /*
    offsets=(long *)malloc(s->ncoefs*sizeof(long));
    coefs=(double *)malloc(s->ncoefs*sizeof(double));
    memcpy(offsets,s->offsets,s->ncoefs*sizeof(long));
    memcpy(coefs,s->coefs,s->ncoefs*sizeof(double));
    
    for(int x = 0; x < s->ncoefs; x++){
	for(int y = 0; y < s->ncoefs-1; y++)
	  if(abs(offsets[y]) > abs(offsets[y+1])) {
	    long offset= offsets[y+1];
	    offsets[y+1] = offsets[y];
	    offsets[y] = offset;
	    double coef= coefs[y+1];
	    coefs[y+1] = coefs[y];
	    coefs[y] = coef;
	  }
    }
    */
    /*    fprintf(stdout,"%ld\t", ncoefs);
    for(int i = 0; i < ncoefs; ++i)
      fprintf(stdout,"(%lf %ld)\t", coefs[i], offsets[i]);
    fprintf(stdout,"\n");
    fprintf(stdout,"%ld\t", ncoefs1);
    for(int i = 0; i < ncoefs1; ++i)
      fprintf(stdout,"(%lf %ld)\t", coefs1[i], offsets1[i]);
      fprintf(stdout,"\n");*/

    /*    cudaMalloc(&(stemp.coefs),sizeof(double)*(s->ncoefs));
    cudaMemcpy(stemp.coefs,s->coefs,sizeof(double)*(s->ncoefs),cudaMemcpyHostToDevice);
    cudaMalloc(&(stemp.offsets),sizeof(double)*(s->ncoefs));
    cudaMemcpy(stemp.offsets,s->offsets,sizeof(double)*(s->ncoefs),cudaMemcpyHostToDevice);
    */
    //fprintf(stdout,"0\n");

    /*cudaMalloc(&sdev,sizeof(bmgsstencil));
    cudaMemcpy(sdev,&stemp,sizeof(bmgsstencil),cudaMemcpyHostToDevice);
    */
    //fprintf(stdout,"1\n");
    
    //    cudaMemcpyToSymbol(c_ncoefs,&(s->ncoefs),sizeof(int))

    /*
    long3 hc_n;
    long3 hc_j;    
    hc_n.x=s->n[0];    hc_n.y=s->n[1];    hc_n.z=s->n[2];
    hc_j.x=s->j[0];    hc_j.y=s->j[1];    hc_j.z=s->j[2];
    cudaMemcpyToSymbol(c_n,&hc_n,sizeof(hc_n));
    cudaMemcpyToSymbol(c_j,&hc_j,sizeof(hc_j));
    */
    //    cudaMemcpyToSymbol(c_n,s->n,sizeof(long)*3);
    //cudaMemcpyToSymbol(c_j,s->j,sizeof(long)*3);
    
    /*    cudaMemcpyToSymbol(c_offsets,offsets,sizeof(long)*ncoefs);
    cudaMemcpyToSymbol(c_coefs,coefs,sizeof(double)*ncoefs);
    cudaMemcpyToSymbol(c_offsets1,offsets1,sizeof(int)*ncoefs1);
    cudaMemcpyToSymbol(c_coefs1,coefs1,sizeof(double)*ncoefs1);*/
    
    //cudaMemcpyToSymbol(c_jb,jb,sizeof(int)*3);
    //    cudaMemcpyToSymbol(c_jb,&jb,sizeof(jb));


    cudaMalloc(&adev,sizeof(double)*asize);


    //  fprintf(stdout,"3\n");

    cudaMalloc(&bdev,sizeof(double)*bsize);

    /*
      bext=make_cudaExtent(s->n[2]*sizeof(double),s->n[1],s->n[0]);
      cudaMalloc3D(&bdev,bext);
    */
    //cudaMemset(bdev,0,sizeof(double)*bsize);
    /*  fprintf(stdout,"1111\n");

    fprintf(stdout,"pitch %zd\n",bdev.pitch);
    fprintf(stdout,"2222\n");
    */

  
    //  fprintf(stdout,"4 \n");

    //  dim3 dimBlock(MIN(BLOCK_SIZE,s->n[2]),1); 
    //dim3 dimGrid(MAX(MIN(s->n[0]/4,GRID_SIZE_MAX),GRID_SIZE_MIN),
    //	       MAX(MIN(s->n[1]/4,GRID_SIZE_MAX),GRID_SIZE_MIN)); 

    /* int gridy=s->n[1]/BLOCK_Y;
    gridy=(s->n[1]%BLOCK_Y) ? gridy+1 : gridy;

    dim3 dimBlock(BLOCK_X,BLOCK_Y); 
    dim3 dimGrid(s->n[0],gridy);*/
    //  dim3 dimGrid(GRID_SIZE, GRID_SIZE); 
    
    /*    ap=a;
    bp=b;
    adevp=adev;
    bdevp=bdev;
    asizep=asize;
    bsizep=bsize;*/
  
   
    cudaMemcpy(adev,a,sizeof(double)*asize,cudaMemcpyHostToDevice);
    gettimeofday(&t0,NULL);  
    
    //cudaMemcpy(adevp,ap,sizeof(double)*asizep,cudaMemcpyHostToDevice);
    
    
    //    bmgs_fd_cuda_kernel<<<dimGrid, dimBlock, 0>>>(sdev,adevp+(s->j[0]+s->j[1]+s->j[2])/2,bdevp,MIN(nstep0,s->n[0]-i0),s->n[1],s->n[2]);
    bmgs_fd_cuda_gpu(&s_gpu, adev,bdev);
      
    //      bmgs_fd_cuda_kernel<<<dimGrid, dimBlock, 0>>>(ncoefs,ncoefs1,adevp+(s->j[0]+s->j[1]+s->j[2])/2,bdevp);
    
    cudaThreadSynchronize(); 
    cudaError_t error = cudaGetLastError();
    
    if(error!=cudaSuccess) {
      fprintf(stderr,"ERROR: %s: %s\n", "bmgs_fd_cuda_cpu", cudaGetErrorString(error) );
      //exit(-1);
      
    } 
    gettimeofday(&t1,NULL);
    //  fprintf(stdout,"5 \n");
    cudaMemcpy(b,bdev,sizeof(double)*bsize,cudaMemcpyDeviceToHost);
    
    //cudaMemcpy(bp,bdevp,sizeof(double)*bsizep,cudaMemcpyDeviceToHost);
  

    /*
      cudaMemcpy3DParms bParms = {0};
      bParms.srcPtr=bdev;
      bParms.extent=bext;
      bParms.srcPos=make_cudaPos(0,0,0);
      bParms.dstPtr=make_cudaPitchedPtr((char*)b,s->n[2]*sizeof(double),s->n[2]*sizeof(double),s->n[1]);
      bParms.dstPos=make_cudaPos(0,0,0);
      bParms.kind=cudaMemcpyDeviceToHost;

      cudaMemcpy3D(&bParms);
    */
    //  fprintf(stdout,"6\n");

    //free(offsets);
    //free(coefs);
    /*    cudaFree(stemp.coefs);
    cudaFree(stemp.offsets);
    cudaFree(sdev);*/
    cudaFree(adev);
    cudaFree(bdev);
    //fprintf(stdout,"7\n");

    //    flops=2*s->ncoefs*bsize/(t1.tv_sec*1.0+t1.tv_usec/1000000.0-t0.tv_sec*1.0-t0.tv_usec/1000000.0); 
    flops=(t1.tv_sec*1.0+t1.tv_usec/1000000.0-t0.tv_sec*1.0-t0.tv_usec/1000000.0); 

    return flops;
  
  
  }


}





#endif
