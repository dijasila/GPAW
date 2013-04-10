#include<cuda.h>
#include<driver_types.h>
#include<cuda_runtime_api.h>

#include <stdio.h>
#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include "gpaw-cuda-int.h"

#ifdef DEBUG_CUDA
#define DEBUG_CUDA_PASTE   
#endif //DEBUG_CUDA

#ifdef DEBUG_CUDA_PASTE
extern "C" {
#include <complex.h>
  typedef double complex double_complex;
#define GPAW_MALLOC(T, n) (T*)(malloc((n) * sizeof(T)))
  void bmgs_paste(const double* a, const int n[3],
		  double* b, const int m[3], const int c[3]);
  void bmgs_pastez(const double_complex* a, const int n[3],
		   double_complex* b, const int m[3],
		   const int c[3]);
}
#endif //DEBUG_CUDA_PASTE

#ifndef CUGPAWCOMPLEX

#define BLOCK_SIZEX 32
#define BLOCK_SIZEY 16
#define BLOCK_MAX 32
#define GRID_MAX 65535
#define BLOCK_TOTALMAX 512
#define XDIV 4

static unsigned int nextPow2( unsigned int x ) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

#endif



extern "C" {


  void Zcuda(bmgs_paste_cuda)(const Tcuda *a, const int sizea[3],
			      Tcuda *b, const int sizeb[3], 
			      const int startb[3],int blocks,enum cudaMemcpyKind kind,
			      cudaStream_t stream)
  {

    if (!(sizea[0] && sizea[2] && sizea[3])) return;

    int ng2 = sizeb[0] * sizeb[1] * sizeb[2];
    int ng = sizea[0] * sizea[1] * sizea[2];

    
    for (int m = 0; m < blocks; m++){            
      cudaMemcpy3DParms myParms = {0};

      myParms.srcPtr=make_cudaPitchedPtr((void*)(a+ng*m), sizea[2]*sizeof(Tcuda), 
					 sizea[2], sizea[1] );
      
      myParms.srcPos=make_cudaPos(0*sizeof(Tcuda),0,0);
      myParms.dstPtr=make_cudaPitchedPtr((void*)(b+ng2*m), sizeb[2]*sizeof(Tcuda), 
					 sizeb[2], sizeb[1] );
      
      myParms.extent=make_cudaExtent(sizea[2]*sizeof(Tcuda),sizea[1],sizea[0]);
      myParms.dstPos=make_cudaPos(startb[2]*sizeof(Tcuda),startb[1],startb[0]);
      
      myParms.kind=kind;


      gpaw_cudaSafeCall(cudaMemcpy3DAsync(&myParms,stream));
    }
  }
}  


__global__ void Zcuda(bmgs_paste_cuda_kernel)(const double* a,
					      const int3 c_sizea,
					      double* b,const int3 c_sizeb,
					      int blocks,int xdiv)
{
  int xx=gridDim.x/xdiv;
  int yy=gridDim.y/blocks;
  
  int blocksi=blockIdx.y/yy;
  
  int i1=(blockIdx.y-blocksi*yy)*blockDim.y+threadIdx.y;

  int xind=blockIdx.x/xx;
  
  int i2=(blockIdx.x-xind*xx)*blockDim.x+threadIdx.x;
  
  b+=i2+(i1+(xind+blocksi*c_sizeb.x)*c_sizeb.y)*c_sizeb.z;
  a+=i2+(i1+(xind+blocksi*c_sizea.x)*c_sizea.y)*c_sizea.z;

  while (xind<c_sizea.x){
    if ((i2<c_sizea.z)&&(i1<c_sizea.y)){
      b[0] = a[0];
    }
    b+=xdiv*c_sizeb.y*c_sizeb.z;
    a+=xdiv*c_sizea.y*c_sizea.z;
    xind+=xdiv;
  }
}

__global__ void Zcuda(bmgs_paste_zero_cuda_kernel)(const Tcuda* a,
						   const int3 c_sizea,
						   Tcuda* b,
						   const int3 c_sizeb,
						   const int3 c_startb,
						   const int3 c_blocks_bc,
						   int blocks)
{
  int xx=gridDim.x/XDIV;
  int yy=gridDim.y/blocks;

  
  int blocksi=blockIdx.y/yy;
  int i1bl=blockIdx.y-blocksi*yy;

  int i1tid=threadIdx.y;
  int i1=i1bl*BLOCK_SIZEY+i1tid;

  int i2tid=threadIdx.x;
  
  int xind=blockIdx.x/xx;
  int i2bl=blockIdx.x-xind*xx;
  
  int i2=i2bl*BLOCK_SIZEX+i2tid;

  int xlen=(c_sizea.x+XDIV-1)/XDIV;
  int xstart=xind*xlen;
  int xend=MIN(xstart+xlen,c_sizea.x);
  
  
  b+=c_sizeb.x*c_sizeb.y*c_sizeb.z*blocksi;
  a+=c_sizea.x*c_sizea.y*c_sizea.z*blocksi;
  
  if (xind==0)  {
    Tcuda *bb=b+i2+i1*c_sizeb.z;
#pragma unroll 3
    for (int i0=0;i0<c_startb.x;i0++) {
      if ((i2<c_sizeb.z) && (i1<c_sizeb.y)) {
	bb[0]=MAKED(0);
      }
      bb+=c_sizeb.y*c_sizeb.z;
      
    }
  }
  if (xind==XDIV-1)   {
    Tcuda *bb=b+(c_startb.x+c_sizea.x)*c_sizeb.y*c_sizeb.z+i2+i1*c_sizeb.z;
#pragma unroll 3
    for (int i0=c_startb.x+c_sizea.x;i0<c_sizeb.x;i0++) {
      if ((i2<c_sizeb.z) && (i1<c_sizeb.y)) {
	bb[0]=MAKED(0);
      }
      bb+=c_sizeb.y*c_sizeb.z;
    }
  }  

  int i1blbc=gridDim.y/blocks-i1bl-1;  
  int i2blbc=gridDim.x/XDIV-i2bl-1;

  if ( i1blbc<c_blocks_bc.y || i2blbc<c_blocks_bc.z) {

    int i1bc=i1blbc*BLOCK_SIZEY+i1tid;
    int i2bc=i2blbc*BLOCK_SIZEX+i2tid;
    
    b+=(c_startb.x+xstart)*c_sizeb.y*c_sizeb.z;
    for (int i0=xstart;i0<xend;i0++) {	      
      if ((i1bc<c_startb.y) && (i2<c_sizeb.z)){
	b[i2+i1bc*c_sizeb.z]=MAKED(0);
      }
      if ((i1bc+c_sizea.y+c_startb.y<c_sizeb.y) && (i2<c_sizeb.z)){
	b[i2+i1bc*c_sizeb.z+(c_sizea.y+c_startb.y)*c_sizeb.z]=MAKED(0);
      }
      if ((i2bc<c_startb.z) && (i1<c_sizeb.y)){
	b[i2bc+i1*c_sizeb.z]=MAKED(0);
      }
      if ((i2bc+c_sizea.z+c_startb.z<c_sizeb.z) && (i1<c_sizeb.y)){
	b[i2bc+i1*c_sizeb.z+c_sizea.z+c_startb.z]=MAKED(0);
      }
      b+=c_sizeb.y*c_sizeb.z;
    }    
  }else{
    
    b+=c_startb.z+(c_startb.y+c_startb.x*c_sizeb.y)*c_sizeb.z;
    
    b+=i2+i1*c_sizeb.z+xstart*c_sizeb.y*c_sizeb.z;
    a+=i2+i1*c_sizea.z+xstart*c_sizea.y*c_sizea.z;
    for (int i0=xstart;i0<xend;i0++) {	
      if ((i2<c_sizea.z)&&(i1<c_sizea.y)){
	b[0] = a[0];
      }
      b+=c_sizeb.y*c_sizeb.z;
      a+=c_sizea.y*c_sizea.z;        
    }
  }
}


extern "C" {

  
  void Zcuda(bmgs_paste_cuda_gpu)(const Tcuda* a, const int sizea[3],
				  Tcuda* b, const int sizeb[3], 
				  const int startb[3],
				  int blocks,cudaStream_t stream)
  {
    if (!(sizea[0] && sizea[1] && sizea[2])) return;    


      
    int3 hc_sizea,hc_sizeb;
    hc_sizea.x=sizea[0];    hc_sizea.y=sizea[1];    hc_sizea.z=sizea[2]*sizeof(Tcuda)/sizeof(double);
    hc_sizeb.x=sizeb[0];    hc_sizeb.y=sizeb[1];    hc_sizeb.z=sizeb[2]*sizeof(Tcuda)/sizeof(double);

    
#ifdef DEBUG_CUDA_PASTE
#ifndef CUGPAWCOMPLEX      
    int ng2 = sizeb[0] * sizeb[1] * sizeb[2];
    int ng = sizea[0] * sizea[1] * sizea[2];
#else
    int ng2 = sizeb[0] * sizeb[1] * sizeb[2] * 2;
    int ng = sizea[0] * sizea[1] * sizea[2] * 2;
#endif //CUGPAWCOMPLEX      
    double* b_cpu=GPAW_MALLOC(double, ng2*blocks);
    double* a_cpu=GPAW_MALLOC(double, ng*blocks);
    double* b_cpu2=GPAW_MALLOC(double, ng2*blocks);
    double* a_cpu2=GPAW_MALLOC(double, ng*blocks);
    Tcuda* b2=b;

    GPAW_CUDAMEMCPY(a_cpu,a,double, ng*blocks, cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(b_cpu,b,double, ng2*blocks, cudaMemcpyDeviceToHost);
#endif //DEBUG_CUDA_PASTE

    int blockx=MIN(nextPow2(hc_sizea.z),BLOCK_MAX);
    int blocky=MIN(MIN(nextPow2(hc_sizea.y),BLOCK_TOTALMAX/blockx),BLOCK_MAX); 
    dim3 dimBlock(blockx,blocky);
    int gridx=((hc_sizea.z+dimBlock.x-1)/dimBlock.x);
    int xdiv=MAX(1,MIN(hc_sizea.x,GRID_MAX/gridx));
    int gridy=blocks*((hc_sizea.y+dimBlock.y-1)/dimBlock.y);    

    gridx=xdiv*gridx;
    dim3 dimGrid(gridx,gridy);    
    b+=startb[2]+(startb[1]+startb[0]*sizeb[1])*sizeb[2];      
    Zcuda(bmgs_paste_cuda_kernel)<<<dimGrid, dimBlock, 0, stream>>>
      ((double*)a,hc_sizea,(double*)b,hc_sizeb,blocks,xdiv);
    gpaw_cudaSafeCall(cudaGetLastError());

#ifdef DEBUG_CUDA_PASTE
    for (int m = 0; m < blocks; m++){            
#ifndef CUGPAWCOMPLEX      
      bmgs_paste(a_cpu + m * ng, sizea, b_cpu + m * ng2,
		 sizeb, startb);
#else
      bmgs_pastez((const double_complex*)(a_cpu + m * ng), sizea,
		  (double_complex*)(b_cpu + m * ng2),
		  sizeb, startb);
#endif //CUGPAWCOMPLEX
    }
    cudaDeviceSynchronize();
    GPAW_CUDAMEMCPY(a_cpu2,a,double, ng*blocks, cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(b_cpu2,b2,double, ng2*blocks, cudaMemcpyDeviceToHost);
    double a_err=0;
    double b_err=0;
    for (int i=0;i<ng2*blocks;i++) {      
      b_err=MAX(b_err,fabs(b_cpu[i]-b_cpu2[i]));
      if (i<ng*blocks){
	a_err=MAX(a_err,fabs(a_cpu[i]-a_cpu2[i]));
      }
    }
    if ((b_err>GPAW_CUDA_ABS_TOL_EXCT) || (a_err>GPAW_CUDA_ABS_TOL_EXCT)){
      fprintf(stderr,"Debug cuda paste errors: a %g b %g\n",a_err,b_err); fflush(stderr);
    }
    free(a_cpu);
    free(b_cpu);
    free(a_cpu2);
    free(b_cpu2);
#endif //DEBUG_CUDA_PASTE
    
  }
  

  void Zcuda(bmgs_paste_zero_cuda_gpu)(const Tcuda* a, const int sizea[3],
				       Tcuda* b, const int sizeb[3], 
				       const int startb[3],
				       int blocks,cudaStream_t stream)
  {
    if (!(sizea[0] && sizea[1] && sizea[2])) return;
    
    int3 hc_sizea,hc_sizeb,hc_startb;    
    hc_sizea.x=sizea[0];    hc_sizea.y=sizea[1];    hc_sizea.z=sizea[2];
    hc_sizeb.x=sizeb[0];    hc_sizeb.y=sizeb[1];    hc_sizeb.z=sizeb[2];
    hc_startb.x=startb[0];    hc_startb.y=startb[1];    hc_startb.z=startb[2];

    int3 bc_blocks;

#ifdef DEBUG_CUDA_PASTE
#ifndef CUGPAWCOMPLEX      
    int ng2 = sizeb[0] * sizeb[1] * sizeb[2];
    int ng = sizea[0] * sizea[1] * sizea[2];
#else
    int ng2 = sizeb[0] * sizeb[1] * sizeb[2] * 2;
    int ng = sizea[0] * sizea[1] * sizea[2] * 2;
#endif //CUGPAWCOMPLEX      
    double* b_cpu=GPAW_MALLOC(double, ng2*blocks);
    double* a_cpu=GPAW_MALLOC(double, ng*blocks);
    double* b_cpu2=GPAW_MALLOC(double, ng2*blocks);
    double* a_cpu2=GPAW_MALLOC(double, ng*blocks);

    GPAW_CUDAMEMCPY(a_cpu,a,double, ng*blocks, cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(b_cpu,b,double, ng2*blocks, cudaMemcpyDeviceToHost);
#endif //DEBUG_CUDA_PASTE


    bc_blocks.y=hc_sizeb.y-hc_sizea.y>0 ? 
      MAX((hc_sizeb.y-hc_sizea.y+BLOCK_SIZEY-1)/BLOCK_SIZEY,1) : 0;
    bc_blocks.z=hc_sizeb.z-hc_sizea.z>0 ?
      MAX((hc_sizeb.z-hc_sizea.z+BLOCK_SIZEX-1)/BLOCK_SIZEX,1) : 0;
    
    int gridy=blocks*((sizeb[1]+BLOCK_SIZEY-1)/BLOCK_SIZEY+bc_blocks.y);
    
    int gridx=XDIV*((sizeb[2]+BLOCK_SIZEX-1)/BLOCK_SIZEX+bc_blocks.z);
    

    dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY); 
    dim3 dimGrid(gridx,gridy);    
    
    //    b+=startb[2]+(startb[1]+startb[0]*hc_sizeb.y)*hc_sizeb.z;
    Zcuda(bmgs_paste_zero_cuda_kernel)<<<dimGrid, dimBlock, 0, stream>>>
      ((Tcuda*)a,hc_sizea,(Tcuda*)b,hc_sizeb,hc_startb,bc_blocks,blocks);
    
    gpaw_cudaSafeCall(cudaGetLastError());
    
#ifdef DEBUG_CUDA_PASTE    
    for (int m = 0; m < blocks; m++){            
      memset(b_cpu + m * ng2, 0, ng2 * sizeof(double));
#ifndef CUGPAWCOMPLEX      
      bmgs_paste(a_cpu + m * ng, sizea, b_cpu + m * ng2,
		 sizeb, startb);
#else
      bmgs_pastez((const double_complex*)(a_cpu + m * ng), sizea,
		  (double_complex*)(b_cpu + m * ng2),
		  sizeb, startb);
#endif //CUGPAWCOMPLEX
    }
    cudaDeviceSynchronize();
    GPAW_CUDAMEMCPY(a_cpu2,a,double, ng*blocks, cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(b_cpu2,b,double, ng2*blocks, cudaMemcpyDeviceToHost);
    double a_err=0;
    double b_err=0;
    for (int i=0;i<ng2*blocks;i++) {      
      b_err=MAX(b_err,fabs(b_cpu[i]-b_cpu2[i]));
      if (i<ng*blocks){
	a_err=MAX(a_err,fabs(a_cpu[i]-a_cpu2[i]));
      }
    }
    if ((b_err>GPAW_CUDA_ABS_TOL_EXCT) || (a_err>GPAW_CUDA_ABS_TOL_EXCT)){
      fprintf(stderr,"Debug cuda paste_zero errors: a %g b %g\n",a_err,b_err);
    }
    free(a_cpu);
    free(b_cpu);
    free(a_cpu2);
    free(b_cpu2);
#endif //DEBUG_CUDA_PASTE
    
  }
}

#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "paste-cuda.cu"

extern "C" {
  double bmgs_paste_cuda_cpu(const double* a, const int sizea[3],
			     double* b, const int sizeb[3], 
			     const int startb[3])
  {
    double *adev,*bdev;
    
    struct timeval  t0, t1; 
    double flops;
    int asize=sizea[0]*sizea[1]*sizea[2];
    int bsize=sizeb[0]*sizeb[1]*sizeb[2];
    
    
    
    gpaw_cudaSafeCall(cudaMalloc(&adev,sizeof(double)*asize));
    gpaw_cudaSafeCall(cudaMalloc(&bdev,sizeof(double)*bsize));
    gpaw_cudaSafeCall(cudaMemcpy(adev,a,sizeof(double)*asize,
				 cudaMemcpyHostToDevice));
    
    gettimeofday(&t0,NULL);  
    bmgs_paste_cuda_gpu(adev, sizea,
			bdev, sizeb, startb,1,0);
    
    
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


  double bmgs_paste_zero_cuda_cpu(const double* a, const int sizea[3],
				   double* b, const int sizeb[3], 
				   const int startb[3])
  {
    double *adev,*bdev;
    
    struct timeval  t0, t1; 
    double flops;
    int asize=sizea[0]*sizea[1]*sizea[2];
    int bsize=sizeb[0]*sizeb[1]*sizeb[2];
    
    
    
    gpaw_cudaSafeCall(cudaMalloc(&adev,sizeof(double)*asize));
    gpaw_cudaSafeCall(cudaMalloc(&bdev,sizeof(double)*bsize));
    gpaw_cudaSafeCall(cudaMemcpy(adev,a,sizeof(double)*asize,
				 cudaMemcpyHostToDevice));
    
    gettimeofday(&t0,NULL);  
    bmgs_paste_zero_cuda_gpu(adev, sizea,
			     bdev, sizeb, startb,1,0);
    
    
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
