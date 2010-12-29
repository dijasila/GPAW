#include<cuda.h>
#include<driver_types.h>
#include<cuda_runtime_api.h>

#include <stdio.h>
#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include "gpaw-cuda-int.h"

#ifndef CUGPAWCOMPLEX

#define BLOCK_SIZEX 16
#define BLOCK_SIZEY 8

#endif


extern "C" {


  void Zcuda(bmgs_paste_cuda)(const Tcuda *a, const int sizea[3],
			      Tcuda *b, const int sizeb[3], 
			      const int startb[3],enum cudaMemcpyKind kind)
  {

    cudaMemcpy3DParms myParms = {0};
  
    myParms.srcPtr=make_cudaPitchedPtr((void*)a, sizea[2]*sizeof(Tcuda), sizea[2], sizea[1] );

    myParms.dstPtr=make_cudaPitchedPtr((void*)b, sizeb[2]*sizeof(Tcuda), sizeb[2], sizeb[1] );
    myParms.extent=make_cudaExtent(sizea[2]*sizeof(Tcuda),sizea[1],sizea[0]);
    myParms.dstPos=make_cudaPos(startb[2]*sizeof(Tcuda),startb[1],startb[0]);
  
    myParms.kind=kind;
    gpaw_cudaSafeCall(cudaMemcpy3D(&myParms));
  }
}  
  
	
__global__ void Zcuda(bmgs_paste_cuda_kernel)(const Tcuda* a,
					      const int4 c_sizea,
					      Tcuda* b,const int4 c_sizeb)
{

  
    int i1=blockIdx.y*BLOCK_SIZEY+threadIdx.y;

    int i2=blockIdx.x*BLOCK_SIZEX+threadIdx.x;

    b+=i2+i1*c_sizeb.z;
    a+=i2+i1*c_sizea.z;
    if ((i1<c_sizea.y)&&(i2<c_sizea.z)){
      for (int i0=0;i0<c_sizea.x;i0++) {	
	b[0]=	a[0];
	b+=c_sizeb.y*c_sizeb.z;
	a+=c_sizea.y*c_sizea.z;
      }
    
    }
  }

  extern "C" {

    void Zcuda(bmgs_paste_cuda_gpu)(const Tcuda* a, const int sizea[3],
			     Tcuda* b, const int sizeb[3], const int startb[3])
    {
    
      int4 hc_sizea,hc_sizeb;    
      hc_sizea.x=sizea[0];    hc_sizea.y=sizea[1];    hc_sizea.z=sizea[2];
      hc_sizeb.x=sizeb[0];    hc_sizeb.y=sizeb[1];    hc_sizeb.z=sizeb[2];

      int gridy=(sizea[1]+BLOCK_SIZEY-1)/BLOCK_SIZEY;

      int gridx=(sizea[2]+BLOCK_SIZEX-1)/BLOCK_SIZEX;
      

      dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY); 
      dim3 dimGrid(gridx,gridy);    


      b+=startb[2]+(startb[1]+startb[0]*hc_sizeb.y)*hc_sizeb.z;    
      Zcuda(bmgs_paste_cuda_kernel)<<<dimGrid, dimBlock, 0>>>((Tcuda*)a,hc_sizea,(Tcuda*)b,hc_sizeb);

      gpaw_cudaSafeCall(cudaGetLastError());
      
    }
  }

#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "paste-cuda.cu"

extern "C" {
  double bmgs_paste_cuda_cpu(const double* a, const int sizea[3],
			     double* b, const int sizeb[3], const int startb[3])
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
			bdev, sizeb, startb);
    
    
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
