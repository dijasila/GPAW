#include<cuda.h>
#include<cublas.h>
#include<driver_types.h>
#include<cuda_runtime_api.h>

#include <stdio.h>
#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include "gpaw-cuda-int.h"


#ifndef CUGPAWCOMPLEX

#define BLOCK_SIZEX 32
#define BLOCK_SIZEY 8
#define XDIV 4

#endif


extern "C" {

  void Zcuda(bmgs_translate_cuda)(Tcuda* a, const int sizea[3], 
				  const int size[3],
				  const int start1[3], const int start2[3],
#ifdef CUGPAWCOMPLEX
				  cuDoubleComplex phase,
#endif
				  enum cudaMemcpyKind kind)
  {		

    if (!(size[0] && size[1] && size[2])) return;
    
    cudaMemcpy3DParms myParms = {0};
    myParms.srcPtr=make_cudaPitchedPtr((void*)a, sizea[2]*sizeof(Tcuda), 
				       sizea[2], sizea[1] );
    
    myParms.dstPtr=make_cudaPitchedPtr((void*)a, sizea[2]*sizeof(Tcuda), 
				       sizea[2], sizea[1] );

    myParms.srcPos=make_cudaPos(start1[2]*sizeof(Tcuda),start1[1],start1[0]);
    myParms.dstPos=make_cudaPos(start2[2]*sizeof(Tcuda),start2[1],start2[0]);

    myParms.extent=make_cudaExtent(size[2]*sizeof(Tcuda),size[1],size[0]);    
    myParms.kind=kind;
    gpaw_cudaSafeCall(cudaMemcpy3D(&myParms));

#ifdef CUGPAWCOMPLEX
    Tcuda* d = a + start2[2] + (start2[1] + start2[0] * sizea[1]) * sizea[2];

    for (int i0 = 0; i0 < size[0]; i0++)
      {
	for (int i1 = 0; i1 < size[1]; i1++)
	  {
	    cublasZscal(size[2],phase,d,1);
	    gpaw_cublasSafeCall(cublasGetError());
	    d += sizea[2];
	  }
	d += sizea[2] * (sizea[1] - size[1]);
      }
#endif
  }

}


__global__ void Zcuda(bmgs_translate_cuda_kernel)(const Tcuda* a,
					   const int3 c_sizea,
					   Tcuda* b,const int3 c_size,
#ifdef CUGPAWCOMPLEX
					   cuDoubleComplex phase,
#endif
					   int blocks)

{
  int xx=gridDim.x/XDIV;
  int yy=gridDim.y/blocks;

  
  int blocksi=blockIdx.y/yy;
  int i1bl=blockIdx.y-yy*blocksi;

  int i1tid=threadIdx.y;
  int i1=i1bl*BLOCK_SIZEY+i1tid;
  
  int xind=blockIdx.x/xx;
  int i2bl=blockIdx.x-xind*xx;

  int i2=i2bl*BLOCK_SIZEX+threadIdx.x;
  
  int xlen=(c_size.x+XDIV-1)/XDIV;
  int xstart=xind*xlen;
  int xend=MIN(xstart+xlen,c_size.x);
  
  b+=c_sizea.x*c_sizea.y*c_sizea.z*blocksi;
  a+=c_sizea.x*c_sizea.y*c_sizea.z*blocksi;


  b+=i2+i1*c_sizea.z+xstart*c_sizea.y*c_sizea.z;
  a+=i2+i1*c_sizea.z+xstart*c_sizea.y*c_sizea.z;
  for (int i0=xstart;i0<xend;i0++) {	
    if ((i2<c_size.z)&&(i1<c_size.y)){
#ifndef CUGPAWCOMPLEX
      b[0] = a[0];
#else
      b[0] = MULTT(phase,a[0]);
#endif
    }
    b+=c_sizea.y*c_sizea.z;
    a+=c_sizea.y*c_sizea.z;        
  }
}

extern "C" {


  void Zcuda(bmgs_translate_cuda_gpu)(Tcuda* a, const int sizea[3], 
				      const int size[3],
				      const int start1[3], const int start2[3],
#ifdef CUGPAWCOMPLEX
				      cuDoubleComplex phase, 
#endif
				      int blocks,cudaStream_t stream)    
  {
    if (!(size[0] && size[1] && size[2])) return;
    
    int3 hc_sizea,hc_size;    
    hc_sizea.x=sizea[0];    hc_sizea.y=sizea[1];    hc_sizea.z=sizea[2];
    hc_size.x=size[0];    hc_size.y=size[1];    hc_size.z=size[2];
    
    int gridy=blocks*((size[1]+BLOCK_SIZEY-1)/BLOCK_SIZEY);
    
    int gridx=XDIV*((size[2]+BLOCK_SIZEX-1)/BLOCK_SIZEX);
    
    
    dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY); 
    dim3 dimGrid(gridx,gridy);    
    gpaw_cudaSafeCall(cudaGetLastError());
    
    Tcuda *b=a+start2[2]+(start2[1]+start2[0]*hc_sizea.y)*hc_sizea.z;
    a+=start1[2]+(start1[1]+start1[0]*hc_sizea.y)*hc_sizea.z;
    

    Zcuda(bmgs_translate_cuda_kernel)<<<dimGrid, dimBlock, 0, stream>>>
      ((Tcuda*)a,hc_sizea,(Tcuda*)b,hc_size,
#ifdef CUGPAWCOMPLEX
       phase,
#endif    
       blocks);

    gpaw_cudaSafeCall(cudaGetLastError());
    
  }
}


#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "translate-cuda.cu"
#endif
