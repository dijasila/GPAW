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

#define BLOCK_MAX 32
#define GRID_MAX 65535
#define BLOCK_TOTALMAX 256

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
						   Tcuda* b,const int3 c_sizeb,
#ifdef CUGPAWCOMPLEX
						   cuDoubleComplex phase,
#endif		
						   int blocks,int xdiv)
{
  int xx=gridDim.x/xdiv;
  int yy=gridDim.y/blocks;
  
  int blocksi=blockIdx.y/yy;
  
  int i1=(blockIdx.y-blocksi*yy)*blockDim.y+threadIdx.y;

  int xind=blockIdx.x/xx;
  
  int i2=(blockIdx.x-xind*xx)*blockDim.x+threadIdx.x;
  
  b+=i2+(i1+(xind+blocksi*c_sizea.x)*c_sizea.y)*c_sizea.z;
  a+=i2+(i1+(xind+blocksi*c_sizea.x)*c_sizea.y)*c_sizea.z;

  while (xind<c_sizeb.x){
    if ((i2<c_sizeb.z)&&(i1<c_sizeb.y)){
#ifndef CUGPAWCOMPLEX
      b[0] = a[0];
#else
      b[0] = MULTT(phase,a[0]);
#endif
    }
    b+=xdiv*c_sizea.y*c_sizea.z;
    a+=xdiv*c_sizea.y*c_sizea.z;
    xind+=xdiv;
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

    int blockx=MIN(nextPow2(hc_size.z),BLOCK_MAX);
    int blocky=MIN(MIN(nextPow2(hc_size.y),BLOCK_TOTALMAX/blockx),BLOCK_MAX); 
    dim3 dimBlock(blockx,blocky);
    int gridx=((hc_size.z+dimBlock.x-1)/dimBlock.x);
    int xdiv=MAX(1,MIN(hc_size.x,GRID_MAX/gridx));
    int gridy=blocks*((hc_size.y+dimBlock.y-1)/dimBlock.y);    
    
    gridx=xdiv*gridx;
    dim3 dimGrid(gridx,gridy);    
    Tcuda *b=a+start2[2]+(start2[1]+start2[0]*hc_sizea.y)*hc_sizea.z;
    a+=start1[2]+(start1[1]+start1[0]*hc_sizea.y)*hc_sizea.z;

    Zcuda(bmgs_translate_cuda_kernel)<<<dimGrid, dimBlock, 0, stream>>>
      ((Tcuda*)a,hc_sizea,(Tcuda*)b,hc_size,
#ifdef CUGPAWCOMPLEX
       phase,
#endif
       blocks,xdiv);
    gpaw_cudaSafeCall(cudaGetLastError());
    

  }
}


#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "translate-cuda.cu"
#endif
