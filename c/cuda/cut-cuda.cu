#include<cuda.h>
#include<driver_types.h>
#include<cuda_runtime_api.h>

#include <string.h>

#include "gpaw-cuda-int.h"

#ifndef CUGPAWCOMPLEX

#define BLOCK_SIZEX 32
#define BLOCK_SIZEY 8
#define XDIV 4

#endif


extern "C" {
  void Zcuda(bmgs_cut_cuda)(const Tcuda* a, const int n[3], const int c[3],
                 Tcuda* b, const int m[3],enum cudaMemcpyKind kind)
  {
    
    if (!(m[0] && m[1] && m[2])) return;
    
    cudaMemcpy3DParms myParms = {0};
    
    myParms.srcPtr=make_cudaPitchedPtr((void*)a, n[2]*sizeof(Tcuda), 
				       n[2], n[1] );
    
    myParms.dstPtr=make_cudaPitchedPtr((void*)b, m[2]*sizeof(Tcuda), 
				       m[2], m[1] );
    myParms.extent=make_cudaExtent(m[2]*sizeof(Tcuda),m[1],m[0]);
    myParms.srcPos=make_cudaPos(c[2]*sizeof(Tcuda),c[1],c[0]);
    
    myParms.kind=kind;
    gpaw_cudaSafeCall(cudaMemcpy3D(&myParms));
  }
  
}

__global__ void Zcuda(bmgs_cut_cuda_kernel)(const Tcuda* a,
					    const int3 c_sizea,
					    Tcuda* b,const int3 c_sizeb,
					    int blocks)
{
  
  int i1bl=blockIdx.y/blocks;
  int blocksi=blockIdx.y-blocks*i1bl;

  int i1tid=threadIdx.y;
  int i1=i1bl*BLOCK_SIZEY+i1tid;

  //  int i1=blockIdx.y*BLOCK_SIZEY+threadIdx.y;
  

  int i2bl=blockIdx.x/XDIV;
  int xind=blockIdx.x-XDIV*i2bl;
  int i2=i2bl*BLOCK_SIZEX+threadIdx.x;
  //  int i2=blockIdx.x*BLOCK_SIZEX+threadIdx.x;
  
  int xlen=(c_sizeb.x+XDIV-1)/XDIV;
  int xstart=xind*xlen;
  int xend=MIN(xstart+xlen,c_sizeb.x);
  
  b+=c_sizeb.x*c_sizeb.y*c_sizeb.z*blocksi;
  a+=c_sizea.x*c_sizea.y*c_sizea.z*blocksi;

  b+=i2+i1*c_sizeb.z+xstart*c_sizeb.y*c_sizeb.z;
  a+=i2+i1*c_sizea.z+xstart*c_sizea.y*c_sizea.z;
  for (int i0=xstart;i0<xend;i0++) {	
    if ((i2<c_sizeb.z)&&(i1<c_sizeb.y)){
      b[0] = a[0];
    }
    b+=c_sizeb.y*c_sizeb.z;
    a+=c_sizea.y*c_sizea.z;        
  }
}


extern "C" {
  
  void Zcuda(bmgs_cut_cuda_gpu)(const Tcuda* a, const int sizea[3],
				const int starta[3],
				Tcuda* b, const int sizeb[3],int blocks)
  {
    if (!(sizea[0] && sizea[1] && sizea[2])) return;    

    int3 hc_sizea,hc_sizeb;    
    hc_sizea.x=sizea[0];    hc_sizea.y=sizea[1];    hc_sizea.z=sizea[2];
    hc_sizeb.x=sizeb[0];    hc_sizeb.y=sizeb[1];    hc_sizeb.z=sizeb[2];
    
    int gridy=blocks*(sizeb[1]+BLOCK_SIZEY-1)/BLOCK_SIZEY;
    
    int gridx=XDIV*((sizeb[2]+BLOCK_SIZEX-1)/BLOCK_SIZEX);
    
    
    dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY); 
    dim3 dimGrid(gridx,gridy);    

    a+=starta[2]+(starta[1]+starta[0]*hc_sizea.y)*hc_sizea.z;
    Zcuda(bmgs_cut_cuda_kernel)<<<dimGrid, dimBlock, 0>>>((Tcuda*)a,hc_sizea,(Tcuda*)b,hc_sizeb,blocks);
    
    gpaw_cudaSafeCall(cudaGetLastError());
    
  }
}

#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "cut-cuda.cu"
#endif
