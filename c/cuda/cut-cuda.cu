#include<cuda.h>
#include<driver_types.h>
#include<cuda_runtime_api.h>

#include <string.h>

#include "gpaw-cuda-int.h"

#ifdef DEBUG_CUDA
#define DEBUG_CUDA_CUT
#endif //DEBUG_CUDA

#ifdef DEBUG_CUDA_CUT
extern "C" {
#include <complex.h>
  typedef double complex double_complex;
#define GPAW_MALLOC(T, n) (T*)(malloc((n) * sizeof(T)))
  void bmgs_cut(const double* a, const int n[3], const int c[3],
		double* b, const int m[3]);
  void bmgs_cutz(const double_complex* a, const int n[3],
		 const int c[3],
		 double_complex* b, const int m[3]);
}
#endif //DEBUG_CUDA_CUT


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
  
  b+=i2+(i1+(xind+blocksi*c_sizeb.x)*c_sizeb.y)*c_sizeb.z;
  a+=i2+(i1+(xind+blocksi*c_sizea.x)*c_sizea.y)*c_sizea.z;

  while (xind<c_sizeb.x){
    if ((i2<c_sizeb.z)&&(i1<c_sizeb.y)){
#ifndef CUGPAWCOMPLEX
      b[0] = a[0];
#else
      b[0] = MULTT(phase,a[0]);
#endif
    }
    b+=xdiv*c_sizeb.y*c_sizeb.z;
    a+=xdiv*c_sizea.y*c_sizea.z;
    xind+=xdiv;
  }
}



extern "C" {
  
  void Zcuda(bmgs_cut_cuda_gpu)(const Tcuda* a, const int sizea[3],
				const int starta[3],
				Tcuda* b, const int sizeb[3],
#ifdef CUGPAWCOMPLEX
				cuDoubleComplex phase, 
#endif
				int blocks,cudaStream_t stream)
  {
    if (!(sizea[0] && sizea[1] && sizea[2])) return;    

    int3 hc_sizea,hc_sizeb;    
    hc_sizea.x=sizea[0];    hc_sizea.y=sizea[1];    hc_sizea.z=sizea[2];
    hc_sizeb.x=sizeb[0];    hc_sizeb.y=sizeb[1];    hc_sizeb.z=sizeb[2];

        
#ifdef DEBUG_CUDA_CUT
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
    const Tcuda* a2=a;

    GPAW_CUDAMEMCPY(a_cpu,a,double, ng*blocks, cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(b_cpu,b,double, ng2*blocks, cudaMemcpyDeviceToHost);
#endif //DEBUG_CUDA_CUT

    int blockx=MIN(nextPow2(hc_sizeb.z),BLOCK_MAX);
    int blocky=MIN(MIN(nextPow2(hc_sizeb.y),BLOCK_TOTALMAX/blockx),BLOCK_MAX); 
    dim3 dimBlock(blockx,blocky);
    int gridx=((hc_sizeb.z+dimBlock.x-1)/dimBlock.x);
    int xdiv=MAX(1,MIN(hc_sizeb.x,GRID_MAX/gridx));
    int gridy=blocks*((hc_sizeb.y+dimBlock.y-1)/dimBlock.y);    

    gridx=xdiv*gridx;
    dim3 dimGrid(gridx,gridy);    
    a+=starta[2]+(starta[1]+starta[0]*hc_sizea.y)*hc_sizea.z;      

    Zcuda(bmgs_cut_cuda_kernel)<<<dimGrid, dimBlock, 0, stream>>>
      ((Tcuda*)a,hc_sizea,(Tcuda*)b,hc_sizeb,
#ifdef CUGPAWCOMPLEX
       phase,
#endif
       blocks,xdiv);
    gpaw_cudaSafeCall(cudaGetLastError());

    
#ifdef DEBUG_CUDA_CUT
    for (int m = 0; m < blocks; m++){            
#ifndef CUGPAWCOMPLEX      
      bmgs_cut(a_cpu + m * ng, sizea, starta, b_cpu + m * ng2,
		 sizeb);
#else
      bmgs_cutz((const double_complex*)(a_cpu + m * ng), sizea, starta,
		  (double_complex*)(b_cpu + m * ng2),
		  sizeb);
#endif //CUGPAWCOMPLEX
    }
    cudaDeviceSynchronize();
    GPAW_CUDAMEMCPY(a_cpu2,a2,double, ng*blocks, cudaMemcpyDeviceToHost);
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
      fprintf(stderr,"Debug cuda cut errors: a %g b %g\n",a_err,b_err);
    }
    free(a_cpu);
    free(b_cpu);
    free(a_cpu2);
    free(b_cpu2);
#endif //DEBUG_CUDA_CUT
    
  }
}

#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "cut-cuda.cu"
#endif
