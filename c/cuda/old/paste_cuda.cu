#include<cuda.h>
#include<driver_types.h>
#include<cuda_runtime_api.h>


/*
void Z(bmgs_paste_cuda)(const T* a, const int sizea[3],
			T* b, const int sizeb[3], const int startb[3],enum cudaMemcpyKind kind)
{
  b += startb[2] + (startb[1] + startb[0] * sizeb[1]) * sizeb[2];
  for (int i0 = 0; i0 < sizea[0]; i0++)
    {
      for (int i1 = 0; i1 < sizea[1]; i1++)
	{
	  cudaMemcpyAsync(b, a, sizea[2] * sizeof(T),kind,0);
	  a += sizea[2];
	  b += sizeb[2];
	}
      b += sizeb[2] * (sizeb[1] - sizea[1]);
    }
  cudaThreadSynchronize();  
}
*/
#define BLOCK_SIZEX 16
#define BLOCK_SIZEY 4
#define MAXCOEFS  24
#define MAXJ      8

#define ACACHE_SIZEX  (BLOCK_SIZEX+MAXJ)    
#define ACACHE_SIZEY  (BLOCK_SIZEY+MAXJ)

__constant__ long3 c_n;
__constant__ long3 c_j;
__constant__ long c_offsets[MAXCOEFS];
__constant__ double c_coefs[MAXCOEFS];
__constant__ int c_offsets1[MAXCOEFS];
__constant__ double c_coefs1[MAXCOEFS];
//__constant__ int c_jb[3];
__constant__ int3 c_jb;
//__constant__ int  c_ncoefs;
	
	
__global__ void bmgs_paste_cuda_kernel(const double* a, const int sizea[3],
			double* b, const int sizeb[3], const int startb[3])
{

  
  if (blockIdx.x>=sizea[0]) return;  
  int i1=blockIdx.y*blockDim.y;
  
  //if (i1>=sizea[1]) return;  
  if (i1+threadIdx.y>=sizea[1]) return;

  //__shared__ double acache[ACACHE_SIZEY][ACACHE_SIZEX];
  int i2;
  
  b+=startb[2]+(startb[1]+startb[0]*sizeb[1])*sizeb[2];

  b+=blockIdx.x*sizeb[1]*sizeb[2]+(i1+threadIdx.y)*sizeb[2]+threadIdx.x;
  a+=blockIdx.x*sizea[1]*sizea[2]+(i1+threadIdx.y)*sizea[2]+threadIdx.x;

  
  
  for (i2=0; i2 < sizea[2]; i2+=BLOCK_SIZEX) {    
    if ((i2+threadIdx.x<sizea[2])  && (i1+threadIdx.y<sizea[1])){    
      b[i2]=a[i2];
    } 
  }
}

extern "C" {

  void bmgs_paste_cuda_gpu(const double* a, const int sizea[3],
		   double* b, const int sizeb[3], const int startb[3])
  {
    //    size_t asize,bsize;
    

    int gridy=sizea[1]/BLOCK_SIZEY;
    gridy=(sizea[1]%BLOCK_SIZEY) ? gridy+1 : gridy;

    dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY); 
    dim3 dimGrid(sizea[0],gridy);    

    //    fprintf(stdout,"array: %d x %d x %d\t",s->n[0],s->n[1],s->n[2]);
    //fprintf(stdout,"block: %d x %d\t grid: %d x %d\n",BLOCK_SIZEX,BLOCK_SIZEY,s->n[0],gridy);
    
    
    bmgs_paste_cuda_kernel<<<dimGrid, dimBlock, 0>>>(a, sizea,b, sizeb, startb);
    
    

  }
}
