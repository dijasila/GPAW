
#ifndef REDUCE

#define REDUCE_MAX_THREADS  (256)
#define REDUCE_MAX_BLOCKS   (64)
//#define REDUCE_MAX_NVEC     (32*1024)
#define REDUCE_MAX_NVEC     (128*1024)
#define REDUCE_BUFFER_SIZE  ((REDUCE_MAX_NVEC+2*GPAW_CUDA_BLOCKS_MAX*REDUCE_MAX_BLOCKS)*16)
static void *reduce_buffer=NULL;

extern "C" {

  void reduce_init_buffers_cuda()
  {    
    reduce_buffer=NULL;
  }
  
  void reduce_dealloc_cuda()
  {
    if (reduce_buffer) cudaFree(reduce_buffer);
    cudaGetLastError();
    reduce_init_buffers_cuda();
  }

}

static unsigned int nextPow2( unsigned int x ) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

static void reduceNumBlocksAndThreads(int n,int *blocks, int *threads)
{
  *threads = (n < REDUCE_MAX_THREADS*2) ? nextPow2((n + 1)/ 2) :
    REDUCE_MAX_THREADS;
  *blocks = MIN((n + (*threads * 2 - 1)) / (*threads * 2),REDUCE_MAX_BLOCKS); 
}


#endif
#define REDUCE


#define INFUNC(a,b) MAPFUNC(a,b)
#define INNAME(f) MAPNAME(f ## _map512)
#define REDUCE_THREADS   512
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## _map256)
#define REDUCE_THREADS   256
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## _map128)
#define REDUCE_THREADS   128
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## _map64)
#define REDUCE_THREADS   64
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## _map32)
#define REDUCE_THREADS   32
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## _map16)
#define REDUCE_THREADS   16
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## _map8)
#define REDUCE_THREADS   8
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## _map4)
#define REDUCE_THREADS   4
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## _map2)
#define REDUCE_THREADS   2
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## _map1)
#define REDUCE_THREADS   1
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#undef  INFUNC


#define INFUNC(a,b) (a)
#define INNAME(f) MAPNAME(f ## 512)
#define REDUCE_THREADS   512
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## 256)
#define REDUCE_THREADS   256
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## 128)
#define REDUCE_THREADS   128
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## 64)
#define REDUCE_THREADS   64
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## 32)
#define REDUCE_THREADS   32
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## 16)
#define REDUCE_THREADS   16
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## 8)
#define REDUCE_THREADS   8
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## 4)
#define REDUCE_THREADS   4
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## 2)
#define REDUCE_THREADS   2
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) MAPNAME(f ## 1)
#define REDUCE_THREADS   1
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#undef  INFUNC



void 
MAPNAME(reducemap)(const Tcuda *d_idata1, const Tcuda *d_idata2, 
		   Tcuda *d_odata,int size,int nvec)
{

  
  int threads;
  int blocks;
  if (reduce_buffer==NULL){
    gpaw_cudaSafeCall(cudaMalloc((void**)(&reduce_buffer),REDUCE_BUFFER_SIZE));
  }
  reduceNumBlocksAndThreads(size,&blocks, &threads);
  int blo2=(blocks+(REDUCE_MAX_THREADS*2-1))/(REDUCE_MAX_THREADS*2);
  int min_wsize=blocks+blo2;
  int work_buffer_size=((REDUCE_BUFFER_SIZE)/sizeof(Tcuda)-nvec);

  assert(min_wsize<work_buffer_size);

  int mynvec=MAX(MIN(work_buffer_size/min_wsize,nvec),1);
  
  Tcuda *result_gpu=(Tcuda*)reduce_buffer;
  Tcuda *work_buffer1=result_gpu+nvec;
  Tcuda *work_buffer2=work_buffer1+mynvec*blocks;

  int smemSize = (threads <= 32) ? 2 * threads * sizeof(Tcuda) : 
    threads * sizeof(Tcuda);

  for (int i=0;i<nvec;i+=mynvec) {    
    int cunvec=MIN(mynvec,nvec-i);

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, cunvec, 1);
    int block_out=blocks;
    
    int s=size;    
    
    switch (threads) 
      {
      case 512:
	MAPNAME(reduce_kernel_map512)<<< dimGrid, dimBlock, smemSize >>>
	  (d_idata1+i*size,d_idata2+i*size, 
	   (Tcuda*)work_buffer1,result_gpu+i, s, size,block_out,cunvec);
	break;
      case 256:
	MAPNAME(reduce_kernel_map256)<<< dimGrid, dimBlock, smemSize >>>
	  (d_idata1+i*size,d_idata2+i*size, 
	   (Tcuda*)work_buffer1,result_gpu+i, s, size,block_out,cunvec);
	break;
      case 128:
	MAPNAME(reduce_kernel_map128)<<< dimGrid, dimBlock, smemSize >>>
	  (d_idata1+i*size,d_idata2+i*size, 
	   (Tcuda*)work_buffer1,result_gpu+i, s, size,block_out,cunvec);
	break;
      case 64:
	MAPNAME(reduce_kernel_map64)<<< dimGrid, dimBlock, smemSize >>>
	  (d_idata1+i*size,d_idata2+i*size, 
	   (Tcuda*)work_buffer1,result_gpu+i, s, size,block_out,cunvec);
	break;
      case 32:
	MAPNAME(reduce_kernel_map32)<<< dimGrid, dimBlock, smemSize >>>
	  (d_idata1+i*size,d_idata2+i*size, 
	   (Tcuda*)work_buffer1,result_gpu+i, s, size,block_out,cunvec);
	break;
      case 16:
	MAPNAME(reduce_kernel_map16)<<< dimGrid, dimBlock, smemSize >>>
	  (d_idata1+i*size,d_idata2+i*size, 
	   (Tcuda*)work_buffer1,result_gpu+i, s, size,block_out,cunvec);
	break;
      case  8:
	MAPNAME(reduce_kernel_map8)<<< dimGrid, dimBlock, smemSize >>>
	  (d_idata1+i*size,d_idata2+i*size, 
	   (Tcuda*)work_buffer1,result_gpu+i, s, size,block_out,cunvec);
	break;
      case  4:
	MAPNAME(reduce_kernel_map4)<<< dimGrid, dimBlock, smemSize >>>
	  (d_idata1+i*size,d_idata2+i*size, 
	   (Tcuda*)work_buffer1,result_gpu+i, s, size,block_out,cunvec);
	break;
      case  2:
	MAPNAME(reduce_kernel_map2)<<< dimGrid, dimBlock, smemSize >>>
	  (d_idata1+i*size,d_idata2+i*size, 
	   (Tcuda*)work_buffer1,result_gpu+i, s, size,block_out,cunvec);
	break;
      case  1:
	MAPNAME(reduce_kernel_map1)<<< dimGrid, dimBlock, smemSize >>>
	  (d_idata1+i*size,d_idata2+i*size, 
	   (Tcuda*)work_buffer1,result_gpu+i, s, size,block_out,cunvec);
	break;
      default:
	assert(0);
      }
    gpaw_cudaSafeCall(cudaGetLastError());
    
    s=blocks;   
    int count=0;
    while(s > 1)  {
      int blocks2,threads2;
      int block_in=block_out;
      reduceNumBlocksAndThreads(s, &blocks2, &threads2);
      block_out=blocks2;
      dim3 dimBlock(threads2, 1, 1);
      dim3 dimGrid(blocks2, cunvec, 1);
      int smemSize = (threads2 <= 32) ? 2 * threads2 * sizeof(Tcuda) : 
	threads2 * sizeof(Tcuda);
      
      Tcuda *work1=(count%2) ?  work_buffer2 : work_buffer1;
      Tcuda *work2=(count%2) ?  work_buffer1 : work_buffer2;
      count++;
      
      switch (threads2) 
	{
	case 512:
	  MAPNAME(reduce_kernel512)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i,
	     s,block_in,block_out,cunvec);        
	  break;
	case 256:
	  MAPNAME(reduce_kernel256)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i,
	     s,block_in,block_out,cunvec);        
	  break;
	case 128:
	  MAPNAME(reduce_kernel128)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i,
	     s,block_in,block_out,cunvec);        
	  break;
	case 64:
	  MAPNAME(reduce_kernel64)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i,
	     s,block_in,block_out,cunvec);        
	  break;
	case 32:
	  MAPNAME(reduce_kernel32)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i,
	     s,block_in,block_out,cunvec);        
	  break;
	case 16:
	  MAPNAME(reduce_kernel16)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i,
	     s,block_in,block_out,cunvec);        
	  break;
	case  8:
	  MAPNAME(reduce_kernel8)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i,
	     s,block_in,block_out,cunvec);        
	  break;
	case  4:
	  MAPNAME(reduce_kernel4)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i,
	     s,block_in,block_out,cunvec);        
	  break;
	case  2:	  
	  MAPNAME(reduce_kernel2)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i,
	     s,block_in,block_out,cunvec);        	  
	  break;
	case  1:
	  MAPNAME(reduce_kernel1)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i,
	     s,block_in,block_out,cunvec);        
	  break;
	default:
	  assert(0);
	}
      gpaw_cudaSafeCall(cudaGetLastError());
    
    
      s = (s + (threads2*2-1)) / (threads2*2);
    }
  }
  
  GPAW_CUDAMEMCPY(d_odata,result_gpu,Tcuda,nvec,
		  cudaMemcpyDeviceToHost);
  
}
