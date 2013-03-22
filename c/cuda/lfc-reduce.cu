#ifndef REDUCE_LFC

#define REDUCE_LFC_MAX_THREADS  (256)
#define REDUCE_LFC_MAX_BLOCKS   (64)
#define REDUCE_LFC_MAX_YBLOCKS   (65535)
#define REDUCE_LFC_BUFFER_SIZE  ((2*GPAW_CUDA_BLOCKS_MAX*REDUCE_LFC_MAX_BLOCKS)*16)
static void *lfc_reduce_buffer=NULL;
static int lfc_reduce_buffer_size=0;

extern "C" {

  void lfc_reduce_init_buffers_cuda()
  {    
    lfc_reduce_buffer=NULL;
    lfc_reduce_buffer_size=0;
  }
  
  void lfc_reduce_dealloc_cuda()
  {
    cudaFree(lfc_reduce_buffer);
    cudaGetLastError();
    lfc_reduce_init_buffers_cuda();
  }

}

static unsigned int lfc_nextPow2( unsigned int x ) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

static void lfc_reduceNumBlocksAndThreads(int n,int *blocks, int *threads)
{
  *threads = (n < REDUCE_LFC_MAX_THREADS) ? lfc_nextPow2(n) :
    REDUCE_LFC_MAX_THREADS;
  *blocks = MIN((n + (*threads - 1)) / (*threads),REDUCE_LFC_MAX_BLOCKS); 
}

static void lfc_reduceNumBlocksAndThreads2(int n,int *blocks, int *threads)
{
  *threads = (n < REDUCE_LFC_MAX_THREADS*2) ? lfc_nextPow2((n + 1)/ 2) :
    REDUCE_LFC_MAX_THREADS;
  *blocks = MIN((n + (*threads * 2 - 1)) / (*threads * 2),REDUCE_LFC_MAX_BLOCKS); 
}

#endif
#define REDUCE_LFC

#define INNAME(f) Zcuda(f ## _map512)
#define REDUCE_LFC_THREADS   512
#include "lfc-reduce-kernel.cu"
#undef  REDUCE_LFC_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## _map256)
#define REDUCE_LFC_THREADS   256
#include "lfc-reduce-kernel.cu"
#undef  REDUCE_LFC_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## _map128)
#define REDUCE_LFC_THREADS   128
#include "lfc-reduce-kernel.cu"
#undef  REDUCE_LFC_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## _map64)
#define REDUCE_LFC_THREADS   64
#include "lfc-reduce-kernel.cu"
#undef  REDUCE_LFC_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## _map32)
#define REDUCE_LFC_THREADS   32
#include "lfc-reduce-kernel.cu"
#undef  REDUCE_LFC_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## _map16)
#define REDUCE_LFC_THREADS   16
#include "lfc-reduce-kernel.cu"
#undef  REDUCE_LFC_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## _map8)
#define REDUCE_LFC_THREADS   8
#include "lfc-reduce-kernel.cu"
#undef  REDUCE_LFC_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## _map4)
#define REDUCE_LFC_THREADS   4
#include "lfc-reduce-kernel.cu"
#undef  REDUCE_LFC_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## _map2)
#define REDUCE_LFC_THREADS   2
#include "lfc-reduce-kernel.cu"
#undef  REDUCE_LFC_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## _map1)
#define REDUCE_LFC_THREADS   1
#include "lfc-reduce-kernel.cu"
#undef  REDUCE_LFC_THREADS 
#undef  INNAME
#undef  INFUNC


#undef  REDUCE_THREADS 
#define INFUNC(a,b) (a)
#define INNAME(f) Zcuda(f ## 512)
#define REDUCE_THREADS   512
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## 256)
#define REDUCE_THREADS   256
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## 128)
#define REDUCE_THREADS   128
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## 64)
#define REDUCE_THREADS   64
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## 32)
#define REDUCE_THREADS   32
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## 16)
#define REDUCE_THREADS   16
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## 8)
#define REDUCE_THREADS   8
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## 4)
#define REDUCE_THREADS   4
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## 2)
#define REDUCE_THREADS   2
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#define INNAME(f) Zcuda(f ## 1)
#define REDUCE_THREADS   1
#include "reduce-kernel.cu"
#undef  REDUCE_THREADS 
#undef  INNAME
#undef  INFUNC


void Zcuda(lfc_reducemap)(LFCObject *lfc,const Tcuda *a_G,int nG,Tcuda *c_xM,int nM, int nvec, int q)

{		
  
  
  int threads;
  int blocks;

  if (lfc_reduce_buffer_size<nM*REDUCE_LFC_BUFFER_SIZE){
    lfc_reduce_dealloc_cuda();
    gpaw_cudaSafeCall(cudaMalloc((void**)(&lfc_reduce_buffer),nM*REDUCE_LFC_BUFFER_SIZE));
    lfc_reduce_buffer_size=nM*REDUCE_LFC_BUFFER_SIZE;

  }
  Tcuda *result_gpu=c_xM;
  int work_buffer_size=(lfc_reduce_buffer_size/sizeof(Tcuda))/2;
  Tcuda *work_buffer1=(Tcuda*)lfc_reduce_buffer;
  Tcuda *work_buffer2=work_buffer1+work_buffer_size;

  
  //  assert(nvec*sizeof(Tcuda)<=REDUCE_LFC_BUFFER_SIZE-2*REDUCE_LFC_MAX_BLOCKS);


  lfc_reduceNumBlocksAndThreads(lfc->max_len_A_gm,&blocks, &threads);
  int mynvec=MAX(MIN((work_buffer_size/nM)/blocks,nvec),1);

  mynvec=MIN(mynvec,(REDUCE_LFC_MAX_YBLOCKS)/nM);
  mynvec=MIN(mynvec,LFC_REDUCE_MAX_NVEC);
  /*int mynvec=1;*/
  
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(Tcuda) : 
    threads * sizeof(Tcuda);
  
  for (int i=0;i<nvec;i+=mynvec) {    
  
    int cunvec=MIN(mynvec,nvec-i);
    int innvec=1;

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, lfc->Mcount, 1);
    //dim3 dimGrid(blocks, cunvec * lfc->Mcount, 1);
    int block_out=blocks;
    
    innvec=cunvec;

    //printf("i %d blocks %d nW %d threads %d maxA %d nM %d Mcount %d cunvec %d\n",i,blocks,lfc->nW,threads,lfc->max_len_A_gm,nM,lfc->Mcount,cunvec);
    switch (threads) 
      {
	
      case 512:
	Zcuda(integrate_mul_kernel_map512)<<< dimGrid, dimBlock, smemSize >>>
	  (a_G+i*nG, nG,lfc->volume_W_gpu,lfc->volume_WMi_gpu,lfc->WMi_gpu,lfc->WMimax,q,(Tcuda*)work_buffer1,block_out,result_gpu+i*nM,lfc->Mcount,nM,innvec);
	break;
      case 256:
	Zcuda(integrate_mul_kernel_map256)<<< dimGrid, dimBlock, smemSize >>>
	  (a_G+i*nG, nG,lfc->volume_W_gpu,lfc->volume_WMi_gpu,lfc->WMi_gpu,lfc->WMimax,q,(Tcuda*)work_buffer1,block_out,result_gpu+i*nM,lfc->Mcount,nM,innvec);
	break;
      case 128:
	Zcuda(integrate_mul_kernel_map128)<<< dimGrid, dimBlock, smemSize >>>
	  (a_G+i*nG, nG,lfc->volume_W_gpu,lfc->volume_WMi_gpu,lfc->WMi_gpu,lfc->WMimax,q,(Tcuda*)work_buffer1,block_out,result_gpu+i*nM,lfc->Mcount,nM,innvec);
	break;
      case 64:
	Zcuda(integrate_mul_kernel_map64)<<< dimGrid, dimBlock, smemSize >>>
	  (a_G+i*nG, nG,lfc->volume_W_gpu,lfc->volume_WMi_gpu,lfc->WMi_gpu,lfc->WMimax,q,(Tcuda*)work_buffer1,block_out,result_gpu+i*nM,lfc->Mcount,nM,innvec);
	break;
      case 32:
	Zcuda(integrate_mul_kernel_map32)<<< dimGrid, dimBlock, smemSize >>>
	  (a_G+i*nG, nG,lfc->volume_W_gpu,lfc->volume_WMi_gpu,lfc->WMi_gpu,lfc->WMimax,q,(Tcuda*)work_buffer1,block_out,result_gpu+i*nM,lfc->Mcount,nM,innvec);
	break;
      case 16:
	Zcuda(integrate_mul_kernel_map16)<<< dimGrid, dimBlock, smemSize >>>
	  (a_G+i*nG, nG,lfc->volume_W_gpu,lfc->volume_WMi_gpu,lfc->WMi_gpu,lfc->WMimax,q,(Tcuda*)work_buffer1,block_out,result_gpu+i*nM,lfc->Mcount,nM,innvec);
	break;
      case  8:
	Zcuda(integrate_mul_kernel_map8)<<< dimGrid, dimBlock, smemSize >>>
	  (a_G+i*nG, nG,lfc->volume_W_gpu,lfc->volume_WMi_gpu,lfc->WMi_gpu,lfc->WMimax,q,(Tcuda*)work_buffer1,block_out,result_gpu+i*nM,lfc->Mcount,nM,innvec);
	break;
      case  4:
	Zcuda(integrate_mul_kernel_map4)<<< dimGrid, dimBlock, smemSize >>>
	  (a_G+i*nG, nG,lfc->volume_W_gpu,lfc->volume_WMi_gpu,lfc->WMi_gpu,lfc->WMimax,q,(Tcuda*)work_buffer1,block_out,result_gpu+i*nM,lfc->Mcount,nM,innvec);
	break;
      case  2:
	Zcuda(integrate_mul_kernel_map2)<<< dimGrid, dimBlock, smemSize >>>
	  (a_G+i*nG, nG,lfc->volume_W_gpu,lfc->volume_WMi_gpu,lfc->WMi_gpu,lfc->WMimax,q,(Tcuda*)work_buffer1,block_out,result_gpu+i*nM,lfc->Mcount,nM,innvec);
	break;
      case  1:
	Zcuda(integrate_mul_kernel_map1)<<< dimGrid, dimBlock, smemSize >>>
	  (a_G+i*nG, nG,lfc->volume_W_gpu,lfc->volume_WMi_gpu,lfc->WMi_gpu,lfc->WMimax,q,(Tcuda*)work_buffer1,block_out,result_gpu+i*nM,lfc->Mcount,nM,innvec);
	break;
      default:
	assert(0);
      }
    //    cudaThreadSynchronize(); 
    assert(!gpaw_cudaSafeCall(cudaGetLastError()));
    
    int s=blocks;   
    int count=0;
    while(s > 1)  {

      int blocks2,threads2;
      lfc_reduceNumBlocksAndThreads2(s, &blocks2, &threads2);
      //      printf("i %d s %d nM %d blocks2  %d threads2 %d\n",i,s,nM,blocks2, threads2);
      dim3 dimBlock(threads2, 1, 1);
      dim3 dimGrid(blocks2, cunvec*nM, 1);
      int smemSize = (threads2 <= 32) ? 2 * threads2 * sizeof(Tcuda) : 
	threads2 * sizeof(Tcuda);
      
      Tcuda *work1=(count%2) ?  work_buffer2 : work_buffer1;
      Tcuda *work2=(count%2) ?  work_buffer1 : work_buffer2;
      count++;
      
      switch (threads2) 
	{
	case 512:
	  Zcuda(reduce_kernel512)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i*nM,
	     s,block_out,block_out,cunvec*nM);        
	  break;
	case 256:
	  Zcuda(reduce_kernel256)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i*nM,
	     s,block_out,block_out,cunvec*nM);        
	  break;
	case 128:
	  Zcuda(reduce_kernel128)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i*nM,
	     s,block_out,block_out,cunvec*nM);        
	  break;
	case 64:
	  Zcuda(reduce_kernel64)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i*nM,
	     s,block_out,block_out,cunvec*nM);        
	  break;
	case 32:
	  Zcuda(reduce_kernel32)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i*nM,
	     s,block_out,block_out,cunvec*nM);        
	  break;
	case 16:
	  Zcuda(reduce_kernel16)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i*nM,
	     s,block_out,block_out,cunvec*nM);        
	  break;
	case  8:
	  Zcuda(reduce_kernel8)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i*nM,
	     s,block_out,block_out,cunvec*nM);        
	  break;
	case  4:
	  Zcuda(reduce_kernel4)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i*nM,
	     s,block_out,block_out,cunvec*nM);        
	  break;
	case  2:	  
	  Zcuda(reduce_kernel2)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i*nM,
	     s,block_out,block_out,cunvec*nM);        	  
	  break;
	case  1:
	  Zcuda(reduce_kernel1)<<< dimGrid, dimBlock, smemSize >>>
	    ((Tcuda*)work1,NULL, (Tcuda*)work2,result_gpu+i*nM,
	     s,block_out,block_out,cunvec*nM);        
	  break;
	default:
	  assert(0);
	}
      //cudaThreadSynchronize(); 
      assert(!gpaw_cudaSafeCall(cudaGetLastError()));
    
    
      s = (s + (threads2*2-1)) / (threads2*2);
    }
  }
  
}
