
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include <sys/types.h>
#include <sys/time.h>



extern "C" {
  
#include </usr/include/complex.h>
#include <Python.h>
  //  typedef double complex double_complex;
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
   
  
#include "../lfc.h"

#include "gpaw-cuda-int.h"

}
#ifndef CUGPAWCOMPLEX

#define INLINE inline
  
static INLINE void* gpaw_malloc(int n)
{
  void* p = malloc(n);
  assert(p != NULL);
  return p;
}
#define GPAW_MALLOC(T, n) (T*)(gpaw_malloc((n) * sizeof(T)))

#define GPAW_CUDA_INT_BLOCKS (GPAW_CUDA_BLOCKS)
//#define GPAW_CUDA_INT_BLOCKS 1

#define BLOCK_Y 16
#define REDUCE_THREADS 64

#define WARP      32
#define HALFWARP  (WARP/2)

#include "cuda.h"
#include "cuda_runtime_api.h"

__device__ unsigned int retirementCount = {0};

static inline unsigned int nextPow2( unsigned int x ) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}


/*
__host__ __device__ static inline unsigned int nextPow2_kernel( unsigned int x ) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

static unsigned int nextHWarp( unsigned int x){
  
    return HALFWARP*((x+HALFWARP-1)/HALFWARP);
}
*/
#endif









__global__ void Zcuda(integrate_reduce_kernel)(LFVolume_gpu* volume_W,
					       int max_n,int swap,int blocks,
					       Tcuda *c_xM,int nW,double dv,
					       int nM)
{
  extern __shared__ Tcuda Zcuda(sdata)[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory

  unsigned int tid = threadIdx.x;
    
  int w=blockIdx.y/blocks;
  int x=blockIdx.y-w*blocks;
  
  
  LFVolume_gpu *v_gpu;
  int nm,len1,len2;
  Tcuda  *work1,*work2;
  
  v_gpu=&volume_W[w]; 
  nm=v_gpu->nm;

  if (!swap) {
    len1=v_gpu->len_work1;
    len2=v_gpu->len_work2;
    work1=(Tcuda*)v_gpu->work1_A_gm;
    work2=(Tcuda*)v_gpu->work2_A_gm;//+len*nm;
  } else {
    len1=v_gpu->len_work2;
    len2=v_gpu->len_work1;
    work1=(Tcuda*)v_gpu->work2_A_gm;//+len*nm;
    work2=(Tcuda*)v_gpu->work1_A_gm;
  }
  //  work1+=2*x*len*nm;
  //work2+=2*x*len*nm;
  work1+=x*len1*nm;
  work2+=x*len2*nm;

  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;     
  int n=min(len1,max_n);
  Tcuda * g_idata=work1;
  Tcuda * g_odata=work2;
  for (int m=0;m<nm;m++){
    
    Tcuda mySum = (i < n) ? g_idata[i] : MAKED(0);
    if (i + blockDim.x < n) 
      IADD(mySum , g_idata[i+blockDim.x]);  
    
    Zcuda(sdata)[tid] = mySum;
    __syncthreads();
      
    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1) 
      {
	if (tid < s) 
	  {
	    Zcuda(sdata)[tid] = mySum = ADD(mySum , Zcuda(sdata)[tid + s]);
	  }
	__syncthreads();
      }
    if (tid < 32)
      {
	volatile Tcuda *smem = Zcuda(sdata);
#ifdef CUGPAWCOMPLEX	
	if (REDUCE_THREADS >=  64){  
	  smem[tid].x = mySum.x = mySum.x + smem[tid + 32].x;
	  smem[tid].y = mySum.y = mySum.y + smem[tid + 32].y;
	}
	if (REDUCE_THREADS >=  32){  
	  smem[tid].x = mySum.x = mySum.x + smem[tid + 16].x;
	  smem[tid].y = mySum.y = mySum.y + smem[tid + 16].y;
	}
	smem[tid].x = mySum.x = mySum.x + smem[tid + 8].x;
	smem[tid].y = mySum.y = mySum.y + smem[tid + 8].y;
	smem[tid].x = mySum.x = mySum.x + smem[tid + 4].x;
	smem[tid].y = mySum.y = mySum.y + smem[tid + 4].y;
	smem[tid].x = mySum.x = mySum.x + smem[tid + 2].x;
	smem[tid].y = mySum.y = mySum.y + smem[tid + 2].y;
	smem[tid].x = mySum.x = mySum.x + smem[tid + 1].x;
	smem[tid].y = mySum.y = mySum.y + smem[tid + 1].y;	
#else
	if (REDUCE_THREADS >=  64)  
	  smem[tid] = mySum = ADD(mySum , smem[tid + 32]);
	if (REDUCE_THREADS >=  32)  
	  smem[tid] = mySum = ADD(mySum , smem[tid + 16]);
	smem[tid] = mySum = ADD(mySum , smem[tid + 8]);
	smem[tid] = mySum = ADD(mySum , smem[tid + 4]);
	smem[tid] = mySum = ADD(mySum , smem[tid + 2]);
	smem[tid] = mySum = ADD(mySum , smem[tid + 1]);	
#endif
      }
    
    // write result for this block to global mem 
    __syncthreads();
    if ((tid == 0) && (blockIdx.x<n)){
      g_odata[blockIdx.x] = Zcuda(sdata)[0];
    }
    g_idata+=len1;
    g_odata+=len2;
     __syncthreads();
  }
  
  if (gridDim.x==1){
    __shared__ bool amLast;
    __threadfence();
    if (tid == 0) {
      unsigned int ticket = atomicInc(&retirementCount, blocks*nW);
      amLast = (ticket == blocks*nW-1);
    }
    __syncthreads();
    if ((amLast) ) {
      Tcuda *c_M;
      Tcuda *work;	
      int len;
      //c_xM+=x*nM;
      for (int ww=0;ww<nW;ww++){
	v_gpu=&volume_W[ww]; 

	nm=v_gpu->nm; 	
	if (!swap) {	 
	  len=v_gpu->len_work2;
	  work=(Tcuda*)v_gpu->work2_A_gm;
	} else {
	  len=v_gpu->len_work1;
	  work=(Tcuda*)v_gpu->work1_A_gm;
	}
	c_M=c_xM+v_gpu->M;
	for (int xx=0;xx<blocks;xx++){
	  if (tid<nm){
	    IADD(c_M[tid],MULTD(work[tid*len] , dv));
	  }
	  c_M+=nM;
	  work+=len*nm;
	  __syncthreads();    
	}
      }
      retirementCount=0;
    }
    
    }
  
}


__global__ void Zcuda(integrate_get_c_xM)(LFVolume_gpu* volume_W,int swap,
					  Tcuda *c_xM,int nW,double dv,int nM)
{
  Tcuda  *work;
  LFVolume_gpu *v_gpu;
  int nm,len;
  Tcuda *c_M;
  int x=blockIdx.x;

  for (int w=0;w<nW;w++){
    v_gpu=&volume_W[w]; 
    len=v_gpu->len_work1;
    nm=v_gpu->nm;    
    if (!swap) {
      len=v_gpu->len_work2;
      //      work=(Tcuda*)v_gpu->work_A_gm+len*nm+2*x*len*nm;
      work=(Tcuda*)v_gpu->work2_A_gm+x*len*nm;
    } else {
      len=v_gpu->len_work1;
      work=(Tcuda*)v_gpu->work1_A_gm+x*len*nm;
      //work=(Tcuda*)v_gpu->work1_A_gm+2*x*len*nm;

    }
    c_M=c_xM+v_gpu->M+x*nM;
    for(int m=0;m<nm;m++){
      IADD(c_M[m],MULTD(work[0] , dv));
      //IADD(c_M[m],MAKED(dv));
      work+=len;
    }
  }
}



__global__ void Zcuda(integrate_mul_kernel)(const Tcuda *a_G,int *G_B1,
					    int *G_B2,LFVolume_gpu **volume_i,
					    int *A_gm_i,int *work_i,int *ni,
					    int nimax,int na_G,
					    cuDoubleComplex *phase_i,int max_k,
					    int q,int nB_gpu)
  
{
  // extern __shared__ Tcuda Zcuda(wdatas)[];
  
  int G=threadIdx.x;
  int B=blockIdx.x*blockDim.y+threadIdx.y;
  if (B>=nB_gpu) return;

  int x=blockIdx.y;

  int nii,Gb,Ga,nG;
  LFVolume_gpu* v;
  double* A_gm;
  int nm;
  Tcuda *work;
  int len,len_w;
  //  Tcuda *Zcuda(wdata)=Zcuda(wdatas)+threadIdx.y*blockDim.x;
  
  
  
  nii=ni[B]; 
  Ga=G_B1[B];
  Gb=G_B2[B];
  nG=Gb-Ga;
  int nend=(nG+1)/2;
  //  int nend2=blockDim.x;
  a_G+=na_G*x;
  
  if (G<nend){
    Tcuda av = a_G[Ga+G];
    Tcuda av2 = (G+nend<nG) ? a_G[Ga+G+nend]:MAKED(0);      
    for(int i=0;i<nii;i++){        
      Tcuda avv=av;
      Tcuda avv2=av2;
     
      v = volume_i[B+i*nB_gpu];
      A_gm = v->A_gm+A_gm_i[B+i*nB_gpu];
      nm = v->nm;

      len=v->len_A_gm;
      len_w=v->len_work1;
      work = (Tcuda*)v->work1_A_gm+work_i[B+i*nB_gpu]+x*len_w*nm;
#ifdef CUGPAWCOMPLEX
      avv=MULTT(avv, phase_i[max_k*nimax*B+q*nimax+i]);
      avv2=MULTT(avv2, phase_i[max_k*nimax*B+q*nimax+i]);
#endif
      for (int m = 0; m < nm; m++){
	Tcuda mySum = MULTD(avv , A_gm[G]);
	//if (G+nend<nG)
	IADD(mySum,MULTD(avv2 , A_gm[G+nend]));

	
	/*Zcuda(wdata)[G]=mySum;
	__syncthreads();
	for(unsigned int s=nend2/2; s>0; s>>=1) {
	  if ((G < s) && ((G+s)< nend))
	    Zcuda(wdata)[G] = mySum = ADD(mySum , Zcuda(wdata)[G + s]);
	  __syncthreads();
	}		
	if (G==0)
	  work[0]=Zcuda(wdata)[0];
	*/

	work[G]=mySum;
	A_gm+=len;
	work+=len_w;	
	//__syncthreads();
	
      }
    }
  }
}



__global__ void Zcuda(add_kernel)(Tcuda *a_G,const Tcuda *c_M,int *G_B1,
				  int *G_B2,LFVolume_gpu **volume_i,
				  int *A_gm_i,int *ni,int nimax,int na_G,
				  int nM,cuDoubleComplex *phase_i,int max_k,
				  int q,int nB_gpu)
  
{

  int G=threadIdx.x;
  //int B=blockIdx.x;
  int B=blockIdx.x*blockDim.y+threadIdx.y;
  if (B>=nB_gpu) return;

  int x=blockIdx.y;
  
  int nii,Gb,Ga,nG;
  LFVolume_gpu* v;
  double* A_gm;
  const Tcuda* c_Mt;
  int nm;
  int len;
   

  nii=ni[B]; 
  Ga=G_B1[B];
  Gb=G_B2[B];
  nG=Gb-Ga;
  a_G+=Ga+na_G*x;
  c_M+=nM*x;  
  Tcuda av=MAKED(0);//=a_G[G];
  if ( G<nG ){
    for(int i=0;i<nii;i++){     
      Tcuda avv;
      v = volume_i[B+i*nB_gpu];
      A_gm =v->A_gm+ A_gm_i[B+i*nB_gpu]+G;
      nm = v->nm;
      len=v->len_A_gm;
      c_Mt=c_M+v->M;
      
      avv= MULTD(c_Mt[0] , A_gm[0]); 	
      for (int m = 1; m < nm; m+=2){
	A_gm+=len;
	IADD(avv, MULTD(c_Mt[m] , A_gm[0])); 	
	A_gm+=len;
	IADD(avv, MULTD(c_Mt[m+1] , A_gm[0])); 	
      }
#ifdef CUGPAWCOMPLEX
      avv=MULTT(avv ,cuConj(phase_i[max_k*nimax*B+q*nimax+i]));
#endif
      IADD(av,avv);
      //    }
    }
    /* if (G<nG)*/ IADD(a_G[G] , av);
  }
}
  
#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "lfc-cuda.cu"

extern "C" {

  void lfc_dealloc_cuda(LFCObject *self)
  {
    if (self->cuda){
      for (int W = 0; W < self->nW; W++){
	LFVolume_gpu* volume_gpu = &self->volume_W_cuda[W];
	cudaFree(volume_gpu->A_gm);
	cudaFree(volume_gpu->work1_A_gm);	
	cudaFree(volume_gpu->work2_A_gm);	
	/*
	gpaw_cudaSafeCall(cudaFree(volume_gpu->A_gm));
	gpaw_cudaSafeCall(cudaFree(volume_gpu->work_A_gm));
	gpaw_cudaSafeCall(cudaFree(volume_gpu->work2_A_gm));
	*/
      }
      free(self->volume_W_cuda);
      /*
      gpaw_cudaSafeCall(cudaFree(self->volume_W_gpu));
      gpaw_cudaSafeCall(cudaFree(self->G_B_gpu));
      gpaw_cudaSafeCall(cudaFree(self->volume_i_gpu));
      gpaw_cudaSafeCall(cudaFree(self->A_gm_i_gpu));
      gpaw_cudaSafeCall(cudaFree(self->ni_gpu));
      */
      cudaFree(self->volume_W_gpu);
      cudaFree(self->G_B1_gpu);
      cudaFree(self->G_B2_gpu);
      cudaFree(self->volume_i_gpu);
      cudaFree(self->A_gm_i_gpu);
      cudaFree(self->work_i_gpu);
      cudaFree(self->ni_gpu);
    }
  }

  
  
  
  static void *transp(void *matrix, int rows, int cols, size_t item_size)
  {
#define ALIGNMENT 16    /* power of 2 >= minimum array boundary alignment; maybe unnecessary but machine dependent */
    
    char *cursor;
    char carry[ALIGNMENT];
    size_t block_size, remaining_size;
    int nadir, lag, orbit, ents;
    
    if (rows == 1 || cols == 1)
      return matrix;
    ents = rows * cols;
    cursor = (char *) matrix;
    remaining_size = item_size;
    while ((block_size = ALIGNMENT < remaining_size ? ALIGNMENT : remaining_size))
      {
	nadir = 1;
	/* first and last entries are always fixed points so aren't visited */
	while (nadir + 1 < ents)
	  {
	    memcpy(carry, &cursor[(lag = nadir) * item_size], block_size);
	    while ((orbit = lag / rows + cols * (lag % rows)) > nadir)
	      /* follow a complete cycle */
	      {
		memcpy(&cursor[lag * item_size], &cursor[orbit * item_size], block_size);
		lag = orbit;
	      }
	    memcpy(&cursor[lag * item_size], carry, block_size);
	    orbit = nadir++;
	    while (orbit < nadir && nadir + 1 < ents) 
	      /* find the next unvisited index by an exhaustive search */
	      {
		orbit = nadir;
		while ((orbit = orbit / rows + cols * (orbit % rows)) > nadir);
		if (orbit < nadir) nadir++;
	      }
	  }
	cursor += block_size;
	remaining_size -= block_size;
      }
    return matrix;
  }
  
  
  PyObject * NewLFCObject_cuda(LFCObject *self, PyObject *args)
  {
    PyObject* A_Wgm_obj;
    const PyArrayObject* M_W_obj;
    const PyArrayObject* G_B_obj;
    const PyArrayObject* W_B_obj;
    double dv;
    const PyArrayObject* phase_kW_obj;
    int cuda=1;
    
    
    if (!PyArg_ParseTuple(args, "OOOOdO|iO",
			  &A_Wgm_obj, &M_W_obj, &G_B_obj, &W_B_obj, &dv,
			  &phase_kW_obj, &cuda))
      return NULL; 
    
    
    if (!cuda) return (PyObject*)self;
    
    //    printf("New lfc object cuda\n");
    
    int nimax=self->nimax;
    int max_k=0;
    
    LFVolume_gpu* volume_W_gpu;
    volume_W_gpu = GPAW_MALLOC(LFVolume_gpu, self->nW);
    if (self->bloch_boundary_conditions){
      max_k=phase_kW_obj->dimensions[0];
    }    
    self->max_k=max_k;
    self->max_len_A_gm=0;
    self->max_len_work=0;
    self->max_nG=0;
    for (int W = 0; W < self->nW; W++) {
      LFVolume_gpu*  v_gpu = &volume_W_gpu[W];
      LFVolume* v = &self->volume_W[W];
      
      const PyArrayObject* A_gm_obj =				\
	(const PyArrayObject*)PyList_GetItem(A_Wgm_obj, W);
      
      double *work_A_gm = GPAW_MALLOC(double, self->ngm_W[W]);

      GPAW_CUDAMALLOC(&(v_gpu->A_gm),double,self->ngm_W[W]);   

      memcpy(work_A_gm,v->A_gm,sizeof(double)*self->ngm_W[W]);
      transp(work_A_gm,A_gm_obj->dimensions[0],A_gm_obj->dimensions[1],
	     sizeof(double));

      GPAW_CUDAMEMCPY(v_gpu->A_gm,work_A_gm,double,self->ngm_W[W],
		      cudaMemcpyHostToDevice);
      free(work_A_gm);
      
      v_gpu->nm=v->nm;
      v_gpu->M=v->M;
      v_gpu->W=v->W;
      v_gpu->len_work1=0;
      v_gpu->len_A_gm=0;
    }
    
    GPAW_CUDAMALLOC(&(self->volume_W_gpu),LFVolume_gpu,self->nW);    
    
    int* i_W = self->i_W; 
    LFVolume_gpu** volume_i = GPAW_MALLOC(LFVolume_gpu*, nimax);  
    int Ga = 0; 
    int ni = 0; 
    LFVolume_gpu **volume_i_gpu=GPAW_MALLOC(LFVolume_gpu*,self->nB*nimax);
    int *A_gm_i_gpu=GPAW_MALLOC(int,self->nB*nimax);
    int *ni_gpu=GPAW_MALLOC(int,self->nB);
    int *G_B1_gpu=GPAW_MALLOC(int,self->nB);
    int *G_B2_gpu=GPAW_MALLOC(int,self->nB);

    cuDoubleComplex *phase_i_gpu;
    cuDoubleComplex *phase_i;

    int *work_i_gpu=GPAW_MALLOC(int,self->nB*nimax);
    
    if (self->bloch_boundary_conditions){
      phase_i_gpu = GPAW_MALLOC(cuDoubleComplex,max_k*self->nB*nimax);
      phase_i = GPAW_MALLOC(cuDoubleComplex,max_k*nimax);      
    }

    int nB_gpu=0;

    for (int B = 0; B < self->nB; B++) {
      int Gb = self->G_B[B]; 
      int nG = Gb - Ga; 	

      if ((nG > 0) && (ni>0)) {	

	for (int i = 0; i < ni; i++) {
	  LFVolume_gpu* v = volume_i[i];
	  volume_i_gpu[nB_gpu*nimax+i]=self->volume_W_gpu+(v-volume_W_gpu);
	  A_gm_i_gpu[nB_gpu*nimax+i]=v->len_A_gm;
	  work_i_gpu[nB_gpu*nimax+i]=v->len_work1;
	  if (self->bloch_boundary_conditions){
	    for (int kk=0;kk<max_k;kk++){	      
	      phase_i_gpu[i+nB_gpu*nimax*max_k+kk*nimax]=phase_i[i+kk*nimax];
	    }
	  }	  
	  int wnG=(nG+1)/2;
 	  v->len_work1+=wnG;
	  v->len_A_gm+=nG;
	}
	self->max_nG=MAX(self->max_nG,nG);	  	  
	G_B1_gpu[nB_gpu]=Ga;
	G_B2_gpu[nB_gpu]=Gb;
	ni_gpu[nB_gpu]=ni;
	nB_gpu++;
      } 
      int Wnew = self->W_B[B];
      if (Wnew >= 0)	  { 
	/* Entering new sphere: */
	volume_i[ni] = &volume_W_gpu[Wnew];
	if (self->bloch_boundary_conditions){
	  for (int i=0;i<max_k;i++){
	    phase_i[ni+i*nimax].x=creal(self->phase_kW[Wnew+i*self->nW]);
	    phase_i[ni+i*nimax].y=cimag(self->phase_kW[Wnew+i*self->nW]);
	  }
	}
	i_W[Wnew] = ni; 
	
	ni++;
      } else { 
	/* Leaving sphere: */
	int Wold = -1 - Wnew;
	int iold = i_W[Wold];
	volume_W_gpu[Wold].len_A_gm = volume_i[iold]->len_A_gm;
	volume_W_gpu[Wold].len_work1 = volume_i[iold]->len_work1;
	ni--;
	volume_i[iold] = volume_i[ni];
	if (self->bloch_boundary_conditions){
	  for (int i=0;i<max_k;i++){
	    phase_i[iold+i*nimax]=phase_i[ni+i*nimax]; 
	  }
	}

	int Wlast = volume_i[iold]->W;
	i_W[Wlast] = iold;
      } 
      Ga = Gb; 
    } 
    for (int W = 0; W < self->nW; W++){
      LFVolume_gpu* v = &volume_W_gpu[W];
      self->max_len_work=MAX(self->max_len_work,v->len_work1);
      self->max_len_A_gm=MAX(self->max_len_A_gm,v->len_A_gm);
      int wsize1=v->len_work1*v->nm*GPAW_CUDA_INT_BLOCKS;
      if (self->bloch_boundary_conditions){
	GPAW_CUDAMALLOC(&(v->work1_A_gm),cuDoubleComplex,wsize1);
      } else {
	GPAW_CUDAMALLOC(&(v->work1_A_gm),double,wsize1);
      }
    }
    for (int W = 0; W < self->nW; W++){
      LFVolume_gpu* v = &volume_W_gpu[W];

      v->len_work2=MAX(1,MIN(v->len_work1,
			   (nextPow2(self->max_len_work)/(REDUCE_THREADS))));
      //printf("len_W %d %d\n",v->len_work1,v->len_work2);
      //v->len_work2=v->len_work1;
      int wsize2=v->len_work2*v->nm*GPAW_CUDA_INT_BLOCKS;
      if (self->bloch_boundary_conditions){
	GPAW_CUDAMALLOC(&(v->work2_A_gm),cuDoubleComplex,wsize2);
      } else {
	GPAW_CUDAMALLOC(&(v->work2_A_gm),double,wsize2);
      }
    }
    self->nB_gpu=nB_gpu;
    
    GPAW_CUDAMALLOC(&(self->G_B1_gpu),int,nB_gpu);
    GPAW_CUDAMEMCPY(self->G_B1_gpu,G_B1_gpu,int,nB_gpu,
		    cudaMemcpyHostToDevice);
    
    GPAW_CUDAMALLOC(&(self->G_B2_gpu),int,nB_gpu);
    GPAW_CUDAMEMCPY(self->G_B2_gpu,G_B2_gpu,int,nB_gpu,
		    cudaMemcpyHostToDevice);


    GPAW_CUDAMALLOC(&(self->ni_gpu),int,nB_gpu);
    GPAW_CUDAMEMCPY(self->ni_gpu,ni_gpu,int,nB_gpu,
		    cudaMemcpyHostToDevice);
    
    transp(volume_i_gpu,nB_gpu,nimax,sizeof(LFVolume_gpu*));
    GPAW_CUDAMALLOC(&(self->volume_i_gpu),LFVolume_gpu*,
		    nB_gpu*nimax);
    GPAW_CUDAMEMCPY(self->volume_i_gpu,volume_i_gpu,LFVolume_gpu*,
		    nB_gpu*nimax,cudaMemcpyHostToDevice);
    
    transp(A_gm_i_gpu,nB_gpu,nimax,sizeof(int));
    GPAW_CUDAMALLOC(&(self->A_gm_i_gpu),int,nB_gpu*nimax);
    GPAW_CUDAMEMCPY(self->A_gm_i_gpu,A_gm_i_gpu,int,nB_gpu*nimax,
		    cudaMemcpyHostToDevice);
    
    if (self->bloch_boundary_conditions){
      GPAW_CUDAMALLOC(&(self->phase_i_gpu),cuDoubleComplex,
		      max_k*nB_gpu*nimax);
      GPAW_CUDAMEMCPY(self->phase_i_gpu,phase_i_gpu,cuDoubleComplex,
		      max_k*nB_gpu*nimax,cudaMemcpyHostToDevice);
      
    }
    transp(work_i_gpu,nB_gpu,nimax,sizeof(int));
    GPAW_CUDAMALLOC(&(self->work_i_gpu),int,nB_gpu*nimax);
    GPAW_CUDAMEMCPY(self->work_i_gpu,work_i_gpu,int,nB_gpu*nimax,
		    cudaMemcpyHostToDevice);

    self->volume_W_cuda=volume_W_gpu;


    GPAW_CUDAMEMCPY(self->volume_W_gpu,volume_W_gpu,LFVolume_gpu,self->nW,
		    cudaMemcpyHostToDevice);
    free(volume_i);
    free(volume_i_gpu);
    free(A_gm_i_gpu);
    free(work_i_gpu);
    free(ni_gpu);
    free(G_B1_gpu);
    free(G_B2_gpu);
    if (self->bloch_boundary_conditions){
      free(phase_i_gpu);
    }

    return (PyObject*)self;
  }

  
  PyObject* integrate_cuda_gpu(LFCObject *lfc, PyObject *args)
  {
    
    CUdeviceptr a_xG_gpu,c_xM_gpu;
    PyObject *shape,*c_shape;
  
    int q;
  
    if (!PyArg_ParseTuple(args, "nOnOi", &a_xG_gpu,&shape, &c_xM_gpu,&c_shape,
			  &q))
      return NULL; 
    
       
    int nd = PyTuple_Size(shape);

    int nx=PyInt_AsLong(PyTuple_GetItem(shape,0));
    for (int i=1;i<nd-3;i++)
      nx*=PyInt_AsLong(PyTuple_GetItem(shape,i));
  
    int nG=PyInt_AsLong(PyTuple_GetItem(shape,nd-3));
    for (int i=nd-2;i<nd;i++)
      nG*=PyInt_AsLong(PyTuple_GetItem(shape,i));

    int c_nd = PyTuple_Size(c_shape);
    int nM = PyInt_AsLong(PyTuple_GetItem(c_shape,c_nd-1));
    double dv = lfc->dv;    
    int smemSize = (REDUCE_THREADS <= 32) ? 2 * REDUCE_THREADS : REDUCE_THREADS;    

    if (!lfc->bloch_boundary_conditions) {


      const double* a_G = (const double*)a_xG_gpu;
      double* c_M = (double*)c_xM_gpu;
      
      smemSize*=sizeof(double);
      
      for (int x = 0; x < nx; x+=GPAW_CUDA_INT_BLOCKS) {
	int int_blocks=MIN(GPAW_CUDA_INT_BLOCKS,nx-x);
	//printf("int _blcoks %d\n",int_blocks);
	//int blockx=nextPow2(MAX((lfc->max_nG+1)/2,1));
	int blockx=MAX((lfc->max_nG+1)/2,1);
	int gridx=(lfc->nB_gpu+BLOCK_Y-1)/BLOCK_Y;
	//int smemSizeI = (blockx <= 32) ? 2 * blockx * sizeof(double) : blockx * sizeof(double);    
	dim3 dimBlock(blockx,BLOCK_Y); 
	dim3 dimGrid(gridx,int_blocks);   

	integrate_mul_kernel<<<dimGrid, dimBlock,0/*BLOCK_Y*smemSizeI*/>>>
	  (a_G+x*nG,lfc->G_B1_gpu,lfc->G_B2_gpu,lfc->volume_i_gpu,
	   lfc->A_gm_i_gpu,lfc->work_i_gpu,lfc->ni_gpu,lfc->nimax,nG,
	   lfc->phase_i_gpu,lfc->max_k,q,lfc->nB_gpu);
	gpaw_cudaSafeCall(cudaGetLastError());

	int swap=1;
            
	int iter=lfc->max_len_work;
      	while (iter>1) {
	  swap=!swap;
	  dim3 dimBlockr(REDUCE_THREADS, 1, 1);
	  dim3 dimGridr(MAX(iter/REDUCE_THREADS,1), lfc->nW*int_blocks);
	  integrate_reduce_kernel<<<dimGridr, dimBlockr, smemSize/**lfc->max_nm*/>>>
	    (lfc->volume_W_gpu,iter,swap,int_blocks,c_M+x*nM,lfc->nW,dv,nM);
	  iter=nextPow2(iter)/(REDUCE_THREADS*2);
	}
	gpaw_cudaSafeCall(cudaGetLastError());
	/*	dim3 dimBlockr(1,1, 1);
	dim3 dimGridr(int_blocks,1);
	integrate_get_c_xM<<<dimGridr, dimBlockr>>>(lfc->volume_W_gpu,swap,c_M+x*nM,lfc->nW,dv,nM);
	gpaw_cudaSafeCall(cudaGetLastError());*/
      }
    }
    else {

      const cuDoubleComplex* a_G = (const cuDoubleComplex*)a_xG_gpu;
      cuDoubleComplex* c_M = (cuDoubleComplex*)c_xM_gpu;

      smemSize*=sizeof(cuDoubleComplex);
      
      for (int x = 0; x < nx; x+=GPAW_CUDA_INT_BLOCKS) {
	int int_blocks=MIN(GPAW_CUDA_INT_BLOCKS,nx-x);
	int blockx=MAX((lfc->max_nG+1)/2,1);
	int gridx=(lfc->nB_gpu+BLOCK_Y-1)/BLOCK_Y;
	//int smemSizeI = (blockx <= 32) ? 2 * blockx * sizeof(cuDoubleComplex) : blockx * sizeof(cuDoubleComplex);    
	dim3 dimBlock(blockx,BLOCK_Y); 
	dim3 dimGrid(gridx,int_blocks);   

	integrate_mul_kernelz<<<dimGrid, dimBlock,0/* BLOCK_Y*smemSizeI*/>>>
	  (a_G+x*nG,lfc->G_B1_gpu,lfc->G_B2_gpu,lfc->volume_i_gpu,
	   lfc->A_gm_i_gpu,lfc->work_i_gpu,lfc->ni_gpu,lfc->nimax,nG,
	   lfc->phase_i_gpu,lfc->max_k,q,lfc->nB_gpu);
	
	gpaw_cudaSafeCall(cudaGetLastError());
	int swap=1;
	int iter=lfc->max_len_work;
      	while (iter>1) {
	  swap=!swap;
	  dim3 dimBlockr(REDUCE_THREADS, 1, 1);
	  dim3 dimGridr(MAX(iter/REDUCE_THREADS,1),lfc->nW*int_blocks);
	  integrate_reduce_kernelz<<<dimGridr, dimBlockr, smemSize/**lfc->max_nm*/>>>
	    (lfc->volume_W_gpu,iter,swap,int_blocks,c_M+x*nM,lfc->nW,dv,nM);
	  iter=nextPow2(iter)/(REDUCE_THREADS*2);
	}
	gpaw_cudaSafeCall(cudaGetLastError());
	/*	dim3 dimBlockr(1,1, 1);
	dim3 dimGridr(int_blocks,1);
	integrate_get_c_xMz<<<dimGridr, dimBlockr>>>(lfc->volume_W_gpu,swap,c_M+x*nM,lfc->nW,dv,nM);
	gpaw_cudaSafeCall(cudaGetLastError());*/

      }



    }
    Py_RETURN_NONE;
  }


  PyObject* add_cuda_gpu(LFCObject *lfc, PyObject *args)
  {

    CUdeviceptr a_xG_gpu,c_xM_gpu;
    PyObject *shape,*c_shape;
  
    int q;
  

    if (!PyArg_ParseTuple(args, "nOnOi", &c_xM_gpu,&c_shape, &a_xG_gpu,&shape, 
			  &q))
      return NULL; 

    int nd = PyTuple_Size(shape);

    int nx=PyInt_AsLong(PyTuple_GetItem(shape,0));
    for (int i=1;i<nd-3;i++)
      nx*=PyInt_AsLong(PyTuple_GetItem(shape,i));
  
    int nG=PyInt_AsLong(PyTuple_GetItem(shape,nd-3));
    for (int i=nd-2;i<nd;i++)
      nG*=PyInt_AsLong(PyTuple_GetItem(shape,i));

    int c_nd = PyTuple_Size(c_shape);
    int nM = PyInt_AsLong(PyTuple_GetItem(c_shape,c_nd-1));

    if (!lfc->bloch_boundary_conditions) {
      double* a_G = (double*)a_xG_gpu;
      const double* c_M = (const double*)c_xM_gpu;
      int blockx=lfc->max_nG;
      int gridx=(lfc->nB_gpu+BLOCK_Y-1)/BLOCK_Y;
      dim3 dimBlock(blockx,BLOCK_Y); 
      dim3 dimGrid(gridx,nx);   

      add_kernel<<<dimGrid, dimBlock, /*lfc->max_nm*sizeof(double)*/0>>>
	(a_G,c_M,lfc->G_B1_gpu,lfc->G_B2_gpu,lfc->volume_i_gpu,
	 lfc->A_gm_i_gpu,lfc->ni_gpu,lfc->nimax,nG,nM,lfc->phase_i_gpu,
	 lfc->max_k,q,lfc->nB_gpu);
      gpaw_cudaSafeCall(cudaGetLastError());
      
    }
    else {      
      cuDoubleComplex* a_G = (cuDoubleComplex*)a_xG_gpu;
      const cuDoubleComplex* c_M = (const cuDoubleComplex*)c_xM_gpu;

      int blockx=lfc->max_nG;
      int gridx=(lfc->nB_gpu+BLOCK_Y-1)/BLOCK_Y;
      dim3 dimBlock(blockx,BLOCK_Y); 
      dim3 dimGrid(gridx,nx);      

      add_kernelz<<<dimGrid, dimBlock, /*lfc->max_nm*sizeof(cuDoubleComplex)*/0>>>
	(a_G,c_M,lfc->G_B1_gpu,lfc->G_B2_gpu,lfc->volume_i_gpu,
	 lfc->A_gm_i_gpu,lfc->ni_gpu,lfc->nimax,nG,nM,lfc->phase_i_gpu,
	 lfc->max_k,q,lfc->nB_gpu);
      gpaw_cudaSafeCall(cudaGetLastError());
    }
    Py_RETURN_NONE;
  }

}


#endif
