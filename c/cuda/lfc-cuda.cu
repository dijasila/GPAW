
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


#define BLOCK_SIZEX 16
#define BLOCK_SIZEY 1

#define REDUCE_THREADS 64

#define TRANS_BLOCK 16

#define MAX(a,b) ((a)>(b))?(a):(b)
#define MIN(a,b) ((a)<(b))?(a):(b)

#define WARP      32
#define HALFWARP  (WARP/2)

#include "cuda.h"
#include "cuda_runtime_api.h"

static inline unsigned int nextPow2( unsigned int x ) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

#endif




/*
static unsigned int nextHWarp( unsigned int x){
  
  if (x<(HALFWARP/2))
    return  nextPow2(x);
  else
    return HALFWARP*((x+HALFWARP-1)/HALFWARP);
}
*/


__global__ void Zcuda(integrate_reduce_kernel)(LFVolume_gpu* volume_W,int max_n,int swap)
{
  extern __shared__ Tcuda Zcuda(sdata)[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory

  unsigned int tid = threadIdx.x;
    
  int w=blockIdx.y;
  
  
  LFVolume_gpu *v_gpu;
  int nm,len;
  Tcuda  *work1,*work2;
  
  v_gpu=&volume_W[w]; 
  nm=v_gpu->nm;
  len=v_gpu->len_A_gm;
  if (!swap) {
    work1=(Tcuda*)v_gpu->work1_A_gm;
    work2=(Tcuda*)v_gpu->work2_A_gm;
  } else {
    work1=(Tcuda*)v_gpu->work2_A_gm;
    work2=(Tcuda*)v_gpu->work1_A_gm;
  }

  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;     
  int n=min(len,max_n);
  Tcuda * g_idata=work1/*+m*len*/;
  Tcuda * g_odata=work2/*+m*len*/;
  for (int m=0;m<nm;m++){
      
    Tcuda mySum = (i < n) ? g_idata[i] : MAKED(0);
    if (i + blockDim.x < n) 
      IADD(mySum , g_idata[i+blockDim.x]);  
      
    Zcuda(sdata)[tid] = mySum;
    __syncthreads();
      
    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
      {
	if (tid < s) 
	  {
	    Zcuda(sdata)[tid] = mySum = ADD(mySum , Zcuda(sdata)[tid + s]);
	  }
	__syncthreads();
      }
    // write result for this block to global mem 
    __syncthreads();
    if ((tid == 0) && (blockIdx.x<n)){
      g_odata[blockIdx.x] = Zcuda(sdata)[0];
      //      if (gridDim.x==1){
	//	c_xM[v_gpu->M+m]=MULTD(Zcuda(sdata)[0] , dv);	
	//	IADD(c_xM[v_gpu->M+m],MULTD(Zcuda(sdata)[0] , 1));	
      //}
    }
    //}
    g_idata+=len;
    g_odata+=len;
    __syncthreads();
  }
      
}

__global__ void Zcuda(integrate_get_c_xM)(LFVolume_gpu* volume_W,int swap,Tcuda *c_xM,int nW,double dv)
{
  Tcuda  *work;
  LFVolume_gpu *v_gpu;
  int nm,len;
  Tcuda *c_M;

  for (int w=0;w<nW;w++){
    v_gpu=&volume_W[w]; 
    if (!swap) {
      work=(Tcuda*)v_gpu->work2_A_gm;
    } else {
      work=(Tcuda*)v_gpu->work1_A_gm;
    }
    len=v_gpu->len_A_gm;
    nm=v_gpu->nm;
    c_M=c_xM+v_gpu->M;
    for(int m=0;m<nm;m++){
      IADD(c_M[m],MULTD(work[0] , dv));
      work+=len;
    }
  }
}




#if 0

__global__ void Zcuda(integrate_cuda_kernel)(const double *a_G, LFVolume_gpu* volume_W,double dv)
  
{

  int w=blockIdx.y;
  int G=threadIdx.x;
  int i=blockIdx.x;    

  __shared__ LFVolume_gpu *v_gpu;
  __shared__ int nm,nB;

  
  if (G==0){
    v_gpu=&volume_W[w]; 
    nm=v_gpu->nm;
    nB=v_gpu->nB;
  }
  __syncthreads();

  if (i>=nB) return;

  //int m=threadIdx.y;
  __shared__ int Ga,Gb,GNB,nG,len;
  __shared__ double  *A_gms,*work;  
  
  
  if (G==0) {
    Ga=v_gpu->G_B[2*i];
    Gb=v_gpu->G_B[2*i+1];
    nG=Gb-Ga;
    GNB=v_gpu->G_NB[i];
    A_gms=v_gpu->A_gm+GNB;
    work=v_gpu->work1_A_gm+GNB;
    len=v_gpu->len_A_gm;
  }
  __syncthreads();
  

  
  
  double *A_gm=A_gms+G;
  double *work_A_gm=work+G;

  if (G<nG){  
    double aG=dv*a_G[Ga+G];
    for (int m=0;m<nm;m++){
    
      //  if (m<nm){
      //    for(G=threadIdx.x;G<nG;G+=BLOCK_SIZEX)
    
      work_A_gm[0] =aG * A_gm[0];
      //work_A_gm[0] =1;
      A_gm+=len;
      work_A_gm+=len;
    }
  }
  
  //work_A_gm[G+GNB] =dv * a_G[Ga+G] * A_gm[nm*(G+GNB)];
  //}
  //return;
}
#endif

__global__ void Zcuda(integrate_mul_kernel)(const Tcuda *a_G,int *G_B,LFVolume_gpu **volume_i,double **A_gm_i,int *ni,int nimax,cuDoubleComplex *phase_i,int max_k,int q)
  
{

  int G=threadIdx.x;
  int B=blockIdx.x;

  int nii,Gb,Ga,nG;
  LFVolume_gpu* v;
  double* A_gm;
  int nm;
  Tcuda *work;
  int len;
  
  
  nii=ni[B]; 
  Ga=G_B[2*B];
  Gb=G_B[2*B+1];
  nG=Gb-Ga;
  if ( G<nG ){
    Tcuda av = a_G[Ga+G];      
    for(int i=0;i<nii;i++){         
      Tcuda avv=av;
      
      v = volume_i[nimax*B+i];
      A_gm = A_gm_i[nimax*B+i]+G;
      nm = v->nm;
      work=(Tcuda*)(v->work1_A_gm)+(A_gm - v->A_gm);
      len=v->len_A_gm;
#ifdef CUGPAWCOMPLEX
      avv=MULTT(avv, phase_i[max_k*nimax*B+q*nimax+i]);
#endif
      work[0]= MULTD(avv , A_gm[0]); 
      for (int m = 1; m < nm; m+=2){
	A_gm+=len;
	work+=len;	
	work[0]= MULTD(avv , A_gm[0]); 
	A_gm+=len;
	work+=len;	
	work[0]= MULTD(avv , A_gm[0]); 	
      }
    }
  }
}


__global__ void Zcuda(add_kernel)(Tcuda *a_G,const Tcuda *c_M,int *G_B,LFVolume_gpu **volume_i,double **A_gm_i,int *ni,int nimax,int na_G,int nM,cuDoubleComplex *phase_i,int max_k,int q)
  
{

  int G=threadIdx.x;
  int B=blockIdx.x;
  int x=blockIdx.y;
  
  int nii,Gb,Ga,nG;
  LFVolume_gpu* v;
  double* A_gm;
  const Tcuda* c_Mt;
  int nm;
  int len;
   

  nii=ni[B]; 
  Ga=G_B[2*B];
  Gb=G_B[2*B+1];
  nG=Gb-Ga;
  a_G+=Ga+na_G*x;
  c_M+=nM*x;  
  if ( G<nG ){

    Tcuda av=MAKED(0);//=a_G[G];
    for(int i=0;i<nii;i++){     
      Tcuda avv;
      v = volume_i[nimax*B+i];
      A_gm = A_gm_i[nimax*B+i]+G;
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
	gpaw_cudaSafeCall(cudaFree(volume_gpu->work1_A_gm));
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
      cudaFree(self->G_B_gpu);
      cudaFree(self->volume_i_gpu);
      cudaFree(self->A_gm_i_gpu);
      cudaFree(self->ni_gpu);
    }
  }
  
  PyObject * NewLFCObject_cuda(LFCObject *self, PyObject *args)
  {
    PyObject* A_Wgm_obj;
    const PyArrayObject* M_W_obj;
    const PyArrayObject* G_B_obj;
    const PyArrayObject* W_B_obj;
    //PyObject* G_WB_to_gpu_obj=NULL;
    double dv;
    const PyArrayObject* phase_kW_obj;
    int cuda=1;
    
    
    if (!PyArg_ParseTuple(args, "OOOOdO|iO",
			  &A_Wgm_obj, &M_W_obj, &G_B_obj, &W_B_obj, &dv,
			  &phase_kW_obj, &cuda))
      return NULL; 
    

    if (!cuda)     return (PyObject*)self;

    // printf("New lfc object cuda\n");

    self->cuda=cuda;

    int max_A=0;
    int max_k=0;

    LFVolume_gpu* volume_W_gpu;
    volume_W_gpu = GPAW_MALLOC(LFVolume_gpu, self->nW);

    if (self->bloch_boundary_conditions){
      max_k=phase_kW_obj->dimensions[0];
    }    
    self->max_k=max_k;
    LFVolume_gpu* volume_gpu =NULL;
    self->max_len_A_gm=0;
    for (int W = 0; W < self->nW; W++) {
      volume_gpu = &volume_W_gpu[W];
      LFVolume* volume = &self->volume_W[W];

      const PyArrayObject* A_gm_obj =				\
	(const PyArrayObject*)PyList_GetItem(A_Wgm_obj, W);
      
      double *work_A_gm = GPAW_MALLOC(double, self->ngm_W[W]);

      gpaw_cudaSafeCall(cudaMalloc(&(volume_gpu->A_gm),
				   sizeof(double)*self->ngm_W[W]));   

      for (int i1=0;i1<A_gm_obj->dimensions[0];i1++)
	for (int i2=0;i2<A_gm_obj->dimensions[1];i2++)
	  work_A_gm[i1+i2*A_gm_obj->dimensions[0]]=volume->A_gm[i1*A_gm_obj->dimensions[1]+i2];
      
      

      gpaw_cudaSafeCall(cudaMemcpy(volume_gpu->A_gm,work_A_gm,
				   sizeof(double)*self->ngm_W[W],
				   cudaMemcpyHostToDevice));
      if (self->bloch_boundary_conditions){
	gpaw_cudaSafeCall(cudaMalloc(&(volume_gpu->work1_A_gm),
				     sizeof(cuDoubleComplex)*self->ngm_W[W]));
	gpaw_cudaSafeCall(cudaMalloc(&(volume_gpu->work2_A_gm),
				     sizeof(cuDoubleComplex)*self->ngm_W[W]));
      } else {
	gpaw_cudaSafeCall(cudaMalloc(&(volume_gpu->work1_A_gm),
				     sizeof(double)*self->ngm_W[W]));
	gpaw_cudaSafeCall(cudaMalloc(&(volume_gpu->work2_A_gm),
				     sizeof(double)*self->ngm_W[W]));
      }

      if (A_gm_obj->dimensions[0]>max_A) max_A=A_gm_obj->dimensions[0];
      
      volume_gpu->nm=volume->nm;
      volume_gpu->M=volume->M;
      volume_gpu->W=volume->W;
      
      volume_gpu->len_A_gm=A_gm_obj->dimensions[0];

      if (volume_gpu->len_A_gm>self->max_len_A_gm) self->max_len_A_gm=volume_gpu->len_A_gm;
      free(work_A_gm);

    }

    gpaw_cudaSafeCall(cudaMalloc(&(self->volume_W_gpu),
				 sizeof(LFVolume_gpu)*self->nW));

    
    int* G_B = self->G_B; 
    int* W_B = self->W_B; 
    int* i_W = self->i_W; 
    LFVolume_gpu** volume_i = GPAW_MALLOC(LFVolume_gpu*, self->nimax);  
    LFVolume_gpu* volume_W = volume_W_gpu; 
    int Ga = 0; 
    int ni = 0; 
    self->max_nG=0;

    LFVolume_gpu **volume_i_gpu=GPAW_MALLOC(LFVolume_gpu*,self->nB*self->nimax);
    double **A_gm_i_gpu=GPAW_MALLOC(double*,self->nB*self->nimax);
    int *ni_gpu=GPAW_MALLOC(int,self->nB);
    int *G_B_gpu=GPAW_MALLOC(int,2*self->nB);

    cuDoubleComplex *phase_i_gpu;
    cuDoubleComplex *phase_i;
    if (self->bloch_boundary_conditions){
      phase_i_gpu = GPAW_MALLOC(cuDoubleComplex,max_k*self->nB*self->nimax);
      phase_i = GPAW_MALLOC(cuDoubleComplex,max_k*self->nimax);      
    }

    self->nB_gpu=0;
    for (int B = 0; B < self->nB; B++) {
      int Gb = G_B[B]; 
      int nG = Gb - Ga; 	
      if (nG > 0) {	
	for (int i = 0; i < ni; i++) {
	  LFVolume_gpu* v = volume_i[i];
	  if (nG>self->max_nG) self->max_nG=nG;	  
	  volume_i_gpu[self->nB_gpu*self->nimax+i]=self->volume_W_gpu+(v-volume_W_gpu);
	  A_gm_i_gpu[self->nB_gpu*self->nimax+i]=v->A_gm;
	  if (self->bloch_boundary_conditions){
	    for (int kk=0;kk<max_k;kk++){

	      phase_i_gpu[i+self->nB_gpu*self->nimax*max_k+kk*self->nimax]=phase_i[i+kk*self->nimax];
	    }
	  }
	  
	}
	if (ni>0) {
	  G_B_gpu[2*self->nB_gpu]=Ga;
	  G_B_gpu[2*self->nB_gpu+1]=Gb;
	  ni_gpu[self->nB_gpu]=ni;
	  self->nB_gpu++;

	}
	for (int i = 0; i < ni; i++) 
	  volume_i[i]->A_gm += nG/* * volume_i[i].nm*/;
      } 
      int Wnew = W_B[B];
      if (Wnew >= 0)	  { 
	/* Entering new sphere: */
	volume_i[ni] = &volume_W[Wnew];
	if (self->bloch_boundary_conditions){
	  for (int i=0;i<max_k;i++){
	    phase_i[ni+i*self->nimax].x=creal(self->phase_kW[Wnew+i*self->nW]);
	    phase_i[ni+i*self->nimax].y=cimag(self->phase_kW[Wnew+i*self->nW]);
	  }
	}
	i_W[Wnew] = ni; 
	
	ni++;
      } else { 
	/* Leaving sphere: */
	int Wold = -1 - Wnew;
	int iold = i_W[Wold];
	volume_W[Wold].A_gm = volume_i[iold]->A_gm;
	ni--;
	volume_i[iold] = volume_i[ni];
	if (self->bloch_boundary_conditions){
	  for (int i=0;i<max_k;i++){
	    phase_i[iold+i*self->nimax]=phase_i[ni+i*self->nimax]; 
	  }
	}

	int Wlast = volume_i[iold]->W;
	i_W[Wlast] = iold;
      } 
      Ga = Gb; 
    } 
    for (int W = 0; W < self->nW; W++)
      volume_W[W].A_gm -= volume_W[W].len_A_gm;
    
    //    self->nimax=nimax;
    // printf("nB %d nB_gpu %d nimax %d\n",self->nB,self->nB_gpu,nimax);



    gpaw_cudaSafeCall(cudaMalloc(&(self->G_B_gpu),sizeof(int)*2*self->nB_gpu));
    gpaw_cudaSafeCall(cudaMemcpy(self->G_B_gpu,G_B_gpu,
				 sizeof(int)*2*self->nB_gpu,
				 cudaMemcpyHostToDevice));

    gpaw_cudaSafeCall(cudaMalloc(&(self->ni_gpu),sizeof(int)*self->nB_gpu));
    gpaw_cudaSafeCall(cudaMemcpy(self->ni_gpu,ni_gpu,sizeof(int)*self->nB_gpu,
				 cudaMemcpyHostToDevice));

    gpaw_cudaSafeCall(cudaMalloc(&(self->volume_i_gpu),
				 sizeof(LFVolume_gpu*)*self->nB_gpu*self->nimax));
    gpaw_cudaSafeCall(cudaMemcpy(self->volume_i_gpu,volume_i_gpu,
				 sizeof(LFVolume_gpu*)*self->nB_gpu*self->nimax,cudaMemcpyHostToDevice));
    
    gpaw_cudaSafeCall(cudaMalloc(&(self->A_gm_i_gpu),
				 sizeof(double*)*self->nB_gpu*self->nimax));
    gpaw_cudaSafeCall(cudaMemcpy(self->A_gm_i_gpu,A_gm_i_gpu,
				 sizeof(double*)*self->nB_gpu*self->nimax,
				 cudaMemcpyHostToDevice));
    
    if (self->bloch_boundary_conditions){
      gpaw_cudaSafeCall(cudaMalloc(&(self->phase_i_gpu),
				   sizeof(cuDoubleComplex)*max_k*self->nB_gpu*self->nimax));
      gpaw_cudaSafeCall(cudaMemcpy(self->phase_i_gpu,phase_i_gpu,
				   sizeof(cuDoubleComplex)*max_k*self->nB_gpu*self->nimax,cudaMemcpyHostToDevice));
      
    }

    self->volume_W_cuda=volume_W_gpu;


    gpaw_cudaSafeCall(cudaMemcpy(self->volume_W_gpu,volume_W_gpu,
				 sizeof(LFVolume_gpu)*self->nW,
				 cudaMemcpyHostToDevice));
    free(volume_i);
    free(volume_i_gpu);
    free(A_gm_i_gpu);
    free(ni_gpu);
    free(G_B_gpu);
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
  
    if (!PyArg_ParseTuple(args, "nOnOi", &a_xG_gpu,&shape, &c_xM_gpu,&c_shape, &q))
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

    if (!lfc->bloch_boundary_conditions) {
      const double* a_G = (const double*)a_xG_gpu;
      double* c_M = (double*)c_xM_gpu;
      
      int smemSize = (REDUCE_THREADS <= 32) ? 2 * REDUCE_THREADS * sizeof(double) : REDUCE_THREADS * sizeof(double);    
      for (int x = 0; x < nx; x++) {
	
	dim3 dimBlock(lfc->max_nG,1); 
	dim3 dimGrid(lfc->nB_gpu,1);   

	integrate_mul_kernel<<<dimGrid, dimBlock, 0>>>(a_G+x*nG,lfc->G_B_gpu,lfc->volume_i_gpu,lfc->A_gm_i_gpu,lfc->ni_gpu,lfc->nimax,lfc->phase_i_gpu,lfc->max_k,q);
	gpaw_cudaSafeCall(cudaGetLastError());

	int swap=1;
            
	int iter=lfc->max_len_A_gm;
      	while (iter>1) {
	  swap=!swap;
	  dim3 dimBlockr(REDUCE_THREADS, /*lfc->max_nm*/1, 1);
	  dim3 dimGridr(MAX(iter/REDUCE_THREADS,1), lfc->nW);
	  integrate_reduce_kernel<<<dimGridr, dimBlockr, smemSize/**lfc->max_nm*/>>>(lfc->volume_W_gpu,iter,swap);
	  iter=nextPow2(iter)/(REDUCE_THREADS*2);
	  gpaw_cudaSafeCall(cudaGetLastError());
	}

	dim3 dimBlockr(1,1, 1);
	dim3 dimGridr(1,1);
	integrate_get_c_xM<<<dimGridr, dimBlockr>>>(lfc->volume_W_gpu,swap,c_M+x*nM,lfc->nW,dv);
	gpaw_cudaSafeCall(cudaGetLastError());
      }
    }
    else {
      const cuDoubleComplex* a_G = (const cuDoubleComplex*)a_xG_gpu;
      cuDoubleComplex* c_M = (cuDoubleComplex*)c_xM_gpu;
      
      int smemSize = (REDUCE_THREADS <= 32) ? 2 * REDUCE_THREADS * sizeof(cuDoubleComplex) : REDUCE_THREADS * sizeof(cuDoubleComplex);    
      for (int x = 0; x < nx; x++) {
	
	dim3 dimBlock(lfc->max_nG,1); 
	dim3 dimGrid(lfc->nB_gpu,1);   

	integrate_mul_kernelz<<<dimGrid, dimBlock, 0>>>(a_G+x*nG,lfc->G_B_gpu,lfc->volume_i_gpu,lfc->A_gm_i_gpu,lfc->ni_gpu,lfc->nimax,lfc->phase_i_gpu,lfc->max_k,q);
	
	gpaw_cudaSafeCall(cudaGetLastError());
	int swap=1;
	int iter=lfc->max_len_A_gm;
      	while (iter>1) {
	  swap=!swap;
	  dim3 dimBlockr(REDUCE_THREADS, /*lfc->max_nm*/1, 1);
	  dim3 dimGridr(MAX(iter/REDUCE_THREADS,1), lfc->nW);
	  integrate_reduce_kernelz<<<dimGridr, dimBlockr, smemSize/**lfc->max_nm*/>>>(lfc->volume_W_gpu,iter,swap);
	  iter=nextPow2(iter)/(REDUCE_THREADS*2);
	   gpaw_cudaSafeCall(cudaGetLastError());
	}

	dim3 dimBlockr(1,1, 1);
	dim3 dimGridr(1,1);
	integrate_get_c_xMz<<<dimGridr, dimBlockr>>>(lfc->volume_W_gpu,swap,c_M+x*nM,lfc->nW,dv);
	gpaw_cudaSafeCall(cudaGetLastError());

      }



    }
    Py_RETURN_NONE;
    //  #endif
#if 0
    if (!lfc->bloch_boundary_conditions) {
      const double* a_G = (const double*)a_xG_obj->data;
      double* c_M = (double*)c_xM_obj->data;
      for (int x = 0; x < nx; x++) {
	int* G_B = lfc->G_B; 
	int* W_B = lfc->W_B; 
	int* i_W = lfc->i_W; 
	LFVolume* volume_i = lfc->volume_i; 
	LFVolume* volume_W = lfc->volume_W; 
	int Ga = 0; 
	int ni = 0; 
	for (int B = 0; B < lfc->nB; B++) {
	  int Gb = G_B[B]; 
	  int nG = Gb - Ga; 	
	  //	fprintf(stdout,"nx %d lfc ni %d Gb %d lfc->nB %d lfc->nW %d Ga %d nm %d nG %d\n",nx,ni,Gb,lfc->nB,lfc->nW,Ga,v->nm,nG);
	  if (nG > 0) {	
	    for (int i = 0; i < ni; i++) {
	 	
	      LFVolume* v = volume_i + i;
	      const double* A_gm = v->A_gm;
	      int nm = v->nm;
	      double* c_M1 = c_M + v->M;

	      for (int gm = 0, G = Ga; G <Gb; G++){
		double av = a_G[G] * dv;
		for (int m = 0; m < nm; m++, gm++){
		  c_M1[m] += av * A_gm[gm];
		}
	      }
	    }
	    for (int i = 0; i < ni; i++) 
	      volume_i[i].A_gm += nG * volume_i[i].nm;
	  } 
	  int Wnew = W_B[B];
	  //fprintf(stdout,"Wnew %d\n",Wnew);
	  if (Wnew >= 0)	  { 
	    /* Entering new sphere: */
	    volume_i[ni] = volume_W[Wnew];
	    i_W[Wnew] = ni; 
	    ni++;
	  } else { 
	    /* Leaving sphere: */
	    int Wold = -1 - Wnew;
	    int iold = i_W[Wold];
	    volume_W[Wold].A_gm = volume_i[iold].A_gm;
	    ni--;
	    volume_i[iold] = volume_i[ni];
	    int Wlast = volume_i[iold].W;
	    i_W[Wlast] = iold;
	  } 
	  Ga = Gb; 
	} 
	for (int W = 0; W < lfc->nW; W++)
	  volume_W[W].A_gm -= lfc->ngm_W[W]; 
      
	c_M += nM;
	a_G += nG;
      }
    }
    else {
      assert(0);
    }
    Py_RETURN_NONE;
#endif
  }


  PyObject* add_cuda_gpu(LFCObject *lfc, PyObject *args)
  {

    CUdeviceptr a_xG_gpu,c_xM_gpu;
    PyObject *shape,*c_shape;
  
    int q;
  

    if (!PyArg_ParseTuple(args, "nOnOi", &c_xM_gpu,&c_shape, &a_xG_gpu,&shape, &q))
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

      dim3 dimBlock(lfc->max_nG,1); 
      dim3 dimGrid(lfc->nB_gpu,nx);   
      add_kernel<<<dimGrid, dimBlock, /*lfc->max_nm*sizeof(double)*/0>>>(a_G,c_M,lfc->G_B_gpu,lfc->volume_i_gpu,lfc->A_gm_i_gpu,lfc->ni_gpu,lfc->nimax,nG,nM,lfc->phase_i_gpu,lfc->max_k,q);
      gpaw_cudaSafeCall(cudaGetLastError());
      
    }
    else {      
      cuDoubleComplex* a_G = (cuDoubleComplex*)a_xG_gpu;
      const cuDoubleComplex* c_M = (const cuDoubleComplex*)c_xM_gpu;

      dim3 dimBlock(lfc->max_nG,1); 
      dim3 dimGrid(lfc->nB_gpu,nx);   
      add_kernelz<<<dimGrid, dimBlock, /*lfc->max_nm*sizeof(cuDoubleComplex)*/0>>>(a_G,c_M,lfc->G_B_gpu,lfc->volume_i_gpu,lfc->A_gm_i_gpu,lfc->ni_gpu,lfc->nimax,nG,nM,lfc->phase_i_gpu,lfc->max_k,q);
      gpaw_cudaSafeCall(cudaGetLastError());
    }
    Py_RETURN_NONE;
  }

}


#endif
