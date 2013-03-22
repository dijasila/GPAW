
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


#define BLOCK_Y 16

#include "cuda.h"
#include "cuda_runtime_api.h"

#endif


#include "lfc-reduce.cu"


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
	cudaFree(volume_gpu->GB1);	
	cudaFree(volume_gpu->nGBcum);	
	cudaFree(volume_gpu->phase_k);	

      }
      free(self->volume_W_cuda);
      cudaFree(self->volume_W_gpu);
      cudaFree(self->G_B1_gpu);
      cudaFree(self->G_B2_gpu);
      cudaFree(self->volume_i_gpu);
      cudaFree(self->A_gm_i_gpu);
      cudaFree(self->volume_WMi_gpu);
      cudaFree(self->WMi_gpu);
      cudaFree(self->ni_gpu);
      cudaGetLastError();
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
    int *GB2s[self->nW];
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
      v_gpu->len_A_gm=0;
      v_gpu->GB1 = GPAW_MALLOC(int, self->ngm_W[W] );
      GB2s[W] = GPAW_MALLOC(int, self->ngm_W[W] );
      v_gpu->nGBcum = GPAW_MALLOC(int, self->ngm_W[W]+1 );
      v_gpu->nB=0;     
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

    
    if (self->bloch_boundary_conditions){
      phase_i_gpu = GPAW_MALLOC(cuDoubleComplex,max_k*self->nB*nimax);
      phase_i = GPAW_MALLOC(cuDoubleComplex,max_k*nimax);      
    }

    int nB_gpu=0;

    for (int B = 0; B < self->nB; B++) {
      int Gb = self->G_B[B]; 
      int nG = Gb - Ga; 	

      if ((nG > 0) && (ni>0)) {	
	//printf("Ni %d",ni);
	for (int i = 0; i < ni; i++) {
	  LFVolume_gpu* v = volume_i[i];
	  volume_i_gpu[nB_gpu*nimax+i]=self->volume_W_gpu+(v-volume_W_gpu);
	  A_gm_i_gpu[nB_gpu*nimax+i]=v->len_A_gm;
	  if (self->bloch_boundary_conditions){
	    for (int kk=0;kk<max_k;kk++){	      
	      phase_i_gpu[i+nB_gpu*nimax*max_k+kk*nimax]=phase_i[i+kk*nimax];
	    }
	  }	  
	  v->len_A_gm+=nG;
	  int *GB2=GB2s[v-volume_W_gpu];
	  if ((v->nB > 0) && (GB2[v->nB-1]==Ga)) {
	    GB2[v->nB-1]=Gb;
	    v->nGBcum[v->nB]+=nG;
	  } else {
	    v->GB1[v->nB]=Ga;
	    GB2[v->nB]=Gb;
	  
	  
	    if (v->nB == 0)
	      v->nGBcum[v->nB]=0;
	  
	    v->nGBcum[v->nB+1]=nG + v->nGBcum[v->nB];
	    v->nB++;
	  }
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
    //printf("\n");
    for (int W = 0; W < self->nW; W++){
      LFVolume_gpu* v = &volume_W_gpu[W];
      self->max_len_A_gm=MAX(self->max_len_A_gm,v->len_A_gm);

      int *GB_gpu;
      GPAW_CUDAMALLOC(&(GB_gpu),int,v->nB);
      GPAW_CUDAMEMCPY(GB_gpu,v->GB1,int,v->nB,
		      cudaMemcpyHostToDevice);
      free(v->GB1);
      v->GB1=GB_gpu;
      free(GB2s[W]);

      GPAW_CUDAMALLOC(&(GB_gpu),int,v->nB+1);
      GPAW_CUDAMEMCPY(GB_gpu,v->nGBcum,int,v->nB+1,
		      cudaMemcpyHostToDevice);
      free(v->nGBcum);
      v->nGBcum=GB_gpu;

      if (self->bloch_boundary_conditions){
	cuDoubleComplex phase_k[max_k];
	for (int q=0;q<max_k;q++) {
	  phase_k[q].x=creal(self->phase_kW[self->nW*q+W]);
	  phase_k[q].y=cimag(self->phase_kW[self->nW*q+W]);
	}

	GPAW_CUDAMALLOC(&(v->phase_k),cuDoubleComplex,max_k);
	GPAW_CUDAMEMCPY(v->phase_k,phase_k,
			cuDoubleComplex,max_k,cudaMemcpyHostToDevice);
      }      

    }

    int WMimax=0;
    int *WMi_gpu=GPAW_MALLOC(int,self->nW);
    int *volume_WMi_gpu=GPAW_MALLOC(int,self->nW*self->nW);

    self->Mcount=0;
    for (int W = 0; W < self->nW; W++) {
      WMi_gpu[W]=0;
    }
    for (int W = 0; W < self->nW; W++) {
      LFVolume_gpu*  v = &volume_W_gpu[W];

      int M=v->M;
      for (int W2 = 0; W2 <= W   ; W2++) {
	if (WMi_gpu[W2]>0) {
	  LFVolume_gpu*  v2 = &volume_W_gpu[volume_WMi_gpu[W2*self->nW]];
	  if (v2->M==M) {	    
	    volume_WMi_gpu[W2*self->nW+WMi_gpu[W2]]=W;
	    WMi_gpu[W2]++;
	    WMimax=MAX(WMi_gpu[W2],WMimax);
	    break;
	  }
	} else {
	  volume_WMi_gpu[W2*self->nW]=W;
	  WMi_gpu[W2]++;
	  self->Mcount++;
	  WMimax=MAX(WMi_gpu[W2],WMimax);
	  break;
	}
      }
    }
    int *volume_WMi_gpu2=GPAW_MALLOC(int,WMimax*self->nW);
    for (int W = 0; W < self->Mcount; W++) {
      for (int W2 = 0; W2 < WMi_gpu[W]; W2++) {
	volume_WMi_gpu2[W*WMimax+W2]=volume_WMi_gpu[W*self->nW+W2];
      }
    }
    self->WMimax=WMimax;

    GPAW_CUDAMALLOC(&(self->WMi_gpu),int,self->Mcount);
    GPAW_CUDAMEMCPY(self->WMi_gpu,WMi_gpu,int,self->Mcount,
		    cudaMemcpyHostToDevice);
    
    GPAW_CUDAMALLOC(&(self->volume_WMi_gpu),int,self->Mcount*WMimax);
    GPAW_CUDAMEMCPY(self->volume_WMi_gpu,volume_WMi_gpu2,int,self->Mcount*WMimax,
		    cudaMemcpyHostToDevice);

    
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
    self->volume_W_cuda=volume_W_gpu;


    GPAW_CUDAMEMCPY(self->volume_W_gpu,volume_W_gpu,LFVolume_gpu,self->nW,
		    cudaMemcpyHostToDevice);
    free(volume_i);
    free(volume_i_gpu);
    free(A_gm_i_gpu);
    free(volume_WMi_gpu);
    free(volume_WMi_gpu2);
    free(WMi_gpu);
    free(ni_gpu);
    free(G_B1_gpu);
    free(G_B2_gpu);
    if (self->bloch_boundary_conditions){
      free(phase_i_gpu);
    }
    if (PyErr_Occurred())
      return NULL;
    else
      return (PyObject*)self;
  }

  
  PyObject* integrate_cuda_gpu(LFCObject *lfc, PyObject *args)
  {
    
    CUdeviceptr a_xG_gpu,c_xM_gpu;
    PyObject *shape,*c_shape;
  
    int q;
    
    assert(lfc->cuda);
  
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
    //  double dv = lfc->dv;    

    if (!lfc->bloch_boundary_conditions) {
      const double* a_G = (const double*)a_xG_gpu;
      double* c_M = (double*)c_xM_gpu;
      
      lfc_reducemap(lfc,a_G,nG,c_M,nM,nx,q);
      gpaw_cudaSafeCall(cudaGetLastError());
    }
    else {      
      const cuDoubleComplex* a_G = (const cuDoubleComplex*)a_xG_gpu;
      cuDoubleComplex* c_M = (cuDoubleComplex*)c_xM_gpu;
      
      lfc_reducemapz(lfc,a_G,nG,c_M,nM,nx,q);
      gpaw_cudaSafeCall(cudaGetLastError());
    }
    if (PyErr_Occurred())
      return NULL;
    else
      Py_RETURN_NONE;
  }


  PyObject* add_cuda_gpu(LFCObject *lfc, PyObject *args)
  {

    CUdeviceptr a_xG_gpu,c_xM_gpu;
    PyObject *shape,*c_shape;
  
    int q;
  
    assert(lfc->cuda);
    
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
    if (PyErr_Occurred())
      return NULL;
    else
      Py_RETURN_NONE;
  }
  
}


#endif
