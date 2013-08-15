#include <Python.h>
//#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
//#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <complex.h>

#include <sys/types.h>
#include <sys/time.h>

#include <cuComplex.h>



#include "gpaw-cuda-int.h"
#ifndef CUGPAWCOMPLEX

#define MBLAS_BLOCK_X  (128)
#define MAX_BLOCKS     (65535)
#define MIN_BLOCKS     (MAX_BLOCKS)

#endif

#define MAPNAME(f) Zcuda(f ## _dotu)
#define MAPFUNC(a,b) MULTT((a),(b))
#include "reduce.cu"
#undef MAPNAME
#undef MAPFUNC

#define MAPNAME(f) Zcuda(f ## _dotc)
#define MAPFUNC(a,b) MULTT(CONJ(a),(b))
#include "reduce.cu"
#undef MAPNAME
#undef MAPFUNC


__global__ void Zcuda(multi_scal_cuda_kernel)(int n,const Tcuda *alpha,Tcuda* a)
{
  int i = blockIdx.x*MBLAS_BLOCK_X+threadIdx.x;    
  int k=blockIdx.y;
  
  a += n*k;
  
  while (i < n) {
    a[i] = MULTT(a[i], alpha[k]);
    i += gridDim.x*MBLAS_BLOCK_X;
  }
}




__global__ void Zcuda(multi_axpy_cuda_kernel)(int n, const Tcuda *alpha, const Tcuda *a, 
					      Tcuda *b)
{

  int k = blockIdx.y;
  int i = blockIdx.x*MBLAS_BLOCK_X+threadIdx.x; 
  
  a += n*k;
  b += n*k;  
  while (i < n) {
      IADD(b[i], MULTT(a[i], alpha[k]));      
      i += gridDim.x*MBLAS_BLOCK_X;
  }
}


#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "mblas-cuda.cu"

extern "C" {

  
  PyObject* multi_scal_cuda_gpu(PyObject *self, PyObject *args)
  {
    CUdeviceptr alpha_gpu,x_gpu;
    PyObject *x_shape;
    PyArray_Descr *type,*a_type; 
    int nvec;
    
    if (!PyArg_ParseTuple(args, "nOnOO", &alpha_gpu, &a_type,
			  &x_gpu,&x_shape, &type))
      return NULL;
    
    int n = PyInt_AsLong(PyTuple_GetItem(x_shape,1));
    Py_ssize_t nd=PyTuple_Size(x_shape);
    for (int d = 2; d < nd; d++)
      n *= PyInt_AsLong(PyTuple_GetItem(x_shape,d));
    
    nvec= PyInt_AsLong(PyTuple_GetItem(x_shape,0));


    if (type->type_num == PyArray_DOUBLE){      
      int gridx = MIN(MAX((n+MBLAS_BLOCK_X-1)/MBLAS_BLOCK_X,1),MAX_BLOCKS);
      int gridy = nvec;
      
      dim3 dimBlock(MBLAS_BLOCK_X,1); 
      dim3 dimGrid(gridx,gridy);    
      
      multi_scal_cuda_kernel<<<dimGrid, dimBlock, 0>>>
	(n, (double *)alpha_gpu, (double*)x_gpu);
    } else if (a_type->type_num == PyArray_DOUBLE){
      double *alpha = (double*)(alpha_gpu);
      int gridx = MIN(MAX((2*n+MBLAS_BLOCK_X-1)/MBLAS_BLOCK_X,1),MAX_BLOCKS);
      int gridy = nvec;
      
      dim3 dimBlock(MBLAS_BLOCK_X,1); 
      dim3 dimGrid(gridx,gridy);    
            
      multi_scal_cuda_kernel<<<dimGrid, dimBlock, 0>>> 
	  (2*n, alpha, (double *)x_gpu);
    }else{
      cuDoubleComplex *alpha = (cuDoubleComplex*)(alpha_gpu);
      int gridx = MIN(MAX((n+MBLAS_BLOCK_X-1)/MBLAS_BLOCK_X,1),MAX_BLOCKS);
      int gridy = nvec;
      
      dim3 dimBlock(MBLAS_BLOCK_X,1); 
      dim3 dimGrid(gridx,gridy);    

      multi_scal_cuda_kernelz<<<dimGrid, dimBlock, 0>>> 
	(n, alpha, (cuDoubleComplex*)x_gpu);
      
    }
    gpaw_cudaSafeCall(cudaGetLastError());
    if (PyErr_Occurred())
      return NULL;
    else
      Py_RETURN_NONE;
  }

  

  
  PyObject* multi_axpy_cuda_gpu(PyObject *self, PyObject *args)
  {
    CUdeviceptr  alpha_gpu;
    CUdeviceptr x_gpu;
    CUdeviceptr y_gpu;
    PyObject *x_shape,*y_shape;
    PyArray_Descr *type,*a_type;
    int nvec;
    
    
    if (!PyArg_ParseTuple(args, "nOnOnOO", &alpha_gpu, &a_type,
			  &x_gpu,&x_shape, &y_gpu, &y_shape, &type))
      return NULL;
    
    int n = PyInt_AsLong(PyTuple_GetItem(x_shape,1));
    Py_ssize_t nd=PyTuple_Size(x_shape);
    for (int d = 2; d < nd; d++)
      n *= PyInt_AsLong(PyTuple_GetItem(x_shape,d));
    nvec= PyInt_AsLong(PyTuple_GetItem(x_shape,0));
    if (type->type_num == PyArray_DOUBLE){
      double *alpha = (double*)alpha_gpu;
      int gridx = MIN(MAX((n+MBLAS_BLOCK_X-1)/MBLAS_BLOCK_X,1),MAX_BLOCKS);
      int gridy = nvec;
      dim3 dimBlock(MBLAS_BLOCK_X,1); 
      dim3 dimGrid(gridx,gridy);       
   
      multi_axpy_cuda_kernel<<<dimGrid, dimBlock, 0>>> 
	(n, alpha, (double*)x_gpu, (double*)y_gpu);      
    } else  if (a_type->type_num == PyArray_DOUBLE){
      double *alpha = (double*)alpha_gpu;
      int gridx = MIN(MAX((2*n+MBLAS_BLOCK_X-1)/MBLAS_BLOCK_X,1),MAX_BLOCKS);
      int gridy = nvec;
      dim3 dimBlock(MBLAS_BLOCK_X,1); 
      dim3 dimGrid(gridx,gridy);       

      multi_axpy_cuda_kernel<<<dimGrid, dimBlock, 0>>> 
	(2*n, alpha,(double*)x_gpu,(double*)y_gpu);
    } else {
      cuDoubleComplex *alpha = (cuDoubleComplex*)alpha_gpu;
      int gridx = MIN(MAX((n+MBLAS_BLOCK_X-1)/MBLAS_BLOCK_X,1),MAX_BLOCKS);
      int gridy = nvec;
      dim3 dimBlock(MBLAS_BLOCK_X,1); 
      dim3 dimGrid(gridx,gridy);       

      multi_axpy_cuda_kernelz<<<dimGrid, dimBlock, 0>>> 
	  (n, alpha,(cuDoubleComplex*)x_gpu,(cuDoubleComplex*)y_gpu);
    }
    
    gpaw_cudaSafeCall(cudaGetLastError());
    if (PyErr_Occurred())
      return NULL;
    else
      Py_RETURN_NONE;
  }
  
  void mdotu_cuda_gpu( const double* a_gpu, const double* b_gpu,double *result,int n,int nvec)
  {
    reducemap_dotu((double*)a_gpu,(double*)b_gpu,result,n,nvec);
  }
  
  
  PyObject* multi_dotu_cuda_gpu(PyObject *self, PyObject *args)
  {
    
    CUdeviceptr a_gpu;
    CUdeviceptr b_gpu;
    CUdeviceptr res_gpu;
    
    PyObject *a_shape;
    PyArray_Descr *type;
    
    if (!PyArg_ParseTuple(args, "nOnOn", &a_gpu,&a_shape,&b_gpu,&type,&res_gpu))
      return NULL;
    int n = PyInt_AsLong(PyTuple_GetItem(a_shape,1));
    
    for (int i = 2; i < PyTuple_Size(a_shape); i++)
      n *= PyInt_AsLong(PyTuple_GetItem(a_shape,i));
    
    
    int nvec= PyInt_AsLong(PyTuple_GetItem(a_shape,0));
    if (type->type_num == PyArray_DOUBLE) {
      double *result = (double *)res_gpu;
      reducemap_dotu((double*)a_gpu, (double*)b_gpu, result, n, nvec);
    } else {
      cuDoubleComplex *result=(cuDoubleComplex *)res_gpu;
      reducemap_dotuz((cuDoubleComplex *)a_gpu, (cuDoubleComplex *)b_gpu,
		      result, n, nvec);
    }
    gpaw_cudaSafeCall(cudaGetLastError());
    if (PyErr_Occurred())
      return NULL;
    else
      Py_RETURN_NONE;
  }
  

  
  
  PyObject* multi_dotc_cuda_gpu(PyObject *self, PyObject *args)
  {
    
    CUdeviceptr a_gpu;
    CUdeviceptr b_gpu;
    CUdeviceptr res_gpu;
    
    PyObject *a_shape;
    PyArray_Descr *type;
    
    if (!PyArg_ParseTuple(args, "nOnOn", &a_gpu,&a_shape,&b_gpu,&type,&res_gpu))
      return NULL;
    int n = PyInt_AsLong(PyTuple_GetItem(a_shape,1));
    Py_ssize_t nd=PyTuple_Size(a_shape);
    for (int i = 2; i < nd; i++){
      n *= PyInt_AsLong(PyTuple_GetItem(a_shape,i));
      
    }
    int nvec= PyInt_AsLong(PyTuple_GetItem(a_shape,0));
    if (type->type_num == PyArray_DOUBLE) {
      
      double *result=(double *)res_gpu;
      reducemap_dotc((double*)a_gpu,(double*)b_gpu,result,n,nvec);

    } else {
      cuDoubleComplex *result=(cuDoubleComplex *)res_gpu;
      reducemap_dotcz((cuDoubleComplex*)a_gpu,(cuDoubleComplex*)b_gpu,
		      result,n,nvec);
    }
    gpaw_cudaSafeCall(cudaGetLastError());
    if (PyErr_Occurred())
      return NULL;
    else
      Py_RETURN_NONE;
  }
  
  
  
}

#endif
