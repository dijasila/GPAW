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


#define MBLAS_BLOCK_X  128



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
  int i=blockIdx.x*MBLAS_BLOCK_X+threadIdx.x;
  int k=blockIdx.y;
  a+=n*k+i;
  if (i<n)
    a[0]=MULTT(a[0],alpha[k]);
}




__global__ void Zcuda(multi_scalx_cuda_kernel)(int n,const double *alpha,Tcuda* a)
{
  int i=blockIdx.x*MBLAS_BLOCK_X+threadIdx.x;
  int k=blockIdx.y;
  a+=n*k+i;
  if (i<n)
    a[0]=MULTD(a[0],alpha[k]);
}


__global__ void Zcuda(multi_axpy_cuda_kernel)(int n,const Tcuda *alpha,const Tcuda *a,Tcuda *b)
{
  int i=blockIdx.x*MBLAS_BLOCK_X+threadIdx.x; 
  int k=blockIdx.y;
  a+=n*k+i;
  b+=n*k+i;
  if (i<n)
      IADD(b[0],MULTT(a[0],alpha[k]));      
}

__global__ void Zcuda(multi_axxpy_cuda_kernel)(int n,const double *alpha,const Tcuda *a,Tcuda *b)
{
  int i=blockIdx.x*MBLAS_BLOCK_X+threadIdx.x; 
  int k=blockIdx.y;
  a+=n*k+i;
  b+=n*k+i;
  if (i<n)
      IADD(b[0],MULTD(a[0],alpha[k]));      
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
    //int incx = 1;
    
    nvec= PyInt_AsLong(PyTuple_GetItem(x_shape,0));
    
    if (type->type_num == PyArray_DOUBLE){
      int gridx=MAX((n+MBLAS_BLOCK_X-1)/MBLAS_BLOCK_X,1);
      int gridy=nvec;
      
      dim3 dimBlock(MBLAS_BLOCK_X,1); 
      dim3 dimGrid(gridx,gridy);    
      multi_scal_cuda_kernel<<<dimGrid, dimBlock, 0>>>
	(n, (double *)alpha_gpu,(double*)x_gpu);
    } else {

      int gridx=MAX((n+MBLAS_BLOCK_X-1)/MBLAS_BLOCK_X,1);
      int gridy=nvec;
      
      dim3 dimBlock(MBLAS_BLOCK_X,1); 
      dim3 dimGrid(gridx,gridy);    
      if (a_type->type_num == PyArray_DOUBLE){
	double *alpha=(double*)(alpha_gpu);
	multi_scalx_cuda_kernelz<<<dimGrid, dimBlock, 0>>> 
	  (n, alpha,(cuDoubleComplex*)x_gpu);
      }else{
	cuDoubleComplex *alpha=(cuDoubleComplex*)(alpha_gpu);
	multi_scal_cuda_kernelz<<<dimGrid, dimBlock, 0>>> 
	  (n, alpha,(cuDoubleComplex*)x_gpu);
      }
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
    /*int incx = 1;
      int incy = 1;*/
    //int nvec = PyArray_SIZE(alpha_i);
    
    nvec= PyInt_AsLong(PyTuple_GetItem(x_shape,0));
    
    if (type->type_num == PyArray_DOUBLE){
      double *alpha=(double*)alpha_gpu;
      int gridx=MAX((n+MBLAS_BLOCK_X-1)/MBLAS_BLOCK_X,1);
      int gridy=nvec;
      
      dim3 dimBlock(MBLAS_BLOCK_X,1); 
      dim3 dimGrid(gridx,gridy);    
      
      multi_axpy_cuda_kernel<<<dimGrid, dimBlock, 0>>> 
	(n, alpha, (double*)x_gpu, (double*)y_gpu);
      
    } else {
      //printf("multi_zaxpy\n");

      int gridx=MAX((n+MBLAS_BLOCK_X-1)/MBLAS_BLOCK_X,1);
      int gridy=nvec;
      
      dim3 dimBlock(MBLAS_BLOCK_X,1); 
      dim3 dimGrid(gridx,gridy);    
      if (a_type->type_num == PyArray_DOUBLE){
	double *alpha=(double*)alpha_gpu;
	multi_axxpy_cuda_kernelz<<<dimGrid, dimBlock, 0>>> 
	  (n, alpha,(cuDoubleComplex*)x_gpu,(cuDoubleComplex*)y_gpu);
      } else {
	cuDoubleComplex *alpha=(cuDoubleComplex*)alpha_gpu;
	multi_axpy_cuda_kernelz<<<dimGrid, dimBlock, 0>>> 
	  (n, alpha,(cuDoubleComplex*)x_gpu,(cuDoubleComplex*)y_gpu);
      }
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
    
    PyArrayObject* res_i;
    CUdeviceptr a_gpu;
    CUdeviceptr b_gpu;
    
    PyObject *a_shape;
    PyArray_Descr *type;
    
    //    cuda_mblas_init();  
    
    if (!PyArg_ParseTuple(args, "nOnOO", &a_gpu,&a_shape,&b_gpu,&type,&res_i))
      return NULL;
    int n = PyInt_AsLong(PyTuple_GetItem(a_shape,1));
    
    for (int i = 2; i < PyTuple_Size(a_shape); i++)
      n *= PyInt_AsLong(PyTuple_GetItem(a_shape,i));
    
    /*  int incx = 1;
	int incy = 1;
    */
    //int nvec = PyArray_SIZE(res_i);
    
    int nvec= PyInt_AsLong(PyTuple_GetItem(a_shape,0));
    
    if (type->type_num == PyArray_DOUBLE) {
      double *result=(double*)PyArray_DATA(res_i);;
      reducemap_dotu((double*)a_gpu,(double*)b_gpu,result,n,nvec);
      /*    for (int i=0;i<nvec;i++){
	    cublasSetKernelStream(mblas_streams[i%MBLAS_BLOCKS]);
	    result[i] = cublasDdot(n, (double*)a_gpu+i*n,
	    incx, (double*)b_gpu+i*n, incy);
	    
	    }*/
    } else {
      cuDoubleComplex *result=(cuDoubleComplex*)PyArray_DATA(res_i);
      reducemap_dotuz((cuDoubleComplex*)a_gpu,(cuDoubleComplex*)b_gpu,
		      result,n,nvec);
      /*for (int i=0;i<nvec;i++){
	cublasSetKernelStream(mblas_streams[i%MBLAS_BLOCKS]);
	result[i] = cublasZdotu(n, (cuDoubleComplex*)a_gpu+i*n,
	incx, (cuDoubleComplex*)b_gpu+i*n, incy);
	}*/
    }
    gpaw_cudaSafeCall(cudaGetLastError());
    /*  cublasSetKernelStream(NULL);
	for (int i=0;i<MBLAS_BLOCKS;i++){
	cudaStreamSynchronize(mblas_streams[i]);
	}*/
    if (PyErr_Occurred())
      return NULL;
    else
      Py_RETURN_NONE;
  }
  

  
  
  PyObject* multi_dotc_cuda_gpu(PyObject *self, PyObject *args)
  {
    
    PyArrayObject* res_i;
    CUdeviceptr a_gpu;
    CUdeviceptr b_gpu;
    
    PyObject *a_shape;
    PyArray_Descr *type;
    
    // cuda_mblas_init();  
    
    if (!PyArg_ParseTuple(args, "nOnOO", &a_gpu,&a_shape,&b_gpu,&type,&res_i))
      return NULL;
    int n = PyInt_AsLong(PyTuple_GetItem(a_shape,1));
    Py_ssize_t nd=PyTuple_Size(a_shape);
    for (int i = 2; i < nd; i++){
      n *= PyInt_AsLong(PyTuple_GetItem(a_shape,i));
      
    }
    /*  int incx = 1;
	int incy = 1;
    */
    //  int nvec = PyArray_SIZE(res_i);
    int nvec= PyInt_AsLong(PyTuple_GetItem(a_shape,0));
    
    
    if (type->type_num == PyArray_DOUBLE) {
      
      double *result=(double*)PyArray_DATA(res_i);
      reducemap_dotc((double*)a_gpu,(double*)b_gpu,result,n,nvec);
      
      
      /*
	for (int i=0;i<nvec;i++){
	cublasSetKernelStream(mblas_streams[i%MBLAS_BLOCKS]);
	result[i] = cublasDdot(n, (double*)a_gpu+i*n,
	incx, (double*)b_gpu+i*n, incy);
	
	}
      */
    } else {
      cuDoubleComplex *result=(cuDoubleComplex*)PyArray_DATA(res_i);
      reducemap_dotcz((cuDoubleComplex*)a_gpu,(cuDoubleComplex*)b_gpu,
		      result,n,nvec);
      /*    for (int i=0;i<nvec;i++){
	    cublasSetKernelStream(mblas_streams[i%MBLAS_BLOCKS]);
	    result[i] = cublasZdotc(n, (cuDoubleComplex*)a_gpu+i*n,
	    incx, (cuDoubleComplex*)b_gpu+i*n, incy);
	    }*/
    }
    gpaw_cudaSafeCall(cudaGetLastError());
    /* cublasSetKernelStream(NULL);
       for (int i=0;i<MBLAS_BLOCKS;i++){
       cudaStreamSynchronize(mblas_streams[i]);
       }*/
    if (PyErr_Occurred())
      return NULL;
    else
      Py_RETURN_NONE;
  }
  
  
  
}

#endif
