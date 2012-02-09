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

#define MBLAS_BLOCKS 16
#define MAX_MBLAS_BLOCKS 128

#define BLOCK_X  128


__constant__ double c_alpha[MAX_MBLAS_BLOCKS];

cudaStream_t mblas_streams[MBLAS_BLOCKS];
int mblas_initialized = 0;

void cuda_mblas_init()
{
  if  (!mblas_initialized){
    for (int i=0;i<MBLAS_BLOCKS;i++){
      cudaStreamCreate(&(mblas_streams[i]));
    }
    mblas_initialized=1;
  }
}


#endif



__global__ void Zcuda(multi_scal_cuda_kernel)(int n,const Tcuda *alpha,Tcuda* a)
{
  int ibl=blockIdx.y;
  int i=blockIdx.x*BLOCK_X+threadIdx.x;
  Tcuda *alp=(Tcuda*)c_alpha;
  //Tcuda A,apl;
  
  a+=i+ibl*n;
  if (i<n){
    a[0]=MULTT(a[0],alp[ibl]);
    //a[0]=MULTT(a[0],alpha[ibl]);
  }

}

__global__ void Zcuda(multi_axpy_cuda_kernel)(int n,const Tcuda *alpha,const Tcuda *a,Tcuda *b)
{
  int ibl=blockIdx.y;
  int i=blockIdx.x*BLOCK_X+threadIdx.x; 
  Tcuda *alp=(Tcuda*)c_alpha;

  a+=i+ibl*n;
  b+=i+ibl*n;
  if (i<n){
    IADD(b[0],MULTT(a[0],alp[ibl]));
    //   IADD(b[0],MULTT(a[0],alpha[ibl]));

  }
}

extern "C" {

  double Zcuda(multi_scal_cuda_cpu)(const Tcuda *alpha,Tcuda *x,int n,int nvec)
  {

    struct timeval  t0, t1; 
    double flops;

    Tcuda *xdev;
    Tcuda *alphadev;
    
    

    int xsize=n*nvec;    
    gpaw_cudaSafeCall(cudaGetLastError());
    gpaw_cudaSafeCall(cudaMalloc(&xdev,sizeof(Tcuda)*xsize));
    
    gpaw_cudaSafeCall(cudaMemcpy(xdev,x,sizeof(Tcuda)*xsize,
				 cudaMemcpyHostToDevice));
    gpaw_cudaSafeCall(cudaMalloc(&alphadev,sizeof(Tcuda)*nvec));
    
    gpaw_cudaSafeCall(cudaMemcpy(alphadev,alpha,sizeof(Tcuda)*nvec,
				 cudaMemcpyHostToDevice));
    gpaw_cudaSafeCall(cudaGetLastError());
    gettimeofday(&t0,NULL);  
    
    /*    for (int i=0;i<nvec;i+=MAX_MBLAS_BLOCKS/2){
	  int myvec=MIN(MAX_MBLAS_BLOCKS/2,nvec-i);*/
    int myvec=nvec;
    int gridx=MAX((n+BLOCK_X-1)/BLOCK_X,1);
    int gridy=myvec;
    
    dim3 dimBlock(BLOCK_X,1); 
    dim3 dimGrid(gridx,gridy);    
    /*      gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_alpha,alphadev+i,
	    sizeof(Tcuda)*myvec,
	    0,
	    cudaMemcpyDeviceToDevice));*/
    
    //printf("mult_scal nvec %d\n",nvec);
    
    
    
    Zcuda(multi_scal_cuda_kernel)<<<dimGrid, dimBlock, 0>>> 
      (n, alphadev+0,xdev+0*n);
    //    }
    cudaThreadSynchronize(); 
    gpaw_cudaSafeCall(cudaGetLastError());
    
    gettimeofday(&t1,NULL);
    gpaw_cudaSafeCall(cudaMemcpy(x,xdev,sizeof(Tcuda)*xsize,
				 cudaMemcpyDeviceToHost));
    
    gpaw_cudaSafeCall(cudaFree(xdev));
    gpaw_cudaSafeCall(cudaFree(alphadev));
    
    flops=(t1.tv_sec*1.0+t1.tv_usec/1000000.0-t0.tv_sec*1.0-t0.tv_usec/1000000.0); 
    //   }
    
    /*
      for (int i=0;i<nvec;i++){
      cublasSetKernelStream(mblas_streams[i%MBLAS_BLOCKS]);
      cublasZscal(n, alpha[i],
      (cuDoubleComplex*)x_gpu+i*n, incx);
      
      }*/
    
    //  cublasSetKernelStream(NULL);
    return flops;
  }
  

   
  double Zcuda(multi_scal_cuda_cpu2)(const Tcuda *alpha,Tcuda *x,int n,int nvec)
  {

    struct timeval  t0, t1; 
    double flops;

    Tcuda *xdev;
    Tcuda *alphadev;
    
    //    for (int i=0;i<nvec;i+=MAX_MBLAS_BLOCKS/2){
    //int myvec=MIN(MAX_MBLAS_BLOCKS/2,nvec-i);
    int gridx=MAX((n+BLOCK_X-1)/BLOCK_X,1);
    int gridy=nvec;
    
    dim3 dimBlock(BLOCK_X,1); 
    dim3 dimGrid(gridx,gridy);    
    /*gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_alpha,alpha+i,
      sizeof(cuDoubleComplex)*myvec,
      sizeof(cuDoubleComplex)*i,
      cudaMemcpyDeviceToDevice));
    */
    //printf("mult_scal nvec %d\n",nvec);

    //cuda_mblas_init();  
    int xsize=n*nvec;
    
    gpaw_cudaSafeCall(cudaGetLastError());
    gpaw_cudaSafeCall(cudaMalloc(&xdev,sizeof(Tcuda)*xsize));
    
    gpaw_cudaSafeCall(cudaMemcpy(xdev,x,sizeof(Tcuda)*xsize,
				 cudaMemcpyHostToDevice));
    gpaw_cudaSafeCall(cudaMalloc(&alphadev,sizeof(Tcuda)*nvec));
    
    gpaw_cudaSafeCall(cudaMemcpy(alphadev,alpha,sizeof(Tcuda)*nvec,
				 cudaMemcpyHostToDevice));
    gpaw_cudaSafeCall(cudaGetLastError());
    gettimeofday(&t0,NULL);  

    for (int i=0;i<nvec;i++){
      //      cublasSetKernelStream(mblas_streams[i%MBLAS_BLOCKS]);
#ifndef CUGPAWCOMPLEX
      cublasDscal(n, alpha[i],
		  (Tcuda*)xdev+i*n, 1);
#else
      cublasZscal(n, alpha[i],
		  (Tcuda*)xdev+i*n, 1);
#endif
      
    }
    /*
    Zcuda(multi_scal_cuda_kernel)<<<dimGrid, dimBlock, 0>>> 
      (n, alphadev+0,xdev+0*n);
    */
    cudaThreadSynchronize(); 
    gpaw_cudaSafeCall(cudaGetLastError());
    
    gettimeofday(&t1,NULL);
    gpaw_cudaSafeCall(cudaMemcpy(x,xdev,sizeof(Tcuda)*xsize,
				 cudaMemcpyDeviceToHost));
    
    gpaw_cudaSafeCall(cudaFree(xdev));
    gpaw_cudaSafeCall(cudaFree(alphadev));

    flops=(t1.tv_sec*1.0+t1.tv_usec/1000000.0-t0.tv_sec*1.0-t0.tv_usec/1000000.0); 
      //   }
    
    

    // }
    //    cublasSetKernelStream(NULL);
    return flops;
  } // }

}


#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "mblas-cuda.cu"

extern "C" {

  
  PyObject* multi_scal_cuda_gpu(PyObject *self, PyObject *args)
  {
    PyArrayObject* alpha_i;
    //    CUdeviceptr alpha_gpu;
    CUdeviceptr x_gpu;
    PyObject *x_shape;
    PyArray_Descr *type; 
    int nvec;
    
    if (!PyArg_ParseTuple(args, "OnOOi", &alpha_i, 
			  &x_gpu,&x_shape, &type,&nvec))
      return NULL;
    
    int n = PyInt_AsLong(PyTuple_GetItem(x_shape,1));
    Py_ssize_t nd=PyTuple_Size(x_shape);
    for (int d = 2; d < nd; d++)
      n *= PyInt_AsLong(PyTuple_GetItem(x_shape,d));
    int incx = 1;
    // int nvec = PyArray_SIZE(alpha_i);
    
    assert(nvec== PyInt_AsLong(PyTuple_GetItem(x_shape,0)));
    
    if (type->type_num == PyArray_DOUBLE){
      double *alpha=(double*)(alpha_i->data);
      //double *alpha=(double*)alpha_gpu;
      for (int i=0;i<nvec;i+=MAX_MBLAS_BLOCKS){
	int myvec=MIN(MAX_MBLAS_BLOCKS,nvec-i);
	int gridx=MAX((n+BLOCK_X-1)/BLOCK_X,1);
	int gridy=myvec;
	
	dim3 dimBlock(BLOCK_X,1); 
	dim3 dimGrid(gridx,gridy);    
	gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_alpha,alpha+i,
					     sizeof(double)*myvec,
					     0,
					     cudaMemcpyHostToDevice));
	
	multi_scal_cuda_kernel<<<dimGrid, dimBlock, 0>>>(n, alpha+i,(double*)x_gpu+i*n);
      }
      /*
	for (int i=0;i<nvec;i++){
	cublasSetKernelStream(mblas_streams[i%MBLAS_BLOCKS]);
	cublasDscal(n, alpha[i],
	(double*)x_gpu+i*n, incx);	       
	}
      */
    } else {
      cuDoubleComplex *alpha=(cuDoubleComplex*)(alpha_i->data);
      //cuDoubleComplex *alpha=(cuDoubleComplex*)alpha_gpu;
      for (int i=0;i<nvec;i+=MAX_MBLAS_BLOCKS/2){
	int myvec=MIN(MAX_MBLAS_BLOCKS/2,nvec-i);
	int gridx=MAX((n+BLOCK_X-1)/BLOCK_X,1);
	int gridy=myvec;
	
	dim3 dimBlock(BLOCK_X,1); 
	dim3 dimGrid(gridx,gridy);    
	gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_alpha,alpha+i,
					     sizeof(cuDoubleComplex)*myvec,
					     0,
					     cudaMemcpyHostToDevice));	
	multi_scal_cuda_kernelz<<<dimGrid, dimBlock, 0>>> 
	  (n, alpha+i,(cuDoubleComplex*)x_gpu+i*n);
      }
	
	/*
	for (int i=0;i<nvec;i++){
	cublasSetKernelStream(mblas_streams[i%MBLAS_BLOCKS]);
	cublasZscal(n, alpha[i],
	(cuDoubleComplex*)x_gpu+i*n, incx);
	
	}*/
    }
    gpaw_cublasSafeCall(cublasGetError());
    //  cublasSetKernelStream(NULL);
    
    Py_RETURN_NONE;
}
  

  
PyObject* multi_axpy_cuda_gpu(PyObject *self, PyObject *args)
{
  PyArrayObject* alpha_i;
  //CUdeviceptr  alpha_gpu;
  CUdeviceptr x_gpu;
  CUdeviceptr y_gpu;
  PyObject *x_shape,*y_shape;
  PyArray_Descr *type; 
  int nvec;

  //cuda_mblas_init();  
  
  if (!PyArg_ParseTuple(args, "OnOnOOi", &alpha_i, 
			&x_gpu,&x_shape, &y_gpu, &y_shape, &type, &nvec))
    return NULL;

  int n = PyInt_AsLong(PyTuple_GetItem(x_shape,1));
  Py_ssize_t nd=PyTuple_Size(x_shape);
  for (int d = 2; d < nd; d++)
    n *= PyInt_AsLong(PyTuple_GetItem(x_shape,d));
  int incx = 1;
  int incy = 1;
  //int nvec = PyArray_SIZE(alpha_i);

  assert(nvec== PyInt_AsLong(PyTuple_GetItem(x_shape,0)));

  if (type->type_num == PyArray_DOUBLE){
    //    double *alpha=(double*)alpha_gpu;
    double *alpha=(double*)(alpha_i->data);
    for (int i=0;i<nvec;i+=MAX_MBLAS_BLOCKS){
      int myvec=MIN(MAX_MBLAS_BLOCKS,nvec-i);
      int gridx=MAX((n+BLOCK_X-1)/BLOCK_X,1);
      int gridy=myvec;
      
      dim3 dimBlock(BLOCK_X,1); 
      dim3 dimGrid(gridx,gridy);    
      gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_alpha,alpha+i,
					   sizeof(double)*myvec,
					   0,
					   cudaMemcpyHostToDevice));
      
      multi_axpy_cuda_kernel<<<dimGrid, dimBlock, 0>>> 
	(n, alpha+i, (double*)x_gpu+i*n, (double*)y_gpu+i*n);
    }
    /*
    for (int i=0;i<nvec;i++){
      cublasSetKernelStream(mblas_streams[i%MBLAS_BLOCKS]);
      cublasDaxpy(n, alpha[i],
		  (double*)x_gpu+i*n, incx,
		  (double*)y_gpu+i*n, incy);
		  }*/
  } else {
    //cuDoubleComplex *alpha=(cuDoubleComplex*)alpha_gpu;
    cuDoubleComplex *alpha=(cuDoubleComplex*)(alpha_i->data);
    for (int i=0;i<nvec;i+=MAX_MBLAS_BLOCKS/2){
      int myvec=MIN(MAX_MBLAS_BLOCKS/2,nvec-i);
      int gridx=MAX((n+BLOCK_X-1)/BLOCK_X,1);
      int gridy=myvec;
      
      dim3 dimBlock(BLOCK_X,1); 
      dim3 dimGrid(gridx,gridy);    
      gpaw_cudaSafeCall(cudaMemcpyToSymbol(c_alpha,alpha+i,
					   sizeof(cuDoubleComplex)*myvec,
					   0,
					   cudaMemcpyHostToDevice));      
      multi_axpy_cuda_kernelz<<<dimGrid, dimBlock, 0>>> 
	(n, alpha+i,(cuDoubleComplex*)x_gpu+i*n,(cuDoubleComplex*)y_gpu+i*n);
    }
    
    /*
    for (int i=0;i<nvec;i++){
      cublasSetKernelStream(mblas_streams[i%MBLAS_BLOCKS]);
      cublasZaxpy(n, alpha[i],
		  (cuDoubleComplex*)x_gpu+i*n, incx,
		  (cuDoubleComplex*)y_gpu+i*n, incy);

    }
    */
  }
  gpaw_cublasSafeCall(cublasGetError());
  //cublasSetKernelStream(NULL);
  
  Py_RETURN_NONE;
}

PyObject* multi_dotu_cuda_gpu(PyObject *self, PyObject *args)
{

  PyArrayObject* res_i;
  CUdeviceptr a_gpu;
  CUdeviceptr b_gpu;
  
  PyObject *a_shape;
  PyArray_Descr *type;

  cuda_mblas_init();  
  
  if (!PyArg_ParseTuple(args, "nOnOO", &a_gpu,&a_shape,&b_gpu,&type,&res_i))
    return NULL;
  int n = PyInt_AsLong(PyTuple_GetItem(a_shape,1));
  
  for (int i = 2; i < PyTuple_Size(a_shape); i++)
    n *= PyInt_AsLong(PyTuple_GetItem(a_shape,i));

  int incx = 1;
  int incy = 1;

  int nvec = PyArray_SIZE(res_i);

  assert(nvec== PyInt_AsLong(PyTuple_GetItem(a_shape,0)));
  
  if (type->type_num == PyArray_DOUBLE) {
    double *result=(double*)PyArray_DATA(res_i);;
    for (int i=0;i<nvec;i++){
      cublasSetKernelStream(mblas_streams[i%MBLAS_BLOCKS]);
      result[i] = cublasDdot(n, (double*)a_gpu+i*n,
			     incx, (double*)b_gpu+i*n, incy);
      
    }
  } else {
    cuDoubleComplex *result=(cuDoubleComplex*)PyArray_DATA(res_i);
    for (int i=0;i<nvec;i++){
      cublasSetKernelStream(mblas_streams[i%MBLAS_BLOCKS]);
      result[i] = cublasZdotu(n, (cuDoubleComplex*)a_gpu+i*n,
			      incx, (cuDoubleComplex*)b_gpu+i*n, incy);
    }
  }
  gpaw_cublasSafeCall(cublasGetError());
  cublasSetKernelStream(NULL);
  for (int i=0;i<MBLAS_BLOCKS;i++){
    cudaStreamSynchronize(mblas_streams[i]);
  }
  Py_RETURN_NONE;
}


PyObject* multi_dotc_cuda_gpu(PyObject *self, PyObject *args)
{

  PyArrayObject* res_i;
  CUdeviceptr a_gpu;
  CUdeviceptr b_gpu;
  
  PyObject *a_shape;
  PyArray_Descr *type;

  cuda_mblas_init();  
  
  if (!PyArg_ParseTuple(args, "nOnOO", &a_gpu,&a_shape,&b_gpu,&type,&res_i))
    return NULL;
  int n = PyInt_AsLong(PyTuple_GetItem(a_shape,1));
  
  for (int i = 2; i < PyTuple_Size(a_shape); i++)
    n *= PyInt_AsLong(PyTuple_GetItem(a_shape,i));

  int incx = 1;
  int incy = 1;

  int nvec = PyArray_SIZE(res_i);

  assert(nvec== PyInt_AsLong(PyTuple_GetItem(a_shape,0)));
  
  if (type->type_num == PyArray_DOUBLE) {
    double *result=(double*)PyArray_DATA(res_i);;
    for (int i=0;i<nvec;i++){
      cublasSetKernelStream(mblas_streams[i%MBLAS_BLOCKS]);
      result[i] = cublasDdot(n, (double*)a_gpu+i*n,
			     incx, (double*)b_gpu+i*n, incy);
      
    }
  } else {
    cuDoubleComplex *result=(cuDoubleComplex*)PyArray_DATA(res_i);
    for (int i=0;i<nvec;i++){
      cublasSetKernelStream(mblas_streams[i%MBLAS_BLOCKS]);
      result[i] = cublasZdotc(n, (cuDoubleComplex*)a_gpu+i*n,
			      incx, (cuDoubleComplex*)b_gpu+i*n, incy);
    }
  }
  gpaw_cublasSafeCall(cublasGetError());
  cublasSetKernelStream(NULL);
  for (int i=0;i<MBLAS_BLOCKS;i++){
    cudaStreamSynchronize(mblas_streams[i]);
  }
  Py_RETURN_NONE;
}



}

#endif
