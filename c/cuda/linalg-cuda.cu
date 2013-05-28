#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

#include <gpaw-cuda-int.h>
#ifndef CUGPAWCOMPLEX
#define BLOCK_X  128
#define MAX_BLOCKS  (65535)
#endif

__global__ void Zcuda(elmenwise_mul_add_kernelx)(int n,const double* a,const Tcuda* b,Tcuda *c)
{
  int i=blockIdx.x*BLOCK_X+threadIdx.x;
  while (i<n){
    IADD(c[i],MULDT(a[i],b[i]));
    i+=gridDim.x*BLOCK_X;
  }
}

__global__ void Zcuda(multi_elmenwise_mul_add_kernel1x)(int n,const double* a,const Tcuda* b,Tcuda *c)
{
  int i=blockIdx.x*BLOCK_X+threadIdx.x;
  int k=blockIdx.y;
  a+=k*n;
  c+=k*n;
  while (i<n){
    IADD(c[i],MULDT(a[i],b[i]));
    i+=gridDim.x*BLOCK_X;
  }
}

__global__ void Zcuda(multi_elmenwise_mul_add_kernel2x)(int n,const double* a,const Tcuda* b,Tcuda *c)
{
  int i=blockIdx.x*BLOCK_X+threadIdx.x;
  int k=blockIdx.y;
  b+=k*n;
  c+=k*n;
  while (i<n){
    IADD(c[i],MULDT(a[i],b[i]));
    i+=gridDim.x*BLOCK_X;
  }
}


__global__ void Zcuda(ax2py_kernel)(int n,double a,const Tcuda* x,double* y)
{
  int i=blockIdx.x*BLOCK_X+threadIdx.x;
  while (i<n){
    y[i]+=a*(REAL(x[i])*REAL(x[i])+IMAG(x[i])*IMAG(x[i]));
    i+=gridDim.x*BLOCK_X;
  }
}


__global__ void Zcuda(csign_kernel)(int n,Tcuda* x)
{
  int i=blockIdx.x*BLOCK_X+threadIdx.x;
  while (i<n){
    x[i]=NEG(x[i]);
    i+=gridDim.x*BLOCK_X;
  }
}


__global__ void Zcuda(multi_ax2py_kernel)(int n,int nvec,double *a,const Tcuda* x,double* y)
{
  int i=blockIdx.x*BLOCK_X+threadIdx.x;
  for (int k=0;k<nvec;k++) {
    while (i<n){
      y[i]+=a[k]*(REAL(x[i])*REAL(x[i])+IMAG(x[i])*IMAG(x[i]));
    }
    i+=gridDim.x*BLOCK_X;
    x+=n;
  }
}



#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "linalg-cuda.cu"

__global__ void elmenwise_mul_add_kernelzz(int n,const cuDoubleComplex* a,const cuDoubleComplex* b, cuDoubleComplex*c)
{
  int i=blockIdx.x*BLOCK_X+threadIdx.x;
  while (i<n){
    c[i]=cuCadd(c[i],cuCmul(a[i],b[i]));
    i+=gridDim.x*BLOCK_X;
  }
}


__global__ void multi_elmenwise_mul_add_kernel1zz(int n,const cuDoubleComplex* a,const cuDoubleComplex* b, cuDoubleComplex*c)
{
  int i=blockIdx.x*BLOCK_X+threadIdx.x;
  int k=blockIdx.y;
  a+=k*n;
  c+=k*n;
  while (i<n){
    c[i]=cuCadd(c[i],cuCmul(a[i],b[i]));
    i+=gridDim.x*BLOCK_X;
  }
}

__global__ void multi_elmenwise_mul_add_kernel2zz(int n,const cuDoubleComplex* a,const cuDoubleComplex* b, cuDoubleComplex*c)
{
  int i=blockIdx.x*BLOCK_X+threadIdx.x;
  int k=blockIdx.y;
  b+=k*n;
  c+=k*n;
  while (i<n){
    c[i]=cuCadd(c[i],cuCmul(a[i],b[i]));
    i+=gridDim.x*BLOCK_X;
  }
}

extern "C" {
  
  PyObject* elementwise_multiply_add_gpu(PyObject *self, PyObject *args)
  {		
    CUdeviceptr x_gpu,y_gpu,c_gpu;
    PyObject *a_shape;
    PyArray_Descr *a_type,*y_type; 
    
    if (!PyArg_ParseTuple(args, "nOOnOn", &x_gpu,&a_shape,&a_type,&y_gpu,&y_type,&c_gpu))
      return NULL;
    
    int n = PyInt_AsLong(PyTuple_GetItem(a_shape,0));
    Py_ssize_t nd=PyTuple_Size(a_shape);
    for (int d = 1; d < nd; d++)
      n *= PyInt_AsLong(PyTuple_GetItem(a_shape,d));
    
    int gridx=MIN(MAX((n+BLOCK_X-1)/BLOCK_X,1),MAX_BLOCKS);
    
    dim3 dimBlock(BLOCK_X,1); 
    dim3 dimGrid(gridx,1);    
    if (a_type->type_num == PyArray_DOUBLE){
      if (y_type->type_num == PyArray_DOUBLE){
	elmenwise_mul_add_kernelx<<<dimGrid, dimBlock, 0>>>
	  (n,(double*)x_gpu,(double*)y_gpu,(double*)c_gpu);
      }else{
	elmenwise_mul_add_kernelxz<<<dimGrid, dimBlock, 0>>>
	  (n,(double*)x_gpu,(cuDoubleComplex*)y_gpu,(cuDoubleComplex*)c_gpu);
      }
    } else {
      if (y_type->type_num == PyArray_DOUBLE){
	elmenwise_mul_add_kernelxz<<<dimGrid, dimBlock, 0>>>
	  (n,(double*)y_gpu,(cuDoubleComplex*)x_gpu,(cuDoubleComplex*)c_gpu);
	
      }else{
	elmenwise_mul_add_kernelzz<<<dimGrid, dimBlock, 0>>>
	  (n,(cuDoubleComplex*)x_gpu,(cuDoubleComplex*)y_gpu,(cuDoubleComplex*)c_gpu);
      }
    }
    gpaw_cudaSafeCall(cudaGetLastError());
    if (PyErr_Occurred())
      return NULL;
    else
      Py_RETURN_NONE;
  }


  PyObject* multi_elementwise_multiply_add_gpu(PyObject *self, PyObject *args)
  {		
    CUdeviceptr x_gpu,y_gpu,c_gpu;
    PyObject *x_shape,*y_shape,*shape;
    PyArray_Descr *x_type,*y_type; 
    
    if (!PyArg_ParseTuple(args, "nOOnOOn", &x_gpu,&x_shape,&x_type,&y_gpu,&y_shape,&y_type,&c_gpu))
      return NULL;
    
    Py_ssize_t x_nd=PyTuple_Size(x_shape);
    Py_ssize_t y_nd=PyTuple_Size(y_shape);


    shape=(x_nd>y_nd) ? x_shape : y_shape;


    int n = PyInt_AsLong(PyTuple_GetItem(shape,1));
    Py_ssize_t nd=PyTuple_Size(shape);
    for (int d = 2; d < nd; d++)
      n *= PyInt_AsLong(PyTuple_GetItem(shape,d));

    int nvec = PyInt_AsLong(PyTuple_GetItem(shape,0));
    
    int gridx=MIN(MAX((n+BLOCK_X-1)/BLOCK_X,1),MAX_BLOCKS);
    
    dim3 dimBlock(BLOCK_X,1); 
    dim3 dimGrid(gridx,nvec);
    
    if (x_type->type_num == PyArray_DOUBLE){
      if (y_type->type_num == PyArray_DOUBLE){
	if (x_nd>y_nd)
	  multi_elmenwise_mul_add_kernel1x<<<dimGrid, dimBlock, 0>>>	  
	    (n,(double*)x_gpu,(double*)y_gpu,(double*)c_gpu);
	else
	  multi_elmenwise_mul_add_kernel2x<<<dimGrid, dimBlock, 0>>>	  
	    (n,(double*)x_gpu,(double*)y_gpu,(double*)c_gpu);
      }else{	
	if (x_nd>y_nd)
	  multi_elmenwise_mul_add_kernel1xz<<<dimGrid, dimBlock, 0>>>
	    (n,(double*)x_gpu,(cuDoubleComplex*)y_gpu,(cuDoubleComplex*)c_gpu);
	else
	  multi_elmenwise_mul_add_kernel2xz<<<dimGrid, dimBlock, 0>>>
	    (n,(double*)x_gpu,(cuDoubleComplex*)y_gpu,(cuDoubleComplex*)c_gpu);
      }
    } else {
      if (y_type->type_num == PyArray_DOUBLE){
	if (y_nd>x_nd)
	  multi_elmenwise_mul_add_kernel1xz<<<dimGrid, dimBlock, 0>>>
	    (n,(double*)y_gpu,(cuDoubleComplex*)x_gpu,(cuDoubleComplex*)c_gpu);
	else
	  multi_elmenwise_mul_add_kernel2xz<<<dimGrid, dimBlock, 0>>>
	    (n,(double*)y_gpu,(cuDoubleComplex*)x_gpu,(cuDoubleComplex*)c_gpu);
      }else{
	if (x_nd>y_nd)
	  multi_elmenwise_mul_add_kernel1zz<<<dimGrid, dimBlock, 0>>>
	    (n,(cuDoubleComplex*)x_gpu,(cuDoubleComplex*)y_gpu,(cuDoubleComplex*)c_gpu);
	else
	  multi_elmenwise_mul_add_kernel2zz<<<dimGrid, dimBlock, 0>>>
	    (n,(cuDoubleComplex*)x_gpu,(cuDoubleComplex*)y_gpu,(cuDoubleComplex*)c_gpu);
      }
    }
    gpaw_cudaSafeCall(cudaGetLastError());
    if (PyErr_Occurred())
      return NULL;
    else
      Py_RETURN_NONE;
  }

  PyObject* ax2py_gpu(PyObject *self, PyObject *args)
  {

    double alpha;		
    CUdeviceptr x_gpu,y_gpu;
    PyObject *x_shape,y_shape;
    PyArray_Descr *type;
    
    if (!PyArg_ParseTuple(args, "dnOnOO", &alpha,&x_gpu,&x_shape,&y_gpu,
			  &y_shape,&type))
      return NULL;
    
    int n = PyInt_AsLong(PyTuple_GetItem(x_shape,0));
    Py_ssize_t nd=PyTuple_Size(x_shape);
    for (int d = 1; d < nd; d++)
      n *= PyInt_AsLong(PyTuple_GetItem(x_shape,d));
    
    int gridx=MIN(MAX((n+BLOCK_X-1)/BLOCK_X,1),MAX_BLOCKS);
    
    dim3 dimBlock(BLOCK_X,1); 
    dim3 dimGrid(gridx,1);    
    if (type->type_num == PyArray_DOUBLE){
      ax2py_kernel<<<dimGrid, dimBlock, 0>>>
	(n,alpha,(double*) x_gpu,(double*) y_gpu);

    } else {
      ax2py_kernelz<<<dimGrid, dimBlock, 0>>>
	(n,alpha,(Tcuda*) x_gpu,(double*) y_gpu);
    }
    gpaw_cudaSafeCall(cudaGetLastError());
    if (PyErr_Occurred())
      return NULL;
    else
      Py_RETURN_NONE;
  }


  PyObject* csign_gpu(PyObject *self, PyObject *args)
  {
    
    CUdeviceptr x_gpu;
    PyObject *x_shape;
    PyArray_Descr *type;
    
    if (!PyArg_ParseTuple(args, "nOO", &x_gpu,&x_shape,&type))
      return NULL;
    
    int n = PyInt_AsLong(PyTuple_GetItem(x_shape,0));
    Py_ssize_t nd=PyTuple_Size(x_shape);
    for (int d = 1; d < nd; d++)
      n *= PyInt_AsLong(PyTuple_GetItem(x_shape,d));
    
    int gridx=MIN(MAX((n+BLOCK_X-1)/BLOCK_X,1),MAX_BLOCKS);
    
    dim3 dimBlock(BLOCK_X,1); 
    dim3 dimGrid(gridx,1);    
    if (type->type_num == PyArray_DOUBLE){
      csign_kernel<<<dimGrid, dimBlock, 0>>>
	(n,(double*) x_gpu);

    } else {
      csign_kernelz<<<dimGrid, dimBlock, 0>>>
	(n,(Tcuda*) x_gpu);
    }
    gpaw_cudaSafeCall(cudaGetLastError());
    if (PyErr_Occurred())
      return NULL;
    else
      Py_RETURN_NONE;
  }
  



 PyObject* multi_ax2py_gpu(PyObject *self, PyObject *args)
  {
    CUdeviceptr  alpha_gpu;
    CUdeviceptr x_gpu;
    CUdeviceptr y_gpu;
    PyObject *x_shape,*y_shape;
    PyArray_Descr *type; 
    int nvec;
    
    
    if (!PyArg_ParseTuple(args, "nnOnOO", &alpha_gpu, 
			  &x_gpu,&x_shape, &y_gpu, &y_shape, &type))
      return NULL;
    
    int n = PyInt_AsLong(PyTuple_GetItem(x_shape,1));
    Py_ssize_t nd=PyTuple_Size(x_shape);
    for (int d = 2; d < nd; d++)
      n *= PyInt_AsLong(PyTuple_GetItem(x_shape,d));

    
    nvec= PyInt_AsLong(PyTuple_GetItem(x_shape,0));
    
    if (type->type_num == PyArray_DOUBLE){
      double *alpha=(double*)alpha_gpu;
      int gridx=MIN(MAX((n+BLOCK_X-1)/BLOCK_X,1),MAX_BLOCKS);
      int gridy=1;
      
      dim3 dimBlock(BLOCK_X,1); 
      dim3 dimGrid(gridx,gridy);    
      
      multi_ax2py_kernel<<<dimGrid, dimBlock, 0>>> 
	(n, nvec,alpha, (double*)x_gpu, (double*)y_gpu);
      
    } else {
      double *alpha=(double*)alpha_gpu;
      int gridx=MIN(MAX((n+BLOCK_X-1)/BLOCK_X,1),MAX_BLOCKS);
      int gridy=1;
      
      dim3 dimBlock(BLOCK_X,1); 
      dim3 dimGrid(gridx,gridy);    
      multi_ax2py_kernelz<<<dimGrid, dimBlock, 0>>> 
	(n, nvec, alpha,(cuDoubleComplex*)x_gpu,(double*)y_gpu);
    }
    gpaw_cudaSafeCall(cudaGetLastError());
    	
    if (PyErr_Occurred())
      return NULL;	
    else		
      Py_RETURN_NONE;
  }	
  
}
#endif
