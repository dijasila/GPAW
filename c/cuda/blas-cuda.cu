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
#endif


__global__ void Zcuda(elmenwise_mul_add_kernelx)(int n,const double* a,const Tcuda* b,Tcuda *c)
{
  int i=blockIdx.x*BLOCK_X+threadIdx.x;
  a+=i;
  b+=i;
  c+=i;
  if (i<n){
    IADD(c[0],MULDT(a[0],b[0]));
  }
}




#ifndef CUGPAWCOMPLEX
#define CUGPAWCOMPLEX
#include "blas-cuda.cu"

__global__ void elmenwise_mul_add_kernelzz(int n,const cuDoubleComplex* a,const cuDoubleComplex* b, cuDoubleComplex*c)
{
  int i=blockIdx.x*BLOCK_X+threadIdx.x;
  a+=i;
  b+=i;
  c+=i;
  if (i<n){
    c[0]=cuCadd(c[0],cuCmul(a[0],b[0]));
  }
}

extern "C" {

PyObject* elementwise_multiply_add_gpu(PyObject *self, PyObject *args)
{
  CUdeviceptr a_gpu,b_gpu,c_gpu;
  PyObject *a_shape;
  PyArray_Descr *a_type,*b_type; 
  
  if (!PyArg_ParseTuple(args, "nOOnOn", &a_gpu,&a_shape,&a_type,&b_gpu,&b_type,&c_gpu))
    return NULL;

  int n = PyInt_AsLong(PyTuple_GetItem(a_shape,0));
  Py_ssize_t nd=PyTuple_Size(a_shape);
  for (int d = 1; d < nd; d++)
    n *= PyInt_AsLong(PyTuple_GetItem(a_shape,d));
  
  int gridx=MAX((n+BLOCK_X-1)/BLOCK_X,1);
  
  dim3 dimBlock(BLOCK_X,1); 
  dim3 dimGrid(gridx,1);    
  if (a_type->type_num == PyArray_DOUBLE){
    if (b_type->type_num == PyArray_DOUBLE){
      elmenwise_mul_add_kernelx<<<dimGrid, dimBlock, 0>>>
	(n,(double*)a_gpu,(double*)b_gpu,(double*)c_gpu);
    }else{
      elmenwise_mul_add_kernelxz<<<dimGrid, dimBlock, 0>>>
	(n,(double*)a_gpu,(cuDoubleComplex*)b_gpu,(cuDoubleComplex*)c_gpu);
    }
  } else {
    if (b_type->type_num == PyArray_DOUBLE){
      elmenwise_mul_add_kernelxz<<<dimGrid, dimBlock, 0>>>
	(n,(double*)b_gpu,(cuDoubleComplex*)a_gpu,(cuDoubleComplex*)c_gpu);
      
    }else{
      elmenwise_mul_add_kernelzz<<<dimGrid, dimBlock, 0>>>
	(n,(cuDoubleComplex*)a_gpu,(cuDoubleComplex*)b_gpu,(cuDoubleComplex*)c_gpu);
    }
  }
  gpaw_cublasSafeCall(cublasGetError());
  Py_RETURN_NONE;
}
  

PyObject* scal_cuda_gpu(PyObject *self, PyObject *args)
{
  Py_complex alpha;

  CUdeviceptr x_gpu;
  PyObject *x_shape;
  PyArray_Descr *type; 


  if (!PyArg_ParseTuple(args, "DnOO", &alpha, &x_gpu,&x_shape,&type))
    return NULL;

  int n = PyInt_AsLong(PyTuple_GetItem(x_shape,0));
  Py_ssize_t nd=PyTuple_Size(x_shape);
  for (int d = 1; d < nd; d++)
    n *= PyInt_AsLong(PyTuple_GetItem(x_shape,d));
  int incx = 1;
  if (type->type_num == PyArray_DOUBLE)
    cublasDscal(n, alpha.real,
		(double*)x_gpu, incx);		
  else {
    cuDoubleComplex alpha_gpu={alpha.real,alpha.imag};
    cublasZscal(n, alpha_gpu,
		(cuDoubleComplex*)x_gpu, incx);
  }
  gpaw_cublasSafeCall(cublasGetError());

  Py_RETURN_NONE;
}




PyObject* gemm_cuda_gpu(PyObject *self, PyObject *args)
{
  Py_complex alpha;
  Py_complex beta;

  CUdeviceptr a_gpu;
  CUdeviceptr b_gpu;
  CUdeviceptr c_gpu;
  PyObject *a_shape,*b_shape,*c_shape;
  PyArray_Descr *type; 

  char transa = 'n';
  if (!PyArg_ParseTuple(args, "DnOnODnOO|c", &alpha, &a_gpu,&a_shape, &b_gpu,
			&b_shape, &beta, &c_gpu,&c_shape,&type,&transa))
    return NULL;
  int m, k, lda, ldb, ldc;

  int n = PyInt_AsLong(PyTuple_GetItem(b_shape,0));

  if (transa == 'n')
    {
      m = PyInt_AsLong(PyTuple_GetItem(a_shape,1));
      
      for (int i = 2; i < PyTuple_Size(a_shape); i++)
	m *= PyInt_AsLong(PyTuple_GetItem(a_shape,i));
      k = PyInt_AsLong(PyTuple_GetItem(a_shape,0));
      lda = m;
      ldb = k;
      ldc = m;
    }
  else
    {
      k = PyInt_AsLong(PyTuple_GetItem(a_shape,1));
      for (int i = 2; i < PyTuple_Size(a_shape); i++)
	k *= PyInt_AsLong(PyTuple_GetItem(a_shape,i));
      m = PyInt_AsLong(PyTuple_GetItem(a_shape,0));
      
      lda = k;
      ldb = k;
      ldc = m;
      
    }
  
  if (type->type_num == PyArray_DOUBLE)
    cublasDgemm(transa, 'n', m, n, k, 
		alpha.real,(double*)a_gpu ,lda, (double*)b_gpu, ldb, 
		beta.real, (double*)c_gpu, ldc);

  else {
    cuDoubleComplex alpha_gpu={alpha.real,alpha.imag};
    cuDoubleComplex beta_gpu={beta.real,beta.imag};
    cublasZgemm(transa, 'n', m, n, k, 
		alpha_gpu,
		(cuDoubleComplex*)a_gpu ,lda,
		(cuDoubleComplex*)b_gpu, ldb, 
		beta_gpu,
		(cuDoubleComplex*)c_gpu, ldc);
  }
  
  gpaw_cublasSafeCall(cublasGetError());
  Py_RETURN_NONE;
}


PyObject* gemv_cuda_gpu(PyObject *self, PyObject *args)
{
  Py_complex alpha;

  CUdeviceptr a_gpu;
  CUdeviceptr x_gpu;
  CUdeviceptr y_gpu;

  Py_complex beta;
  PyObject *a_shape,*x_shape;
  PyArray_Descr *type;

  char trans = 't';
  if (!PyArg_ParseTuple(args, "DnOnODn0|c", &alpha, &a_gpu,&a_shape, &x_gpu,&x_shape, &beta, &y_gpu,&type,&trans))
    return NULL;

  int m, n, lda, incx, incy;

  if (trans == 'n')
    {
      m = PyInt_AsLong(PyTuple_GetItem(a_shape,1));
      
      for (int i = 2; i < PyTuple_Size(a_shape); i++)
	m *= PyInt_AsLong(PyTuple_GetItem(a_shape,i));
      n = PyInt_AsLong(PyTuple_GetItem(a_shape,0));
      lda = m;
    }
  else
    {
      n = PyInt_AsLong(PyTuple_GetItem(a_shape,0));
      for (int i = 1; i < PyTuple_Size(a_shape)-1; i++)
	n *= PyInt_AsLong(PyTuple_GetItem(a_shape,i));
      m = PyInt_AsLong(PyTuple_GetItem(a_shape,PyTuple_Size(a_shape)-1));
      
      lda = m;

    }


  incx = 1;
  incy = 1;

  if (type->type_num == PyArray_DOUBLE)
    cublasDgemv(trans, m, n, 
		alpha.real,(double*)a_gpu ,lda, (double*)x_gpu, incx, 
		beta.real, (double*)y_gpu, incy);
  else{
    cuDoubleComplex alpha_gpu={alpha.real,alpha.imag};
    cuDoubleComplex beta_gpu={beta.real,beta.imag};
    cublasZgemv(trans, m, n, 
		alpha_gpu,
		(cuDoubleComplex*)a_gpu ,lda,
		(cuDoubleComplex*)x_gpu, incx, 
		beta_gpu,
		(cuDoubleComplex*)y_gpu, incy);
  }

  Py_RETURN_NONE;
}




PyObject* axpy_cuda_gpu(PyObject *self, PyObject *args)
{
  Py_complex alpha;

  CUdeviceptr x_gpu;
  CUdeviceptr y_gpu;
  PyObject *x_shape,*y_shape;
  PyArray_Descr *type; 


  if (!PyArg_ParseTuple(args, "DnOnOO", &alpha, &x_gpu,&x_shape, &y_gpu,
			&y_shape,&type))
    return NULL;

  int n = PyInt_AsLong(PyTuple_GetItem(x_shape,0));
  Py_ssize_t nd=PyTuple_Size(x_shape);
  for (int d = 1; d < nd; d++)
    n *= PyInt_AsLong(PyTuple_GetItem(x_shape,d));
  int incx = 1;
  int incy = 1;
  if (type->type_num == PyArray_DOUBLE)
    cublasDaxpy(n, alpha.real,
		(double*)x_gpu, incx,
		(double*)y_gpu, incy);
  else {
    cuDoubleComplex alpha_gpu={alpha.real,alpha.imag};
    cublasZaxpy(n, alpha_gpu,
		(cuDoubleComplex*)x_gpu, incx,
		(cuDoubleComplex*)y_gpu, incy);
  }
  gpaw_cublasSafeCall(cublasGetError());

  
  Py_RETURN_NONE;
}


PyObject* rk_cuda_gpu(PyObject *self, PyObject *args)
{
  double alpha;

  double beta;
  
  CUdeviceptr a_gpu;
  CUdeviceptr c_gpu;
  PyObject *a_shape,*c_shape;
  PyArray_Descr *type; 


  if (!PyArg_ParseTuple(args, "dnOdnOO", &alpha, &a_gpu,&a_shape, &beta, 
			&c_gpu,&c_shape,&type))
    return NULL;


  int n = PyInt_AsLong(PyTuple_GetItem(a_shape,0));
  int k = PyInt_AsLong(PyTuple_GetItem(a_shape,1));

  for (int d = 2; d < PyTuple_Size(a_shape); d++)
    k *= PyInt_AsLong(PyTuple_GetItem(a_shape,d));
  int ldc = n;
  if (type->type_num == PyArray_DOUBLE)
    cublasDsyrk('u', 't', n, k,
		alpha, (double*)a_gpu, k, beta,
		(double*)c_gpu, ldc);
  else {
    cublasZherk('u', 't', n, k,
		alpha, (cuDoubleComplex*)a_gpu, k,
		beta, (cuDoubleComplex*)c_gpu, ldc);
  }
  gpaw_cublasSafeCall(cublasGetError());

  Py_RETURN_NONE;
}


PyObject* r2k_cuda_gpu(PyObject *self, PyObject *args)
{
  Py_complex alpha;
  double beta;
  
  CUdeviceptr a_gpu;
  CUdeviceptr b_gpu;
  CUdeviceptr c_gpu;
  PyObject *a_shape,*b_shape,*c_shape;
  PyArray_Descr *type; 


  if (!PyArg_ParseTuple(args, "DnOnOdnOO", &alpha, &a_gpu,&a_shape,&b_gpu,
			&b_shape, &beta, &c_gpu,&c_shape,&type))
    return NULL;

  int n = PyInt_AsLong(PyTuple_GetItem(a_shape,0));
  int k = PyInt_AsLong(PyTuple_GetItem(a_shape,1));

  for (int d = 2; d < PyTuple_Size(a_shape); d++)
    k *= PyInt_AsLong(PyTuple_GetItem(a_shape,d));

  int ldc = n;
  
  if (type->type_num == PyArray_DOUBLE)
    cublasDsyr2k('u', 't', n, k,
		 alpha.real, (double*)a_gpu, k, (double*)b_gpu,k,beta,
		 (double*)c_gpu, ldc);
  else {
    cuDoubleComplex alpha_gpu={alpha.real,alpha.imag};
    cublasZher2k('u', 't', n, k,
		 alpha_gpu, 
		 (cuDoubleComplex*)a_gpu, k, 
		 (cuDoubleComplex*)b_gpu,k,
		 beta, 
		 (cuDoubleComplex*)c_gpu, ldc);
  }
  gpaw_cublasSafeCall(cublasGetError());


  Py_RETURN_NONE;
}


PyObject* dotc_cuda_gpu(PyObject *self, PyObject *args)
{
  CUdeviceptr a_gpu;
  CUdeviceptr b_gpu;
  
  PyObject *a_shape;
  PyArray_Descr *type;


  if (!PyArg_ParseTuple(args, "nOnO", &a_gpu,&a_shape,&b_gpu,&type))
    return NULL;

  int n = PyInt_AsLong(PyTuple_GetItem(a_shape,0));
  
  for (int i = 1; i < PyTuple_Size(a_shape); i++)
    n *= PyInt_AsLong(PyTuple_GetItem(a_shape,i));

  int incx = 1;
  int incy = 1;
  if (type->type_num == PyArray_DOUBLE)
    {
      double result;
      result = cublasDdot(n, (double*)a_gpu,
			  incx, (double*)b_gpu, incy);
      gpaw_cublasSafeCall(cublasGetError());
      return PyFloat_FromDouble(result);
    }
  else
    {
      cuDoubleComplex result;
      result = cublasZdotc(n, (cuDoubleComplex*)a_gpu,
			  incx, (cuDoubleComplex*)b_gpu, incy);
      gpaw_cublasSafeCall(cublasGetError());
      return PyComplex_FromDoubles(result.x,result.y);
    }

}


PyObject* dotu_cuda_gpu(PyObject *self, PyObject *args)
{
  CUdeviceptr a_gpu;
  CUdeviceptr b_gpu;
  
  PyObject *a_shape;
  PyArray_Descr *type;


  if (!PyArg_ParseTuple(args, "nOnO", &a_gpu,&a_shape,&b_gpu,&type))
    return NULL;
  int n = PyInt_AsLong(PyTuple_GetItem(a_shape,0));
  
  for (int i = 1; i < PyTuple_Size(a_shape); i++)
    n *= PyInt_AsLong(PyTuple_GetItem(a_shape,i));

  int incx = 1;
  int incy = 1;
  if (type->type_num == PyArray_DOUBLE)
    {
      double result;
      result = cublasDdot(n, (double*)a_gpu,
			  incx, (double*)b_gpu, incy);
      gpaw_cublasSafeCall(cublasGetError());
      return PyFloat_FromDouble(result);
    }
  else
    {
      cuDoubleComplex result;
      result = cublasZdotu(n, (cuDoubleComplex*)a_gpu,
			  incx, (cuDoubleComplex*)b_gpu, incy);
      gpaw_cublasSafeCall(cublasGetError());
      return PyComplex_FromDoubles(result.x,result.y);
    }
}

}

#endif
