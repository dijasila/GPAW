#include <string.h>
#include <assert.h>
#include "../extensions.h"
#include "gpaw-cuda-int.h"
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

void bc_init_buffers_cuda();
void blas_init_cuda();
void transformer_init_buffers_cuda();
void operator_init_buffers_cuda();
void reduce_init_buffers_cuda();
void lfc_reduce_init_buffers_cuda();
void bc_dealloc_cuda(int force);
void transformer_dealloc_cuda(int force);
void operator_dealloc_cuda(int force);
void reduce_dealloc_cuda();
void lfc_reduce_dealloc_cuda();

struct cudaDeviceProp _gpaw_cuda_dev_prop;
int _gpaw_cuda_dev;

void gpaw_cuda_init_c() 
{
  gpaw_cuSCall(cudaGetDevice(&_gpaw_cuda_dev));
  gpaw_cuSCall(cudaGetDeviceProperties(&_gpaw_cuda_dev_prop, _gpaw_cuda_dev));
  
  bc_init_buffers_cuda();
  transformer_init_buffers_cuda();
  operator_init_buffers_cuda();
  reduce_init_buffers_cuda();
  lfc_reduce_init_buffers_cuda();
  blas_init_cuda();
}

PyObject* gpaw_cuda_init(PyObject *self, PyObject *args)
{
  if (!PyArg_ParseTuple(args, ""))
    return NULL;

  gpaw_cuda_init_c();

  if (PyErr_Occurred())
    return NULL;
  else
    Py_RETURN_NONE;
}

PyObject* gpaw_cuda_delete(PyObject *self, PyObject *args)
{

  if (!PyArg_ParseTuple(args, ""))
    return NULL;
  reduce_dealloc_cuda();
  lfc_reduce_dealloc_cuda();
  bc_dealloc_cuda(1);
  transformer_dealloc_cuda(1);
  operator_dealloc_cuda(1);  
  if (PyErr_Occurred())
    return NULL;
  else
    Py_RETURN_NONE;
}

