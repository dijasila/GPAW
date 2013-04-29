#include <string.h>
#include <assert.h>
#include "../extensions.h"
#include "gpaw-cuda-int.h"
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

void bc_init_buffers_cuda();
void transformer_init_buffers_cuda();
void operator_init_buffers_cuda();
void reduce_init_buffers_cuda();
void lfc_reduce_init_buffers_cuda();
void reduce_dealloc_cuda();
void lfc_reduce_dealloc_cuda();

PyObject* gpaw_cuda_init(PyObject *self, PyObject *args)
{
  if (!PyArg_ParseTuple(args, ""))
    return NULL;

  bc_init_buffers_cuda();
  transformer_init_buffers_cuda();
  operator_init_buffers_cuda();
  reduce_init_buffers_cuda();
  lfc_reduce_init_buffers_cuda();

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
  
  if (PyErr_Occurred())
    return NULL;
  else
    Py_RETURN_NONE;
}

