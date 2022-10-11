#include <string.h>
#include <assert.h>
#include "../extensions.h"
#include "gpu.h"
#include "gpu-complex.h"
#include <stdio.h>
#include <stdlib.h>

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

struct gpuDeviceProp _gpaw_gpu_dev_prop;
int gpaw_cuda_debug=0;  // if true, debug CUDA kernels

PyObject* set_gpaw_cuda_debug(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "p", &gpaw_cuda_debug))
        return NULL;
    Py_RETURN_NONE;
}

void gpaw_cuda_init_c()
{
    int device;
    gpuGetDevice(&device);
    gpuGetDeviceProperties(&_gpaw_gpu_dev_prop, device);

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
