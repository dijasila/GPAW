#include <string.h>
#include <assert.h>
#include "../extensions.h"
#include "gpu.h"
#include "gpu-complex.h"
#include <stdio.h>
#include <stdlib.h>

void bc_init_buffers_gpu();
void blas_init_gpu();
void transformer_init_buffers_gpu();
void operator_init_buffers_gpu();
void reduce_init_buffers_gpu();
void lfc_reduce_init_buffers_gpu();
void bc_dealloc_gpu(int force);
void transformer_dealloc_gpu(int force);
void operator_dealloc_gpu(int force);
void reduce_dealloc_gpu();
void lfc_reduce_dealloc_gpu();

struct gpuDeviceProp _gpaw_gpu_dev_prop;
int gpaw_cuda_debug=0;  // if true, debug CUDA kernels

PyObject* set_gpaw_gpu_debug(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "p", &gpaw_cuda_debug))
        return NULL;
    Py_RETURN_NONE;
}

void gpaw_gpu_init_c()
{
    int device;
    gpuGetDevice(&device);
    gpuGetDeviceProperties(&_gpaw_gpu_dev_prop, device);

    bc_init_buffers_gpu();
    transformer_init_buffers_gpu();
    operator_init_buffers_gpu();
    reduce_init_buffers_gpu();
    lfc_reduce_init_buffers_gpu();
    blas_init_gpu();
}

PyObject* gpaw_gpu_init(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    gpaw_gpu_init_c();

    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

PyObject* gpaw_gpu_delete(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    reduce_dealloc_gpu();
    lfc_reduce_dealloc_gpu();
    bc_dealloc_gpu(1);
    transformer_dealloc_gpu(1);
    operator_dealloc_gpu(1);

    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}
