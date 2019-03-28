#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gpaw-cuda-int.h>

#ifndef CUGPAWCOMPLEX
#  define BLOCK_X  128
#  define MAX_BLOCKS  (65535)
#endif

__global__ void axpbyz_kernel(double a, double *x, double b, double *y,
                              double *z, unsigned long n)
{
    unsigned tid = threadIdx.x;
    unsigned threads = gridDim.x*blockDim.x;
    unsigned start = blockDim.x*blockIdx.x;
    unsigned i;

    for (i = start + tid; i < n; i += threads) {
        z[i] = a*x[i] + b*y[i];
    }
}

__global__ void axpbz_kernel(double a, double *x, double b,
                             double *z, unsigned long n)
{
    unsigned tid = threadIdx.x;
    unsigned threads = gridDim.x*blockDim.x;
    unsigned start = blockDim.x*blockIdx.x;
    unsigned i;

    for (i = start + tid; i < n; i += threads) {
        z[i] = a*x[i] + b;
    }
}

__global__ void axpbyz_kernelz(double a, cuDoubleComplex *x,
                               double b, cuDoubleComplex *y,
                               cuDoubleComplex *z, unsigned long n)
{
    unsigned tid = threadIdx.x;
    unsigned threads = gridDim.x*blockDim.x;
    unsigned start = blockDim.x*blockIdx.x;
    unsigned i;

    for (i = start + tid; i < n; i += threads) {
        (z[i]).x = a * cuCreal(x[i]) + b * cuCreal(y[i]);
        (z[i]).y = a * cuCimag(x[i]) + b * cuCimag(y[i]);
    }
}

__global__ void axpbz_kernelz(double a, cuDoubleComplex *x, double b,
                              cuDoubleComplex *z, unsigned long n)
{
    unsigned tid = threadIdx.x;
    unsigned threads = gridDim.x*blockDim.x;
    unsigned start = blockDim.x*blockIdx.x;
    unsigned i;

    for (i = start + tid; i < n; i += threads) {
        (z[i]).x = a * cuCreal(x[i]) + b;
        (z[i]).y = a * cuCimag(x[i]) + b;
    }
}

__global__ void fill_kernel(double a, double *z, unsigned long n)
{
    unsigned tid = threadIdx.x;
    unsigned threads = gridDim.x*blockDim.x;
    unsigned start = blockDim.x*blockIdx.x;
    unsigned i;

    for (i = start + tid; i < n; i += threads) {
        z[i] = a;
    }
}

__global__ void fill_kernelz(double a, cuDoubleComplex *z, unsigned long n)
{
    unsigned tid = threadIdx.x;
    unsigned threads = gridDim.x*blockDim.x;
    unsigned start = blockDim.x*blockIdx.x;
    unsigned i;

    for (i = start + tid; i < n; i += threads) {
        (z[i]).x = a;
        (z[i]).y = 0;
    }
}

extern "C" {
    PyObject* axpbyz_gpu(PyObject *self, PyObject *args)
    {
        double a, b;
        CUdeviceptr x, y, z;
        PyObject *shape;
        PyArray_Descr *type;

        if (!PyArg_ParseTuple(args, "dndnnOO", &a, &x, &b, &y, &z, &shape,
                              &type))
            return NULL;

        int n = 1;
        Py_ssize_t nd = PyTuple_Size(shape);
        for (int d=0; d < nd; d++)
            n *= PyInt_AsLong(PyTuple_GetItem(shape, d));

        int gridx = MIN(MAX((n + BLOCK_X - 1) / BLOCK_X, 1), MAX_BLOCKS);

        dim3 dimBlock(BLOCK_X, 1);
        dim3 dimGrid(gridx, 1);
        if (type->type_num == PyArray_DOUBLE) {
            axpbyz_kernel<<<dimGrid, dimBlock, 0>>>
                (a, (double*) x, b, (double*) y, (double *) z, n);

        } else {
            axpbyz_kernelz<<<dimGrid, dimBlock, 0>>>
                (a, (cuDoubleComplex*) x, b, (cuDoubleComplex*) y,
                 (cuDoubleComplex*) z, n);
        }
        gpaw_cudaSafeCall(cudaGetLastError());
        if (PyErr_Occurred())
            return NULL;
        else
            Py_RETURN_NONE;
    }

    PyObject* axpbz_gpu(PyObject *self, PyObject *args)
    {
        double a, b;
        CUdeviceptr x, z;
        PyObject *shape;
        PyArray_Descr *type;

        if (!PyArg_ParseTuple(args, "dndnnOO", &a, &x, &b, &z, &shape,
                              &type))
            return NULL;

        int n = 1;
        Py_ssize_t nd = PyTuple_Size(shape);
        for (int d=0; d < nd; d++)
            n *= PyInt_AsLong(PyTuple_GetItem(shape, d));

        int gridx = MIN(MAX((n + BLOCK_X - 1) / BLOCK_X, 1), MAX_BLOCKS);

        dim3 dimBlock(BLOCK_X, 1);
        dim3 dimGrid(gridx, 1);
        if (type->type_num == PyArray_DOUBLE) {
            axpbz_kernel<<<dimGrid, dimBlock, 0>>>
                (a, (double*) x, b, (double *) z, n);

        } else {
            axpbz_kernelz<<<dimGrid, dimBlock, 0>>>
                (a, (cuDoubleComplex*) x, b, (cuDoubleComplex*) z, n);
        }
        gpaw_cudaSafeCall(cudaGetLastError());
        if (PyErr_Occurred())
            return NULL;
        else
            Py_RETURN_NONE;
    }

    PyObject* fill_gpu(PyObject *self, PyObject *args)
    {
        double a;
        CUdeviceptr z;
        PyObject *shape;
        PyArray_Descr *type;

        if (!PyArg_ParseTuple(args, "dnOO", &a, &z, &shape, &type))
            return NULL;

        int n = 1;
        Py_ssize_t nd = PyTuple_Size(shape);
        for (int d=0; d < nd; d++)
            n *= PyInt_AsLong(PyTuple_GetItem(shape, d));

        int gridx = MIN(MAX((n + BLOCK_X - 1) / BLOCK_X, 1), MAX_BLOCKS);

        dim3 dimBlock(BLOCK_X, 1);
        dim3 dimGrid(gridx, 1);
        if (type->type_num == PyArray_DOUBLE) {
            fill_kernel<<<dimGrid, dimBlock, 0>>>
                (a, (double*) z, n);

        } else {
            fill_kernelz<<<dimGrid, dimBlock, 0>>>
                (a, (cuDoubleComplex*) z, n);
        }
        gpaw_cudaSafeCall(cudaGetLastError());
        if (PyErr_Occurred())
            return NULL;
        else
            Py_RETURN_NONE;
    }
}
