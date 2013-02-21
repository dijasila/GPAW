#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>

PyObject* cufft_plan3d(PyObject *self, PyObject *args)
{
  cufftHandle plan;
  cufftType ffttype = CUFFT_Z2Z;
  int nx, ny, nz;

  if (!PyArg_ParseTuple(args, "iii|s", &nx, &ny, &nz, &ffttype))
    return NULL;
  cufftPlan3d(&plan, nx, ny, nz, ffttype);

  return Py_BuildValue("L",plan);
}

PyObject* cufft_planmany(PyObject *self, PyObject *args)
{
  cufftHandle plan;
  cufftType ffttype = CUFFT_Z2Z;
  int nx, ny, nz, batch;

  if (!PyArg_ParseTuple(args, "iiii|s", &nx, &ny, &nz, &batch, &ffttype))
    return NULL;
  int n[3] = {nx, ny, nz};
  cufftPlanMany(&plan, 3, n, 
		NULL, 1, nx*ny*nz,
		NULL, 1, nx*ny*nz,
		ffttype, batch);

  return Py_BuildValue("L",plan);
}


PyObject* cufft_execZ2Z(PyObject *self, PyObject *args)
{
  cufftHandle plan;
  int fftdirection; /* forward: -1 ; inverse: 1*/
  void *dev_a, *dev_b; 

  if (!PyArg_ParseTuple(args, "LLLi",&plan, &dev_a, &dev_b, &fftdirection))
    return NULL;

  cufftExecZ2Z( plan, dev_a, dev_b, fftdirection );

  Py_RETURN_NONE;
}

PyObject* cufft_destroy(PyObject *self, PyObject *args)
{
  cufftHandle plan;

  if (!PyArg_ParseTuple(args, "L",&plan))
    return NULL;

  cufftDestroy(plan);

  Py_RETURN_NONE;
}



