#include "gpaw_gpu.h"

#include <stdio.h>
#include <numpy/arrayobject.h>
#include <complex.h>

#include "hip_kernels.h"

static PyMethodDef gpaw_gpu_module_functions[] = { {"pwlfc_expand_gpu", pwlfc_expand_gpu, METH_VARARGS, 0},
                                                   { 0,0,0,0 } };

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_gpaw_gpu",
    "HIP GPU kernels for GPAW",
    -1,
    gpaw_gpu_module_functions,
    NULL,
    NULL,
    NULL,
    NULL
};

double* CuPyArray_DATA(PyObject* obj)
{
    PyObject* data_str = Py_BuildValue("s", "data");
    PyObject* ndarray_data = PyObject_GetAttr(obj, data_str);
    Py_DECREF(data_str);
    if (ndarray_data == NULL) return NULL;
    PyObject* ptr_str = Py_BuildValue("s", "ptr");
    PyObject* ptr_data = PyObject_GetAttr(ndarray_data, ptr_str);
    Py_DECREF(ptr_str);
    if (ptr_data == NULL) return NULL;
    return (void*) PyLong_AS_LONG(ptr_data);
}

int CuPyArray_DIM(PyObject* obj, int dim)
{
    PyObject* shape_str = Py_BuildValue("s", "shape");
    PyObject* shape = PyObject_GetAttr(obj, shape_str);
    Py_DECREF(shape_str);
    if (shape == NULL) return -1;
    PyObject* pydim = PyTuple_GetItem(shape, dim);
    if (pydim == NULL) return -1;
    return (int) PyLong_AS_LONG(pydim);
}

int CuPyArray_ITEMSIZE(PyObject* obj)
{
    PyObject* itemsize_str = Py_BuildValue("s", "itemsize");
    int itemsize = (int) PyLong_AS_LONG(PyObject_GetAttr(obj, itemsize_str));
    Py_DECREF(itemsize_str);
    return itemsize;
}

PyObject* pwlfc_expand_gpu(PyObject* self, PyObject* args)
{
    PyArrayObject *f_Gs_obj;
    PyArrayObject *emiGR_Ga_obj;
    PyArrayObject *Y_GL_obj;
    PyArrayObject *l_s_obj;
    PyArrayObject *a_J_obj;
    PyArrayObject *s_J_obj;
    int cc;
    PyArrayObject *f_GI_obj;
    PyArrayObject *I_J_obj;

    if (!PyArg_ParseTuple(args, "OOOOOOiOO",
                          &f_Gs_obj, &emiGR_Ga_obj, &Y_GL_obj,
                          &l_s_obj, &a_J_obj, &s_J_obj,
                          &cc, &f_GI_obj, &I_J_obj))
        return NULL;

    double *f_Gs = (double*) CuPyArray_DATA(f_Gs_obj);
    double complex* emiGR_Ga = CuPyArray_DATA(emiGR_Ga_obj);
    double *Y_GL = CuPyArray_DATA(Y_GL_obj);
    npy_int32 *l_s = CuPyArray_DATA(l_s_obj);
    npy_int32 *a_J = CuPyArray_DATA(a_J_obj);
    npy_int32 *s_J = CuPyArray_DATA(s_J_obj);
    double *f_GI = CuPyArray_DATA(f_GI_obj);
    npy_int32 *I_J = CuPyArray_DATA(I_J_obj);

    int nG = CuPyArray_DIM(emiGR_Ga_obj, 0);
    int nJ = CuPyArray_DIM(a_J_obj, 0);
    int nL = CuPyArray_DIM(Y_GL_obj, 1);
    int nI = CuPyArray_DIM(f_GI_obj, 1);
    int natoms = CuPyArray_DIM(emiGR_Ga_obj, 1);
    int nsplines = CuPyArray_DIM(f_Gs_obj, 1);
    int itemsize = CuPyArray_ITEMSIZE(f_GI_obj);

    pwlfc_expand_gpu_launch_kernel(itemsize, f_Gs, emiGR_Ga, Y_GL, l_s, a_J, s_J, f_GI,
                                       I_J, nG, nJ, nL, nI, natoms, nsplines, cc);
    hipDeviceSynchronize(); // Is needed?
    Py_RETURN_NONE;
}

static PyObject* moduleinit(void)
{
    PyObject* m = PyModule_Create(&moduledef);

    if (m == NULL)
        return NULL;

    return m;
}

PyMODINIT_FUNC PyInit__gpaw_gpu(void)
{
    return moduleinit();
}
