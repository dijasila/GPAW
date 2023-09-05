#include "../extensions.h"

#define GPAW_ARRAY_DISABLE_NUMPY
#define GPAW_ARRAY_ALLOW_CUPY
#include "../array.h"
#undef GPAW_ARRAY_DISABLE_NUMPY

#include "gpu.h"
#include "gpu-complex.h"

void pwlfc_expand_gpu_launch_kernel(int itemsize,
                                    double* f_Gs,
                                    gpuDoubleComplex *emiGR_Ga,
                                    double *Y_GL,
                                    int* l_s,
                                    int* a_J,
                                    int* s_J,
                                    double* f_GI,
                                    int* I_J,
                                    int nG,
                                    int nJ,
                                    int nL,
                                    int nI,
                                    int natoms,
                                    int nsplines,
                                    bool cc);

void pw_insert_gpu_launch_kernel(int itemsize,
                             int nb,
                             int nG,
                             int nQ,
                             double* c_nG,
                             npy_int32* Q_G,
                             double scale,
                             double* tmp_nQ);

PyObject* pwlfc_expand_gpu(PyObject* self, PyObject* args)
{
    PyObject *f_Gs_obj;
    PyObject *emiGR_Ga_obj;
    PyObject *Y_GL_obj;
    PyObject *l_s_obj;
    PyObject *a_J_obj;
    PyObject *s_J_obj;
    int cc;
    PyObject *f_GI_obj;
    PyObject *I_J_obj;

    if (!PyArg_ParseTuple(args, "OOOOOOiOO",
                          &f_Gs_obj, &emiGR_Ga_obj, &Y_GL_obj,
                          &l_s_obj, &a_J_obj, &s_J_obj,
                          &cc, &f_GI_obj, &I_J_obj))
        return NULL;
    double *f_Gs = (double*)Array_DATA(f_Gs_obj);
    double *Y_GL = (double*)Array_DATA(Y_GL_obj);
    int *l_s = (int*)Array_DATA(l_s_obj);
    int *a_J = (int*)Array_DATA(a_J_obj);
    int *s_J = (int*)Array_DATA(s_J_obj);
    double *f_GI = (double*)Array_DATA(f_GI_obj);
    int nG = Array_DIM(emiGR_Ga_obj, 0);
    int *I_J = (int*)Array_DATA(I_J_obj);
    int nJ = Array_DIM(a_J_obj, 0);
    int nL = Array_DIM(Y_GL_obj, 1);
    int nI = Array_DIM(f_GI_obj, 1);
    int natoms = Array_DIM(emiGR_Ga_obj, 1);
    int nsplines = Array_DIM(f_Gs_obj, 1);
    gpuDoubleComplex* emiGR_Ga = (gpuDoubleComplex*)Array_DATA(emiGR_Ga_obj);
    int itemsize = Array_ITEMSIZE(f_GI_obj);
    pwlfc_expand_gpu_launch_kernel(itemsize, f_Gs, emiGR_Ga, Y_GL, l_s, a_J, s_J, f_GI,
                                   I_J, nG, nJ, nL, nI, natoms, nsplines, cc);
    gpuDeviceSynchronize(); // Is needed?
    Py_RETURN_NONE;
}


PyObject* pw_insert_gpu(PyObject* self, PyObject* args)
{
    PyObject *c_nG_obj, *Q_G_obj, *tmp_nQ_obj;
    double scale;
    if (!PyArg_ParseTuple(args, "OOdO",
                          &c_nG_obj, &Q_G_obj, &scale, &tmp_nQ_obj))
        return NULL;
    double complex *c_nG = Array_DATA(c_nG_obj);
    npy_int32 *Q_G = Array_DATA(Q_G_obj);
    double complex *tmp_nQ = Array_DATA(tmp_nQ_obj);
    int nG = 0;
    int nQ = 0;
    int nb = 0;
    assert(Array_NDIM(c_nG_obj) == Array_NDIM(tmp_nQ_obj));
    assert(Array_ITEMSIZE(c_nG_obj) == 16);
    assert(Array_ITEMSIZE(tmp_nQ_obj) == 16);
    if (Array_NDIM(c_nG_obj) == 1)
    {
        nG = Array_DIM(c_nG_obj, 0);
        nb = 1;
        nQ = Array_DIM(tmp_nQ_obj, 1);
    }
    else
    {
        nG = Array_DIM(c_nG_obj, 1);
        nb = Array_DIM(c_nG_obj, 0);
        nQ = Array_DIM(tmp_nQ_obj, 0);
    }

    pw_insert_gpu_launch_kernel(16, nb, nG, nQ, c_nG, Q_G, scale, tmp_nQ);
    Py_RETURN_NONE;
}

