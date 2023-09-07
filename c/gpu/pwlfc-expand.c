#include "../extensions.h"

#define GPAW_ARRAY_DISABLE_NUMPY
#define GPAW_ARRAY_ALLOW_CUPY
#include "../array.h"
#undef GPAW_ARRAY_DISABLE_NUMPY

#include "gpu.h"
#include "gpu-complex.h"
#include <stdio.h>

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

void add_to_density_gpu_launch_kernel(int nb,
                                      int nR,
                                      double* f_n,
                                      double complex* psit_nR,
                                      double* rho_R);

void multiple_low_rank_rect_sqr_rect_updates_launch_kernel(int nN, int nA, int* ni_a, gpuDoubleComplex** P_ani, double** H_aii, gpuDoubleComplex* H_nn);

PyObject* dH_aii_times_P_ani_gpu(PyObject* self, PyObject* args)
{
    PyObject* dH_aii_obj;
    PyObject* ni_a_obj;
    PyObject* P_ani_obj;
    PyObject* outP_ani_obj;
    if (!PyArg_ParseTuple(args, "OOOO",
                          &dH_aii_obj, &ni_a_obj, &P_ani_obj, &outP_ani_obj))
        return NULL;

    printf("ni_a"); print_array_info(ni_a_obj);
    printf("dH_aii"); print_array_info(dH_aii_obj);
    printf("P_ani"); print_array_info(P_ani_obj);
    printf("outP_ani"); print_array_info(outP_ani_obj);
    
    int nA = Array_DIM(ni_a_obj, 0);
    int nn = Array_DIM(P_ani_obj, 0);
    int nI = Array_DIM(P_ani_obj, 1);
    fflush(stdout);
    fprintf("nA = %d nn = %d nI = %d\n", nA, nn, nI);
    
    double* dH_aii_dev = Array_DATA(dH_aii_obj);
    gpuDoubleComplex* P_ani_dev = Array_DATA(P_ani_obj);
    gpuDoubleComplex* outP_ani_dev = Array_DATA(outP_ani_obj);
    npy_int32* ni_a = Array_DATA(ni_a_obj);

    dH_aii_times_P_ani_launch_kernel(nA, nn, nI, ni_a, dH_aii_dev, P_ani_dev, outP_ani_dev);
}

PyObject* multple_low_rank_rect_sqr_rect_updates_gpu(PyObject* self, PyObject* args)
{
    // H_nn, P_ani, H_aii
    PyObject *H_nn_obj;
    PyObject *P_ani_obj, *H_aii_obj;
    if (!PyArg_ParseTuple(args, "OOO",
                          &H_nn_obj, &P_ani_obj, &H_aii_obj))
        return NULL;
    // Total size of H_nn matrix
    int nN = Array_DIM(H_nn_obj, 0);
    printf("%d nN\n", nN);
    // Number of atoms
    int nA = Py_SIZE(P_ani_obj);
    // Check that number of atoms is the same for H_aii array as well
    assert(nA == Py_SIZE(H_aii_obj));

    // Allocate atom pointer arrays
    int* ni_a_host;
    int* ni_a_dev;
    hipMalloc((void**) &ni_a_dev, sizeof(int) * nA);
    ni_a_host = (int*) malloc(sizeof(int) * nA);

    gpuDoubleComplex** P_ani_host;
    gpuDoubleComplex** P_ani_dev;
    hipMalloc((void**) &P_ani_dev, sizeof(gpuDoubleComplex*) * nA);
    P_ani_host = (gpuDoubleComplex**) malloc(sizeof(gpuDoubleComplex*) * nA);

    double** H_aii_host;
    double** H_aii_dev;
    hipMalloc((void**) &H_aii_dev, sizeof(double*) * nA);
    H_aii_host = (double**) malloc(sizeof(gpuDoubleComplex*) * nA);

    for (int a=0; a<nA; a++)
    {
       PyObject* H_ii = PyList_GetItem(H_aii_obj, a);
       PyObject* P_ni = PyList_GetItem(P_ani_obj, a);
       ni_a_host[a] = Array_DIM(H_ii, 0);
       P_ani_host[a] = Array_DATA(P_ni);
       H_aii_host[a] = Array_DATA(H_ii);
    }

    hipMemcpy(ni_a_dev, ni_a_host, sizeof(int) * nA, hipMemcpyHostToDevice);
    hipMemcpy(P_ani_dev, P_ani_host, sizeof(gpuDoubleComplex*) * nA, hipMemcpyHostToDevice);
    hipMemcpy(H_aii_dev, H_aii_host, sizeof(double*) * nA, hipMemcpyHostToDevice);

    gpuDoubleComplex* H_nn_dev = Array_DATA(H_nn_obj);
    gpuDeviceSynchronize(); // Is needed?

    multiple_low_rank_rect_sqr_rect_updates_launch_kernel(nN, nA, ni_a_dev, P_ani_dev, H_aii_dev, H_nn_dev);
    gpuDeviceSynchronize(); // Is needed?
    hipFree(H_aii_dev);
    hipFree(P_ani_dev);
    hipFree(ni_a_dev);
    free(H_aii_host);
    free(P_ani_host);
    free(ni_a_host);
    Py_RETURN_NONE;
}


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
    npy_int32 *Q_G = Array_DATA(Q_G_obj);
    double complex *c_nG = Array_DATA(c_nG_obj);
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
        nQ = Array_DIM(tmp_nQ_obj, 0);
    }
    else
    {
        nG = Array_DIM(c_nG_obj, 1);
        nb = Array_DIM(c_nG_obj, 0);
        nQ = Array_DIM(tmp_nQ_obj, 1);
    }

    pw_insert_gpu_launch_kernel(16, nb, nG, nQ, c_nG, Q_G, scale, tmp_nQ);
    Py_RETURN_NONE;
}



PyObject* add_to_density_gpu(PyObject* self, PyObject* args)
{
    PyObject *f_n_obj, *psit_nR_obj, *rho_R_obj;
    if (!PyArg_ParseTuple(args, "OOO",
                          &f_n_obj, &psit_nR_obj, &rho_R_obj))
        return NULL;
    double *f_n = Array_DATA(f_n_obj);
    double complex *psit_nR = Array_DATA(psit_nR_obj);
    double* rho_R = Array_DATA(rho_R_obj);
    int nb = Array_SIZE(f_n_obj);
    int nR = Array_SIZE(psit_nR_obj) / nb;
    assert(Array_ITEMSIZE(psit_nR_obj) == 16);
    //assert(Array_ITEMSIZE(rho_nR_obj) == 8);
    add_to_density_gpu_launch_kernel(nb, nR, f_n, psit_nR, rho_R);
    Py_RETURN_NONE;
}

